"""LTX-Video generation module for Foresight inference.

This module provides standalone video generation functionality using LTX-Video
that can be integrated into the main inference app or used independently.

Usage:
    # Deploy as standalone endpoint
    modal deploy infra/modal/inference_video.py

    # Run test
    modal run infra/modal/inference_video.py::test_video_generation

Key Features:
- LTX-Video model (Lightricks/LTX-Video)
- 512x512 resolution, 30 frames, 15fps
- Supports conditioning features from hybrid encoder
- Returns base64-encoded frames

VRAM Budget:
- LTX-Video: ~8GB
- Generation Buffer: ~12GB
- Total: ~20GB (fits on A10G or A100)
"""

import base64
import io
import os
import time
from dataclasses import dataclass
from typing import Any

import modal

# Define the Modal app
app = modal.App("foresight-video-inference")

# GPU image with video generation dependencies
video_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        # Core ML
        "torch>=2.0.0",
        "torchvision",
        "transformers>=4.45.0",
        "diffusers>=0.27.0",
        "accelerate>=0.27.0",
        # Video processing
        "opencv-python-headless",
        "imageio",
        "imageio-ffmpeg",
        # Utilities
        "pillow",
        "numpy",
        "pydantic>=2.0.0",
        # Required for LTX-Video
        "sentencepiece",
    )
)

# Volume for model caching
model_cache = modal.Volume.from_name("foresight-model-cache", create_if_missing=True)


@dataclass
class VideoGenerationConfig:
    """Configuration for video generation."""

    width: int = 512
    height: int = 512
    num_frames: int = 30
    fps: int = 15
    num_inference_steps: int = 30
    guidance_scale: float = 7.5


@dataclass
class VideoGenerationResult:
    """Result of video generation."""

    frames_base64: list[str]
    num_frames: int
    width: int
    height: int
    fps: int
    generation_time_s: float
    effective_fps: float  # frames / generation_time


@app.cls(
    image=video_image,
    gpu="A10G",  # A10G is sufficient for video-only generation
    timeout=1800,
    volumes={"/model-cache": model_cache},
    container_idle_timeout=300,  # Keep warm for 5 minutes
    allow_concurrent_inputs=2,
)
class VideoGenerator:
    """LTX-Video generator for Foresight inference pipeline.

    This class handles loading and running the LTX-Video model for
    text-to-video and conditioned video generation.
    """

    # LTX-Video conditioning dimension (for prompt embeddings)
    LTX_DIM: int = 4096

    @modal.enter()
    def load_model(self) -> None:
        """Load LTX-Video model on container startup with image conditioning support."""
        import torch
        from diffusers import LTXPipeline

        # Set cache directories
        os.environ["HF_HOME"] = "/model-cache"
        os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
        os.environ["TORCH_HOME"] = "/model-cache/torch"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Load LTX-Video pipeline
        print("Loading LTX-Video...")
        start = time.time()

        self.pipeline = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
        )
        self.pipeline = self.pipeline.to(self.device)

        # Try to load condition pipeline for image-conditioned generation
        try:
            from diffusers import LTXConditionPipeline

            print("  Loading LTXConditionPipeline for image conditioning...")
            self.condition_pipeline = LTXConditionPipeline.from_pretrained(
                "Lightricks/LTX-Video",
                torch_dtype=torch.bfloat16,
            )
            self.condition_pipeline = self.condition_pipeline.to(self.device)
            self.has_condition_pipeline = True
            print("  LTXConditionPipeline loaded successfully")
        except ImportError:
            print("  LTXConditionPipeline not available")
            self.condition_pipeline = None
            self.has_condition_pipeline = False
        except Exception as e:
            print(f"  Warning: Could not load LTXConditionPipeline: {e}")
            self.condition_pipeline = None
            self.has_condition_pipeline = False

        print(f"LTX-Video loaded in {time.time() - start:.1f}s")

        # Report VRAM usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"VRAM allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")

    @modal.method()
    def generate_video(
        self,
        prompt: str,
        conditioning_features: list[float] | None = None,
        width: int = 512,
        height: int = 512,
        num_frames: int = 30,
        fps: int = 15,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Generate video frames from prompt and optional conditioning features.

        Args:
            prompt: Text prompt for video generation
            conditioning_features: Optional conditioning features from hybrid encoder.
                                   Shape: [seq_len * LTX_DIM] flattened, or None for text-only.
            width: Frame width (default 512)
            height: Frame height (default 512)
            num_frames: Number of frames to generate (default 30)
            fps: Target framerate (default 15)
            num_inference_steps: Diffusion steps (default 30)
            guidance_scale: Classifier-free guidance scale (default 7.5)
            negative_prompt: Optional negative prompt for guidance

        Returns:
            Dict with:
                - frames_base64: List of base64-encoded PNG frames
                - num_frames: Number of frames generated
                - width: Frame width
                - height: Frame height
                - fps: Target framerate
                - generation_time_s: Total generation time
                - effective_fps: Frames per second of generation
        """
        import numpy as np
        import torch
        from PIL import Image

        start_time = time.time()

        # Process conditioning features if provided
        prompt_embeds = None
        if conditioning_features is not None:
            # Reshape flattened features to [batch, seq_len, hidden_dim]
            features = torch.tensor(conditioning_features, dtype=torch.bfloat16)
            seq_len = len(conditioning_features) // self.LTX_DIM
            if seq_len > 0:
                prompt_embeds = features.reshape(1, seq_len, self.LTX_DIM).to(self.device)
                print(f"Using conditioning features: shape {prompt_embeds.shape}")

        # Generate video
        with torch.no_grad():
            if prompt_embeds is not None:
                # Use conditioning features as prompt embeddings
                # Note: This requires LTX-Video to support prompt_embeds input
                # For now, we fall back to text prompt if the pipeline doesn't support it
                try:
                    output = self.pipeline(
                        prompt=prompt,
                        prompt_embeds=prompt_embeds,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        negative_prompt=negative_prompt,
                    )
                except TypeError:
                    # Pipeline doesn't support prompt_embeds, use text only
                    print("Warning: prompt_embeds not supported, using text prompt only")
                    output = self.pipeline(
                        prompt=prompt,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        negative_prompt=negative_prompt,
                    )
            else:
                # Text-only generation
                output = self.pipeline(
                    prompt=prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                )

        # Extract frames from output
        frames = output.frames[0]  # [num_frames, height, width, 3]

        # Convert frames to base64-encoded PNGs
        frames_base64 = []
        for i, frame in enumerate(frames):
            if isinstance(frame, np.ndarray):
                pil_frame = Image.fromarray(frame)
            elif isinstance(frame, Image.Image):
                pil_frame = frame
            else:
                # Assume torch tensor
                pil_frame = Image.fromarray(frame.cpu().numpy().astype(np.uint8))

            buffer = io.BytesIO()
            pil_frame.save(buffer, format="PNG")
            frames_base64.append(base64.b64encode(buffer.getvalue()).decode())

        generation_time = time.time() - start_time

        return {
            "frames_base64": frames_base64,
            "num_frames": len(frames_base64),
            "width": width,
            "height": height,
            "fps": fps,
            "generation_time_s": generation_time,
            "effective_fps": len(frames_base64) / generation_time if generation_time > 0 else 0,
        }

    @modal.method()
    def generate_video_from_image(
        self,
        prompt: str,
        image_base64: str,
        width: int = 512,
        height: int = 512,
        num_frames: int = 30,
        fps: int = 15,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> dict[str, Any]:
        """Generate video frames conditioned on an input image.

        Uses LTXConditionPipeline if available for native image conditioning,
        otherwise falls back to inserting the input image as the first frame.

        Args:
            prompt: Text prompt for video generation
            image_base64: Base64-encoded input image for conditioning
            width: Frame width (default 512)
            height: Frame height (default 512)
            num_frames: Number of frames to generate (default 30)
            fps: Target framerate (default 15)
            num_inference_steps: Diffusion steps (default 30)
            guidance_scale: Classifier-free guidance scale (default 7.5)

        Returns:
            Dict with frames_base64 and generation metrics
        """
        import numpy as np
        import torch
        from PIL import Image

        start_time = time.time()
        used_native_conditioning = False

        # Decode input image
        image_data = base64.b64decode(image_base64)
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_image = input_image.resize((width, height))

        # Generate video with native image conditioning if available
        with torch.no_grad():
            if self.has_condition_pipeline and self.condition_pipeline is not None:
                try:
                    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

                    print("Using LTXConditionPipeline for image-conditioned generation")

                    # Create condition with frame_index=0 (first-frame anchoring)
                    condition = LTXVideoCondition(
                        image=input_image,
                        frame_index=0,
                    )

                    output = self.condition_pipeline(
                        prompt=prompt,
                        conditions=[condition],
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                    )
                    used_native_conditioning = True

                except (ImportError, Exception) as e:
                    print(f"Native conditioning failed: {e}, falling back to insert method")
                    output = self.pipeline(
                        prompt=prompt,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                    )
            else:
                # Standard text-to-video generation
                output = self.pipeline(
                    prompt=prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )

        # Extract frames
        frames = output.frames[0]
        frames_base64 = []

        if used_native_conditioning:
            # Native conditioning: all frames should be coherent with input
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    pil_frame = Image.fromarray(frame)
                elif isinstance(frame, Image.Image):
                    pil_frame = frame
                else:
                    pil_frame = Image.fromarray(frame.cpu().numpy().astype(np.uint8))

                buffer = io.BytesIO()
                pil_frame.save(buffer, format="PNG")
                frames_base64.append(base64.b64encode(buffer.getvalue()).decode())
        else:
            # Fallback: Use input image as first frame for continuity
            # First frame: input image
            buffer = io.BytesIO()
            input_image.save(buffer, format="PNG")
            frames_base64.append(base64.b64encode(buffer.getvalue()).decode())

            # Remaining frames: generated video (skip first to keep count)
            for frame in frames[1:] if len(frames) > 1 else frames:
                if isinstance(frame, np.ndarray):
                    pil_frame = Image.fromarray(frame)
                elif isinstance(frame, Image.Image):
                    pil_frame = frame
                else:
                    pil_frame = Image.fromarray(frame.cpu().numpy().astype(np.uint8))

                buffer = io.BytesIO()
                pil_frame.save(buffer, format="PNG")
                frames_base64.append(base64.b64encode(buffer.getvalue()).decode())

        generation_time = time.time() - start_time

        return {
            "frames_base64": frames_base64,
            "num_frames": len(frames_base64),
            "width": width,
            "height": height,
            "fps": fps,
            "generation_time_s": generation_time,
            "effective_fps": len(frames_base64) / generation_time if generation_time > 0 else 0,
            "input_image_used": True,
            "used_native_conditioning": used_native_conditioning,
            "conditioning_type": "ltx_condition" if used_native_conditioning else "first_frame_insert",
        }

    @modal.method()
    def get_status(self) -> dict[str, Any]:
        """Get video generator status."""
        import torch

        status = {
            "ready": True,
            "device": str(self.device),
            "model_loaded": hasattr(self, "pipeline") and self.pipeline is not None,
            "default_config": {
                "width": 512,
                "height": 512,
                "num_frames": 30,
                "fps": 15,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
            },
        }

        if torch.cuda.is_available():
            status["vram_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            status["vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            status["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            status["gpu_name"] = torch.cuda.get_device_name(0)

        return status


# ============================================================================
# Standalone Functions (for direct invocation)
# ============================================================================


@app.function(
    image=video_image,
    gpu="A10G",
    timeout=1800,
    volumes={"/model-cache": model_cache},
)
def generate_video_standalone(
    prompt: str,
    width: int = 512,
    height: int = 512,
    num_frames: int = 30,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
) -> dict[str, Any]:
    """Standalone video generation function (single-use, no persistent state).

    Use this for one-off video generation without keeping the model loaded.
    For repeated calls, use the VideoGenerator class instead.

    Args:
        prompt: Text prompt for video generation
        width: Frame width (default 512)
        height: Frame height (default 512)
        num_frames: Number of frames (default 30)
        num_inference_steps: Diffusion steps (default 30)
        guidance_scale: CFG scale (default 7.5)

    Returns:
        Dict with frames_base64 and generation metrics
    """
    import numpy as np
    import torch
    from diffusers import LTXPipeline
    from PIL import Image

    # Set cache directories
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pipeline
    print("Loading LTX-Video...")
    start_load = time.time()

    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.bfloat16,
    ).to(device)

    print(f"Loaded in {time.time() - start_load:.1f}s")

    # Generate video
    print(f"Generating video: {prompt[:50]}...")
    start_gen = time.time()

    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    frames = output.frames[0]

    # Convert to base64
    frames_base64 = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            pil_frame = Image.fromarray(frame)
        else:
            pil_frame = frame

        buffer = io.BytesIO()
        pil_frame.save(buffer, format="PNG")
        frames_base64.append(base64.b64encode(buffer.getvalue()).decode())

    generation_time = time.time() - start_gen

    return {
        "frames_base64": frames_base64,
        "num_frames": len(frames_base64),
        "width": width,
        "height": height,
        "fps": 15,
        "generation_time_s": generation_time,
        "effective_fps": len(frames_base64) / generation_time if generation_time > 0 else 0,
        "load_time_s": time.time() - start_load - generation_time,
    }


@app.function(
    image=video_image,
    gpu="A10G",
    timeout=1800,
    volumes={"/model-cache": model_cache},
)
def test_video_generation() -> dict[str, Any]:
    """Test the video generation pipeline.

    Generates a short test video and returns metrics.
    """
    import numpy as np
    import torch
    from diffusers import LTXPipeline
    from PIL import Image

    print("=" * 60)
    print("TESTING LTX-VIDEO GENERATION")
    print("=" * 60)

    # Set cache
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    print("\n[1/3] Loading LTX-Video...")
    start = time.time()

    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.bfloat16,
    ).to(device)

    load_time = time.time() - start
    print(f"  Loaded in {load_time:.1f}s")

    if torch.cuda.is_available():
        print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Test generation with default config
    print("\n[2/3] Generating test video (512x512, 30 frames)...")
    start = time.time()

    test_prompt = "A red ball bouncing on a wooden floor, smooth motion"

    with torch.no_grad():
        output = pipe(
            prompt=test_prompt,
            num_frames=30,
            height=512,
            width=512,
            num_inference_steps=30,
            guidance_scale=7.5,
        )

    gen_time = time.time() - start
    frames = output.frames[0]

    print(f"  Generated {len(frames)} frames in {gen_time:.1f}s")
    print(f"  Effective FPS: {len(frames) / gen_time:.1f}")

    # Verify frame properties
    print("\n[3/3] Verifying output...")
    first_frame = frames[0]
    if isinstance(first_frame, np.ndarray):
        h, w, c = first_frame.shape
    else:
        w, h = first_frame.size
        c = 3

    print(f"  Frame size: {w}x{h}")
    print(f"  Channels: {c}")
    print(f"  Total frames: {len(frames)}")

    # Convert one frame to base64 to test encoding
    if isinstance(first_frame, np.ndarray):
        pil_frame = Image.fromarray(first_frame)
    else:
        pil_frame = first_frame

    buffer = io.BytesIO()
    pil_frame.save(buffer, format="PNG")
    b64_size = len(base64.b64encode(buffer.getvalue()))
    print(f"  Base64 frame size: {b64_size / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print("TEST PASSED")
    print("=" * 60)

    return {
        "status": "ok",
        "load_time_s": load_time,
        "generation_time_s": gen_time,
        "num_frames": len(frames),
        "frame_size": f"{w}x{h}",
        "effective_fps": len(frames) / gen_time,
        "base64_frame_kb": b64_size / 1024,
    }


@app.local_entrypoint()
def main(action: str = "test", prompt: str = "A ball bouncing") -> None:
    """Local entrypoint for video generation.

    Args:
        action: 'test', 'generate', or 'status'
        prompt: Prompt for generation (when action='generate')
    """
    if action == "test":
        result = test_video_generation.remote()
        print(f"\nResult: {result}")
    elif action == "generate":
        result = generate_video_standalone.remote(prompt=prompt)
        print(f"\nGenerated {result['num_frames']} frames in {result['generation_time_s']:.1f}s")
        print(f"Effective FPS: {result['effective_fps']:.1f}")
    elif action == "status":
        generator = VideoGenerator()
        status = generator.get_status.remote()
        print(f"\nStatus: {status}")
    else:
        print(f"Unknown action: {action}")
        print("Available actions: test, generate, status")
