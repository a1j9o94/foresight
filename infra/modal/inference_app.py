"""Modal inference application for Foresight.

This module provides a persistent GPU inference endpoint with:
- Qwen2.5-VL-7B for vision-language understanding
- DINOv2-ViT-L for spatial feature extraction (hybrid encoder)
- LTX-Video for video generation
- QueryAdapter for bridging hybrid encoder to video decoder

Usage:
    # Deploy the inference endpoint
    modal deploy infra/modal/inference_app.py

    # Run a test generation
    modal run infra/modal/inference_app.py::test_inference

VRAM Budget (A100-80GB):
- Qwen2.5-VL-7B: ~14GB
- DINOv2-ViT-L: ~1.2GB
- LTX-Video: ~8GB
- Adapter: ~40MB
- Generation Buffer: ~20GB
- Streaming Buffer: ~10GB
- Total: ~53GB (within 60GB target)
"""

import asyncio
import base64
import io
import os
import time
from pathlib import Path
from typing import Any, Iterator

import modal

# Define the Modal app
app = modal.App("foresight-inference")

# GPU image with all dependencies
inference_image = (
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
        # VLM utils
        "qwen-vl-utils",
        "sentencepiece",
    )
)

# Volumes for model caching
model_cache = modal.Volume.from_name("foresight-model-cache", create_if_missing=True)

# Secrets for W&B (optional for logging)
try:
    wandb_secret = modal.Secret.from_name("wandb-api-key")
except Exception:
    wandb_secret = None


@app.cls(
    image=inference_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/model-cache": model_cache},
    secrets=[wandb_secret] if wandb_secret else [],
    container_idle_timeout=300,  # Keep warm for 5 minutes
    allow_concurrent_inputs=4,  # Allow concurrent requests
)
class InferenceEngine:
    """GPU inference engine with all models loaded."""

    # Model dimensions
    VLM_DIM = 3584  # Qwen2.5-VL hidden size
    DINO_DIM = 1024  # DINOv2-ViT-L
    LTX_DIM = 4096  # LTX-Video conditioning dim

    @modal.enter()
    def load_models(self) -> None:
        """Load all models on container startup."""
        import torch

        # Set cache directories
        os.environ["HF_HOME"] = "/model-cache"
        os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
        os.environ["TORCH_HOME"] = "/model-cache/torch"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Load all models
        self._load_vlm()
        self._load_dinov2()
        self._load_ltx_video()
        self._load_adapter()

        # Report VRAM usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"VRAM allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")

    def _load_vlm(self) -> None:
        """Load Qwen2.5-VL-7B for vision-language understanding."""
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print("Loading Qwen2.5-VL-7B...")
        start = time.time()

        self.vlm_processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
        )

        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.vlm_model.eval()

        print(f"VLM loaded in {time.time() - start:.1f}s")
        print(f"  Parameters: {self.vlm_model.num_parameters() / 1e9:.1f}B")

    def _load_dinov2(self) -> None:
        """Load DINOv2-ViT-L for spatial feature extraction."""
        import torch

        print("Loading DINOv2-ViT-L...")
        start = time.time()

        self.dinov2_model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitl14",
        )
        self.dinov2_model = self.dinov2_model.to(self.device)
        self.dinov2_model.eval()

        print(f"DINOv2 loaded in {time.time() - start:.1f}s")
        print(f"  Parameters: {sum(p.numel() for p in self.dinov2_model.parameters()) / 1e6:.1f}M")

        # Store transform for later
        from torchvision import transforms
        self.dinov2_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_ltx_video(self) -> None:
        """Load LTX-Video for video generation."""
        import torch
        from diffusers import LTXPipeline

        print("Loading LTX-Video...")
        start = time.time()

        self.ltx_pipeline = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
        )
        self.ltx_pipeline = self.ltx_pipeline.to(self.device)

        print(f"LTX-Video loaded in {time.time() - start:.1f}s")

    def _load_adapter(self) -> None:
        """Load or initialize the QueryAdapter for bridging encoder to decoder."""
        import torch
        import torch.nn as nn

        print("Initializing QueryAdapter...")

        # QueryAdapter from C2 experiments
        class QueryAdapter(nn.Module):
            """Query-based adapter using learned queries to attend to hybrid features."""

            def __init__(
                self,
                vlm_dim: int = 3584,
                dino_dim: int = 1024,
                ltx_dim: int = 4096,
                hidden_dim: int = 384,
                n_layers: int = 2,
                n_queries: int = 77,
            ):
                super().__init__()
                self.name = "query_adapter"

                # Input projections
                self.vlm_proj = nn.Linear(vlm_dim, hidden_dim)
                self.dino_proj = nn.Linear(dino_dim, hidden_dim)

                # Learnable queries
                self.queries = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)

                # Cross-attention layers
                self.layers = nn.ModuleList([
                    nn.TransformerDecoderLayer(
                        d_model=hidden_dim,
                        nhead=max(1, hidden_dim // 64),
                        dim_feedforward=hidden_dim * 4,
                        batch_first=True,
                        dropout=0.1,
                    )
                    for _ in range(n_layers)
                ])

                # Output projection
                self.out_proj = nn.Linear(hidden_dim, ltx_dim)

            def forward(
                self, vlm_features: torch.Tensor, dino_features: torch.Tensor
            ) -> torch.Tensor:
                B = vlm_features.size(0)

                vlm = self.vlm_proj(vlm_features)
                dino = self.dino_proj(dino_features)
                context = torch.cat([vlm, dino], dim=1)

                queries = self.queries.unsqueeze(0).expand(B, -1, -1)
                for layer in self.layers:
                    queries = layer(queries, context)

                return self.out_proj(queries)

        self.adapter = QueryAdapter(
            vlm_dim=self.VLM_DIM,
            dino_dim=self.DINO_DIM,
            ltx_dim=self.LTX_DIM,
        )
        self.adapter = self.adapter.to(self.device).to(torch.bfloat16)
        self.adapter.eval()

        n_params = sum(p.numel() for p in self.adapter.parameters())
        print(f"  Adapter parameters: {n_params / 1e6:.2f}M")

    @modal.method()
    def generate_text(
        self,
        image_b64: str,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate text response from VLM given an image and prompt.

        Args:
            image_b64: Base64-encoded image
            prompt: Text prompt for the VLM
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text response as string
        """
        import torch
        from PIL import Image

        # Decode image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process with VLM
        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.vlm_processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.vlm_model.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode only the newly generated tokens
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        response = self.vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response

    @modal.method()
    def generate_text_stream(
        self,
        prompt: str,
        image_base64: str | None = None,
        max_new_tokens: int = 512,
    ) -> Iterator[dict[str, Any]]:
        """Generate streaming text response from VLM.

        Args:
            prompt: Text prompt for the VLM
            image_base64: Optional base64-encoded image
            max_new_tokens: Maximum tokens to generate

        Yields:
            Dict with 'token' (str) and 'done' (bool)
        """
        import torch
        from PIL import Image

        # Decode image if provided
        image = None
        if image_base64:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Prepare messages
        if image:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]

        # Process with VLM
        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if image:
            inputs = self.vlm_processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.vlm_model.device)
        else:
            inputs = self.vlm_processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            ).to(self.vlm_model.device)

        # Stream generation
        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(
            self.vlm_processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "streamer": streamer,
        }

        thread = Thread(target=self.vlm_model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            yield {"token": token, "done": False}

        thread.join()
        yield {"token": "", "done": True}

    @modal.method()
    def extract_features(
        self,
        image_base64: str,
        use_hybrid: bool = True,
    ) -> dict[str, Any]:
        """Extract features from image using hybrid encoder.

        Args:
            image_base64: Base64-encoded image
            use_hybrid: Whether to use both VLM and DINOv2 (vs VLM only)

        Returns:
            Dict with 'vlm_features', 'dino_features' (if hybrid), and 'adapter_output'
        """
        import torch
        from PIL import Image
        import numpy as np

        # Decode image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        result = {}

        # Extract VLM features
        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]

            text = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.vlm_processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.vlm_model.device)

            outputs = self.vlm_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            vlm_features = outputs.hidden_states[-1].float()
            result["vlm_shape"] = list(vlm_features.shape)

        # Extract DINOv2 features if hybrid
        if use_hybrid:
            with torch.no_grad():
                img_tensor = self.dinov2_transform(image).unsqueeze(0).to(self.device)
                dino_out = self.dinov2_model.forward_features(img_tensor)

                if isinstance(dino_out, dict):
                    dino_features = dino_out.get(
                        "x_norm_patchtokens",
                        dino_out.get("x_prenorm", dino_out["x_norm"][:, 1:, :])
                    )
                else:
                    dino_features = dino_out[:, 1:, :]  # Exclude CLS token

                result["dino_shape"] = list(dino_features.shape)

                # Run through adapter
                adapter_input_vlm = vlm_features.to(torch.bfloat16)
                adapter_input_dino = dino_features.to(torch.bfloat16)

                # Pad/truncate VLM features to match expected sequence length
                if adapter_input_vlm.shape[1] > 256:
                    adapter_input_vlm = adapter_input_vlm[:, :256, :]
                elif adapter_input_vlm.shape[1] < 256:
                    pad = torch.zeros(
                        1, 256 - adapter_input_vlm.shape[1], adapter_input_vlm.shape[2],
                        device=adapter_input_vlm.device,
                        dtype=adapter_input_vlm.dtype,
                    )
                    adapter_input_vlm = torch.cat([adapter_input_vlm, pad], dim=1)

                adapter_out = self.adapter(adapter_input_vlm, adapter_input_dino)
                result["adapter_shape"] = list(adapter_out.shape)

        return result

    @modal.method()
    def generate_video(
        self,
        prompt: str,
        image_base64: str | None = None,
        num_frames: int = 30,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> dict[str, Any]:
        """Generate video frames from prompt and optional image.

        Args:
            prompt: Text prompt for video generation
            image_base64: Optional base64-encoded conditioning image
            num_frames: Number of frames to generate
            height: Frame height
            width: Frame width
            num_inference_steps: Diffusion steps
            guidance_scale: Classifier-free guidance scale

        Returns:
            Dict with 'frames_base64' (list of base64 images) and 'metrics'
        """
        import torch
        from PIL import Image
        import numpy as np

        start_time = time.time()

        # Decode conditioning image if provided
        conditioning_image = None
        if image_base64:
            image_data = base64.b64decode(image_base64)
            conditioning_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Generate video
        with torch.no_grad():
            # LTX-Video text-to-video generation
            output = self.ltx_pipeline(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

        # Extract frames
        frames = output.frames[0]  # [num_frames, height, width, 3]

        # Convert to base64
        frames_base64 = []
        for i, frame in enumerate(frames):
            if isinstance(frame, np.ndarray):
                pil_frame = Image.fromarray(frame)
            else:
                pil_frame = frame

            buffer = io.BytesIO()
            pil_frame.save(buffer, format="PNG")
            frames_base64.append(base64.b64encode(buffer.getvalue()).decode())

        generation_time = time.time() - start_time

        return {
            "frames_base64": frames_base64,
            "num_frames": len(frames_base64),
            "height": height,
            "width": width,
            "generation_time_s": generation_time,
            "fps": len(frames_base64) / generation_time if generation_time > 0 else 0,
        }

    @modal.method()
    def generate_concurrent(
        self,
        prompt: str,
        image_base64: str | None = None,
        images_base64: list[str] | None = None,
        max_new_tokens: int = 256,
        num_frames: int = 30,
        height: int = 512,
        width: int = 512,
    ) -> Iterator[dict[str, Any]]:
        """Generate text and video concurrently, streaming results.

        This method interleaves text tokens with video generation progress,
        enabling the frontend to display both simultaneously.

        Args:
            prompt: Text prompt
            image_base64: Primary image for video generation
            images_base64: Optional list of all images for multi-image VLM context
            max_new_tokens: Max tokens for text
            num_frames: Video frames
            height: Video height
            width: Video width

        Yields:
            Dict with type ('text_chunk', 'video_progress', 'video_frame', 'complete')
        """
        import torch
        from PIL import Image
        from threading import Thread
        import queue

        # Decode all images for multi-image VLM context
        all_images = []
        if images_base64:
            for img_b64 in images_base64:
                img_data = base64.b64decode(img_b64)
                all_images.append(Image.open(io.BytesIO(img_data)).convert("RGB"))
        elif image_base64:
            image_data = base64.b64decode(image_base64)
            all_images.append(Image.open(io.BytesIO(image_data)).convert("RGB"))

        # Primary image for video generation (most recent)
        primary_image = all_images[-1] if all_images else None

        # Queues for async communication
        text_queue: queue.Queue = queue.Queue()
        video_queue: queue.Queue = queue.Queue()

        # Text generation thread
        def generate_text():
            try:
                # Build message content with all images
                if all_images:
                    content = []
                    for i, img in enumerate(all_images):
                        content.append({"type": "image", "image": img})
                    content.append({"type": "text", "text": prompt})
                    messages = [{"role": "user", "content": content}]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ]

                text = self.vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                if all_images:
                    inputs = self.vlm_processor(
                        text=[text],
                        images=all_images,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.vlm_model.device)
                else:
                    inputs = self.vlm_processor(
                        text=[text],
                        padding=True,
                        return_tensors="pt",
                    ).to(self.vlm_model.device)

                from transformers import TextIteratorStreamer

                streamer = TextIteratorStreamer(
                    self.vlm_processor.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )

                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "streamer": streamer,
                }

                gen_thread = Thread(target=self.vlm_model.generate, kwargs=gen_kwargs)
                gen_thread.start()

                for token in streamer:
                    text_queue.put({"type": "text_chunk", "token": token})

                gen_thread.join()
                text_queue.put({"type": "text_done"})

            except Exception as e:
                text_queue.put({"type": "text_error", "error": str(e)})

        # Video generation thread
        def generate_video():
            try:
                video_queue.put({"type": "video_start"})

                with torch.no_grad():
                    output = self.ltx_pipeline(
                        prompt=prompt,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                    )

                frames = output.frames[0]
                video_queue.put({"type": "video_progress", "progress": 100})

                # Send frames
                frames_base64 = []
                for i, frame in enumerate(frames):
                    import numpy as np
                    if isinstance(frame, np.ndarray):
                        pil_frame = Image.fromarray(frame)
                    else:
                        pil_frame = frame

                    buffer = io.BytesIO()
                    pil_frame.save(buffer, format="PNG")
                    frame_b64 = base64.b64encode(buffer.getvalue()).decode()
                    frames_base64.append(frame_b64)

                    video_queue.put({
                        "type": "video_frame",
                        "index": i,
                        "total": len(frames),
                        "frame_base64": frame_b64,
                    })

                video_queue.put({"type": "video_done", "num_frames": len(frames_base64)})

            except Exception as e:
                video_queue.put({"type": "video_error", "error": str(e)})

        # Start both threads
        text_thread = Thread(target=generate_text)
        video_thread = Thread(target=generate_video)

        text_thread.start()
        video_thread.start()

        # Interleave outputs
        text_done = False
        video_done = False

        while not (text_done and video_done):
            # Check text queue
            try:
                item = text_queue.get_nowait()
                if item["type"] == "text_done":
                    text_done = True
                elif item["type"] == "text_error":
                    yield {"type": "error", "source": "text", "error": item["error"]}
                    text_done = True
                else:
                    yield item
            except queue.Empty:
                pass

            # Check video queue
            try:
                item = video_queue.get_nowait()
                if item["type"] == "video_done":
                    video_done = True
                    yield item
                elif item["type"] == "video_error":
                    yield {"type": "error", "source": "video", "error": item["error"]}
                    video_done = True
                else:
                    yield item
            except queue.Empty:
                pass

            # Small sleep to avoid busy waiting
            time.sleep(0.01)

        text_thread.join()
        video_thread.join()

        yield {"type": "complete"}

    @modal.method()
    def get_status(self) -> dict[str, Any]:
        """Get inference engine status."""
        import torch

        status = {
            "ready": True,
            "device": str(self.device),
            "models_loaded": {
                "vlm": hasattr(self, "vlm_model") and self.vlm_model is not None,
                "dinov2": hasattr(self, "dinov2_model") and self.dinov2_model is not None,
                "ltx_video": hasattr(self, "ltx_pipeline") and self.ltx_pipeline is not None,
                "adapter": hasattr(self, "adapter") and self.adapter is not None,
            },
        }

        if torch.cuda.is_available():
            status["vram_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            status["vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            status["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            status["gpu_name"] = torch.cuda.get_device_name(0)

        return status


# ============================================================================
# Test Functions
# ============================================================================


@app.function(
    image=inference_image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/model-cache": model_cache},
)
def test_inference():
    """Test the inference engine with a simple example."""
    import base64
    import io
    from PIL import Image
    import numpy as np

    print("=" * 60)
    print("TESTING FORESIGHT INFERENCE ENGINE")
    print("=" * 60)

    # Create a test image
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 80, 180, 180], fill=(255, 0, 0))  # Red circle

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    # Create engine instance
    engine = InferenceEngine()

    # Test status
    print("\n[Test 1] Engine Status")
    status = engine.get_status.local()
    print(f"  Status: {status}")

    # Test text generation
    print("\n[Test 2] Text Generation (streaming)")
    prompt = "What do you see in this image? Describe it briefly."
    text_output = ""
    for chunk in engine.generate_text_stream.local(prompt, image_base64, max_new_tokens=100):
        text_output += chunk["token"]
        if chunk["done"]:
            break
    print(f"  Response: {text_output[:200]}...")

    # Test feature extraction
    print("\n[Test 3] Feature Extraction (hybrid encoder)")
    features = engine.extract_features.local(image_base64, use_hybrid=True)
    print(f"  VLM features shape: {features['vlm_shape']}")
    print(f"  DINOv2 features shape: {features['dino_shape']}")
    print(f"  Adapter output shape: {features['adapter_shape']}")

    # Test video generation
    print("\n[Test 4] Video Generation")
    video_result = engine.generate_video.local(
        prompt="A red ball bouncing",
        num_frames=9,  # Short video for testing
        height=256,
        width=256,
        num_inference_steps=10,  # Fewer steps for testing
    )
    print(f"  Generated {video_result['num_frames']} frames")
    print(f"  Generation time: {video_result['generation_time_s']:.2f}s")
    print(f"  FPS: {video_result['fps']:.1f}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    return {"status": "ok", "tests_passed": 4}


@app.local_entrypoint()
def main(action: str = "status"):
    """Local entrypoint for the inference app.

    Args:
        action: 'status', 'test', or 'deploy'
    """
    if action == "status":
        engine = InferenceEngine()
        status = engine.get_status.remote()
        print(f"Inference engine status: {status}")
    elif action == "test":
        result = test_inference.remote()
        print(f"Test result: {result}")
    else:
        print(f"Unknown action: {action}")
        print("Available actions: status, test")
