"""Inference service for the Foresight demo.

This module provides multiple inference backends:
- Mock: Placeholder responses without GPU (for development)
- Modal: Remote GPU inference via Modal (for production)
- Local: Direct model inference (requires GPU)

Usage:
    # Mock mode (default) - no GPU required
    FORESIGHT_INFERENCE_MODE=mock

    # Modal remote inference - uses deployed Modal app
    FORESIGHT_INFERENCE_MODE=modal

    # Local inference - requires local GPU
    FORESIGHT_INFERENCE_MODE=local

Environment variables:
    FORESIGHT_INFERENCE_MODE: mock | modal | local
    FORESIGHT_MODAL_APP_NAME: Name of deployed Modal app (default: foresight-inference)
"""

import asyncio
import base64
import io
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Any

from PIL import Image

from ..config import Settings, get_settings
from ..models.schemas import PredictionFrame, PredictionMetrics

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a prediction."""

    prediction_id: str
    video_url: str
    thumbnail_url: str
    frames: list[PredictionFrame]
    metrics: PredictionMetrics


@dataclass
class TextChunk:
    """A chunk of streamed text."""

    content: str
    done: bool


class InferenceService(ABC):
    """Abstract base class for inference services."""

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        image: Image.Image | None = None,
    ) -> AsyncGenerator[TextChunk, None]:
        """Generate text response with optional image context."""
        ...

    @abstractmethod
    async def generate_prediction(
        self,
        prompt: str,
        image: Image.Image | None = None,
        on_progress: asyncio.coroutines = None,
    ) -> PredictionResult:
        """Generate a video prediction."""
        ...

    @abstractmethod
    async def get_status(self) -> dict:
        """Get the current service status."""
        ...


class MockInferenceService(InferenceService):
    """Mock inference service for development and demo without GPU."""

    MOCK_RESPONSES = [
        "Based on my analysis of the image, I predict that {action}. "
        "The visual prediction shows this outcome playing out over the next few seconds.",
        "Looking at this scene, I can see that {action}. "
        "Watch the prediction panel to see how this unfolds visually.",
        "My prediction for this scenario: {action}. "
        "The generated video demonstrates the expected physical dynamics.",
        "Analyzing the visual context, I anticipate that {action}. "
        "The prediction visualization captures the key moments of this transition.",
    ]

    MOCK_ACTIONS = [
        "the object will fall and bounce slightly upon impact",
        "the liquid will flow downward and settle at the bottom",
        "there will be a gradual transition in lighting",
        "the motion will continue in the same direction with some deceleration",
        "the elements will interact and produce a visible change",
    ]

    def __init__(self, settings: Settings):
        self.settings = settings
        self._ready = True
        logger.info("MockInferenceService initialized")

    async def generate_text(
        self,
        prompt: str,
        image: Image.Image | None = None,
    ) -> AsyncGenerator[TextChunk, None]:
        """Generate mock text response with simulated streaming."""
        # Select a random response template and action
        template = random.choice(self.MOCK_RESPONSES)
        action = random.choice(self.MOCK_ACTIONS)
        full_response = template.format(action=action)

        # Simulate streaming by yielding words with small delays
        words = full_response.split()
        for i, word in enumerate(words):
            # Add space before word (except first)
            chunk = (" " if i > 0 else "") + word
            yield TextChunk(content=chunk, done=False)
            await asyncio.sleep(random.uniform(0.02, 0.08))

        yield TextChunk(content="", done=True)

    async def generate_prediction(
        self,
        prompt: str,
        image: Image.Image | None = None,
        on_progress=None,
    ) -> PredictionResult:
        """Generate a mock prediction with placeholder video/frames."""
        prediction_id = str(uuid.uuid4())
        start_time = time.time()

        num_frames = self.settings.num_frames
        fps = self.settings.fps

        # Simulate frame generation with progress updates
        frames: list[PredictionFrame] = []
        for i in range(num_frames):
            if on_progress:
                await on_progress(
                    progress=(i + 1) / num_frames * 100,
                    current_frame=i,
                    total_frames=num_frames,
                )

            # Create a placeholder frame
            frame = PredictionFrame(
                index=i,
                timestamp=i / fps,
                # Use a placeholder image URL (data URL with gray gradient)
                image_url=self._generate_placeholder_frame(i, num_frames),
            )
            frames.append(frame)

            # Simulate generation time
            await asyncio.sleep(random.uniform(0.02, 0.05))

        latency_ms = (time.time() - start_time) * 1000

        # Generate mock metrics
        metrics = PredictionMetrics(
            lpips=random.uniform(0.15, 0.35),
            confidence=random.uniform(0.7, 0.95),
            latency_ms=latency_ms,
            spatial_iou=random.uniform(0.5, 0.85) if random.random() > 0.3 else None,
        )

        return PredictionResult(
            prediction_id=prediction_id,
            # Use a placeholder video URL
            video_url=f"/api/static/placeholder_video_{prediction_id}.mp4",
            thumbnail_url=frames[0].image_url if frames else "",
            frames=frames,
            metrics=metrics,
        )

    def _generate_placeholder_frame(self, frame_index: int, total_frames: int) -> str:
        """Generate a placeholder frame as a data URL."""
        # Create a simple gradient image that changes over time
        width, height = 256, 144  # 16:9 aspect ratio, small for performance

        # Create a gradient that shifts based on frame index
        img = Image.new("RGB", (width, height))
        pixels = img.load()

        progress = frame_index / max(total_frames - 1, 1)
        base_hue = int(progress * 60)  # Shift from red to yellow

        for y in range(height):
            for x in range(width):
                # Create a diagonal gradient
                val = int((x / width + y / height) / 2 * 255)
                # Add frame-dependent color shift
                r = min(255, val + base_hue)
                g = min(255, val + int(progress * 30))
                b = max(0, val - base_hue)
                pixels[x, y] = (r, g, b)

        # Convert to base64 data URL
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"

    async def get_status(self) -> dict:
        """Get mock status."""
        return {
            "ready": self._ready,
            "model_loaded": True,
            "mock_mode": True,
            "modal_mode": False,
            "inference_mode": "mock",
            "vram_usage_gb": None,
            "vram_total_gb": None,
            "current_model": "Mock (no GPU)",
        }


class RealInferenceService(InferenceService):
    """Real inference service using actual models.

    Note: This is a placeholder for when real models are integrated.
    The actual implementation would load and use:
    - Qwen2.5-VL for vision-language understanding
    - DINOv2 for spatial features (P2 hybrid encoder)
    - LTX-Video for video generation
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._ready = False
        self._model = None
        logger.info("RealInferenceService initialized (models not loaded)")

    async def load_models(self) -> None:
        """Load the inference models."""
        logger.info("Loading models...")
        # TODO: Implement actual model loading
        # This would involve:
        # 1. Loading Qwen2.5-VL vision-language model
        # 2. Loading DINOv2 if hybrid encoder is enabled
        # 3. Loading LTX-Video decoder
        # 4. Setting up the conditioning adapter

        # For now, fall back to mock
        logger.warning("Real model loading not implemented, using mock behavior")
        self._ready = True

    async def generate_text(
        self,
        prompt: str,
        image: Image.Image | None = None,
    ) -> AsyncGenerator[TextChunk, None]:
        """Generate text using the VLM."""
        # TODO: Implement real VLM inference
        # For now, yield a placeholder
        yield TextChunk(
            content="Real inference not yet implemented. Please use mock mode.",
            done=False,
        )
        yield TextChunk(content="", done=True)

    async def generate_prediction(
        self,
        prompt: str,
        image: Image.Image | None = None,
        on_progress=None,
    ) -> PredictionResult:
        """Generate a real video prediction."""
        # TODO: Implement real prediction pipeline
        # This would involve:
        # 1. Encode image with VLM + optional DINOv2
        # 2. Generate query token predictions
        # 3. Fuse latents with conditioning adapter
        # 4. Decode to video with LTX-Video

        raise NotImplementedError("Real inference not yet implemented")

    async def get_status(self) -> dict:
        """Get real status including VRAM usage."""
        status: dict[str, Any] = {
            "ready": self._ready,
            "model_loaded": self._model is not None,
            "mock_mode": False,
            "modal_mode": False,
            "inference_mode": "local",
            "vram_usage_gb": None,
            "vram_total_gb": None,
            "current_model": self.settings.vlm_model if self._model else None,
        }

        # Try to get VRAM info if torch is available
        try:
            import torch

            if torch.cuda.is_available():
                status["vram_usage_gb"] = torch.cuda.memory_allocated() / 1e9
                status["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        except ImportError:
            pass

        return status


class ModalInferenceService(InferenceService):
    """Inference service using Modal remote GPU.

    This service proxies requests to a Modal inference endpoint running on GPU.
    All models (VLM, DINOv2, LTX-Video, Adapter) are loaded on the Modal container.

    Usage:
        Set FORESIGHT_INFERENCE_MODE=modal to enable this service.

    The Modal app must be deployed first:
        modal deploy infra/modal/inference_app.py
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._engine_cls: Any | None = None
        self._engine_instance: Any | None = None
        self._ready = False
        logger.info(f"ModalInferenceService initialized (app: {settings.modal_app_name})")

    def _get_engine_class(self) -> Any:
        """Get the Modal InferenceEngine class handle (lazy initialization)."""
        if self._engine_cls is None:
            try:
                import modal

                # Look up the deployed inference engine class
                # Uses the pattern: modal.Cls.from_name(app_name, class_name)
                self._engine_cls = modal.Cls.from_name(
                    self.settings.modal_app_name,
                    "InferenceEngine",
                )
                logger.info(f"Modal InferenceEngine class found: {self.settings.modal_app_name}")
            except ImportError:
                raise RuntimeError(
                    "Modal package not installed. Install with: pip install modal"
                )
            except Exception as e:
                logger.error(f"Failed to look up Modal inference engine: {e}")
                raise RuntimeError(
                    f"Modal inference engine not available. "
                    f"Deploy with: modal deploy infra/modal/inference_app.py\n"
                    f"Error: {e}"
                ) from e
        return self._engine_cls

    async def _get_engine(self) -> Any:
        """Get or create the Modal inference engine instance."""
        if self._engine_instance is None:
            engine_cls = self._get_engine_class()
            # Create an instance handle - this doesn't call the container yet
            self._engine_instance = engine_cls()
            self._ready = True
            logger.info("Modal inference engine instance created")
        return self._engine_instance

    async def generate_text(
        self,
        prompt: str,
        image: Image.Image | None = None,
    ) -> AsyncGenerator[TextChunk, None]:
        """Generate streaming text from the remote VLM."""
        engine = await self._get_engine()

        # Encode image if provided
        image_base64 = None
        if image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

        try:
            # Call Modal remote method (streaming)
            for chunk in engine.generate_text_stream.remote_gen(
                prompt=prompt,
                image_base64=image_base64,
                max_new_tokens=self.settings.max_new_tokens,
            ):
                yield TextChunk(
                    content=chunk.get("token", ""),
                    done=chunk.get("done", False),
                )

        except Exception as e:
            logger.error(f"Modal text generation error: {e}")
            yield TextChunk(content=f"Error: {e}", done=False)
            yield TextChunk(content="", done=True)

    async def generate_prediction(
        self,
        prompt: str,
        image: Image.Image | None = None,
        on_progress=None,
    ) -> PredictionResult:
        """Generate video prediction using Modal remote GPU."""
        engine = await self._get_engine()
        prediction_id = str(uuid.uuid4())
        start_time = time.time()

        # Encode image if provided
        image_base64 = None
        if image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

        try:
            # Call Modal video generation
            result = await asyncio.to_thread(
                engine.generate_video.remote,
                prompt=prompt,
                image_base64=image_base64,
                num_frames=self.settings.num_frames,
                height=self.settings.video_height,
                width=self.settings.video_width,
                num_inference_steps=self.settings.num_inference_steps,
                guidance_scale=self.settings.guidance_scale,
            )

            # Convert frames to PredictionFrame objects
            frames: list[PredictionFrame] = []
            frames_base64 = result.get("frames_base64", [])

            for i, frame_b64 in enumerate(frames_base64):
                if on_progress:
                    await on_progress(
                        progress=(i + 1) / len(frames_base64) * 100,
                        current_frame=i,
                        total_frames=len(frames_base64),
                    )

                frame = PredictionFrame(
                    index=i,
                    timestamp=i / self.settings.fps,
                    image_url=f"data:image/png;base64,{frame_b64}",
                )
                frames.append(frame)

            latency_ms = (time.time() - start_time) * 1000

            # Create metrics
            metrics = PredictionMetrics(
                lpips=0.20,  # Placeholder until we compute actual metrics
                confidence=0.85,
                latency_ms=latency_ms,
                spatial_iou=0.75 if self.settings.use_hybrid_encoder else None,
            )

            return PredictionResult(
                prediction_id=prediction_id,
                video_url=f"/api/static/video_{prediction_id}.mp4",
                thumbnail_url=frames[0].image_url if frames else "",
                frames=frames,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Modal video generation error: {e}")
            raise

    async def generate_concurrent(
        self,
        prompt: str,
        image: Image.Image | None = None,
        images: list[Image.Image] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Generate text and video concurrently from Modal.

        This method uses Modal's concurrent generation endpoint to stream
        both text tokens and video frames simultaneously.

        Args:
            image: Primary image for video generation
            images: Optional list of all images for multi-image VLM context
        """
        engine = await self._get_engine()

        # Encode primary image for video generation
        image_base64 = None
        if image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Encode all images for multi-image VLM context
        images_base64 = None
        if images and len(images) > 1:
            images_base64 = []
            for img in images:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                images_base64.append(base64.b64encode(buffer.getvalue()).decode())

        try:
            # Call Modal concurrent generation (streaming)
            for item in engine.generate_concurrent.remote_gen(
                prompt=prompt,
                image_base64=image_base64,
                images_base64=images_base64,
                max_new_tokens=self.settings.max_new_tokens,
                num_frames=self.settings.num_frames,
                height=self.settings.video_height,
                width=self.settings.video_width,
            ):
                yield item

        except Exception as e:
            logger.error(f"Modal concurrent generation error: {e}")
            yield {"type": "error", "error": str(e)}

    async def get_status(self) -> dict:
        """Get Modal inference status."""
        try:
            engine = await self._get_engine()
            # Call the remote status method
            status = await asyncio.to_thread(engine.get_status.remote)
            # Ensure our local status fields are set correctly
            status["mock_mode"] = False
            status["modal_mode"] = True
            status["inference_mode"] = "modal"
            return status
        except Exception as e:
            logger.warning(f"Failed to get Modal status: {e}")
            return {
                "ready": False,
                "model_loaded": False,
                "mock_mode": False,
                "modal_mode": True,
                "inference_mode": "modal",
                "error": str(e),
                "vram_usage_gb": None,
                "vram_total_gb": None,
                "current_model": None,
            }


# Global service instance
_inference_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    """Get or create the inference service singleton.

    Service selection based on FORESIGHT_INFERENCE_MODE:
    - 'mock' -> MockInferenceService (placeholder responses, no GPU)
    - 'modal' -> ModalInferenceService (remote GPU via Modal)
    - 'local' -> RealInferenceService (local GPU)
    """
    global _inference_service

    if _inference_service is None:
        settings = get_settings()

        if settings.inference_mode == "mock":
            logger.info("Using MockInferenceService (no GPU)")
            _inference_service = MockInferenceService(settings)
        elif settings.inference_mode == "modal":
            logger.info(f"Using ModalInferenceService (app: {settings.modal_app_name})")
            _inference_service = ModalInferenceService(settings)
        elif settings.inference_mode == "local":
            logger.info("Using RealInferenceService (local GPU)")
            _inference_service = RealInferenceService(settings)
        else:
            # Fallback to mock for unknown modes
            logger.warning(f"Unknown inference_mode '{settings.inference_mode}', using mock")
            _inference_service = MockInferenceService(settings)

    return _inference_service


def reset_inference_service() -> None:
    """Reset the inference service singleton.

    Useful for testing or when settings change.
    """
    global _inference_service
    _inference_service = None


async def decode_image(image_base64: str) -> Image.Image:
    """Decode a base64-encoded image."""
    image_data = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_data))
