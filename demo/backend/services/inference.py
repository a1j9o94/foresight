"""Inference service for the Foresight demo.

This module provides both mock and real inference capabilities.
In mock mode, it generates placeholder responses without requiring GPU.
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
from typing import AsyncGenerator

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
        status = {
            "ready": self._ready,
            "model_loaded": self._model is not None,
            "mock_mode": False,
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


# Global service instance
_inference_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    """Get or create the inference service singleton."""
    global _inference_service

    if _inference_service is None:
        settings = get_settings()

        if settings.mock_mode:
            _inference_service = MockInferenceService(settings)
        else:
            _inference_service = RealInferenceService(settings)

    return _inference_service


async def decode_image(image_base64: str) -> Image.Image:
    """Decode a base64-encoded image."""
    image_data = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_data))
