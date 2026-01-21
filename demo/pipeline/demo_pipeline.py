"""
Demo Pipeline

Wraps the Foresight inference pipeline for demo-specific functionality,
including streaming responses and mock mode for development.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional

# Type hints for PIL - actual import happens at runtime
try:
    from PIL import Image
except ImportError:
    Image = Any


@dataclass
class PredictionResult:
    """Result from a prediction step."""

    text: str
    video_path: Optional[Path] = None
    frames: list = None
    confidence: Optional[float] = None
    lpips: Optional[float] = None
    latency: Optional[float] = None
    final_frame: Any = None  # PIL Image

    def __post_init__(self):
        if self.frames is None:
            self.frames = []


class DemoPipeline:
    """
    Main demo pipeline wrapping Foresight inference.

    This class handles:
    - Model loading and management
    - Streaming prediction generation
    - Caching for performance
    """

    def __init__(self, config: dict):
        """
        Initialize the demo pipeline.

        Args:
            config: Demo configuration dictionary
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.gen_config = config.get("generation", {})
        self.perf_config = config.get("performance", {})

        # Model references (lazy loaded)
        self._vlm = None
        self._dino = None
        self._adapter = None
        self._decoder = None

        # Caching
        self._encoding_cache = {}

    def _load_models(self):
        """Load all models. Called lazily on first prediction."""
        if self._vlm is not None:
            return  # Already loaded

        # TODO: Implement actual model loading
        # This will integrate with foresight_inference when available
        #
        # from foresight_models import (
        #     QwenVLEncoder,
        #     DINOv2Encoder,
        #     FusionAdapter,
        #     LTXVideoDecoder,
        # )
        #
        # self._vlm = QwenVLEncoder(self.model_config["vlm"])
        # if self.model_config.get("hybrid_encoder"):
        #     self._dino = DINOv2Encoder(self.model_config["dino_model"])
        # self._adapter = FusionAdapter(self.model_config.get("adapter_path"))
        # self._decoder = LTXVideoDecoder(self.model_config["video_decoder"])

        raise NotImplementedError(
            "Model loading not yet implemented. "
            "Use MockPipeline for development or implement foresight_inference integration."
        )

    def predict_streaming(
        self,
        message: str,
        image: Optional[Any] = None,
        state: Optional[dict] = None,
    ) -> Generator[dict, None, None]:
        """
        Generate a streaming prediction response.

        Yields intermediate results as the prediction progresses:
        1. Initial acknowledgment
        2. Streaming text from VLM
        3. Video prediction (when ready)
        4. Final metrics

        Args:
            message: User's text message
            image: Optional input image (PIL Image)
            state: Session state dictionary

        Yields:
            Dict with keys: text, video, confidence, lpips, latency, frames, final_frame
        """
        self._load_models()

        start_time = time.time()

        # Step 1: Initial response
        yield {
            "text": "Processing your request...",
            "video": None,
            "confidence": None,
            "lpips": None,
            "latency": None,
            "frames": [],
            "final_frame": None,
        }

        # Step 2: Encode input (with caching)
        # TODO: Implement actual encoding
        # vlm_latents = self._vlm.encode(image)
        # dino_latents = self._dino.encode(image) if self._dino else None

        # Step 3: Generate text response (streaming)
        # TODO: Implement actual text generation
        # for text_chunk in self._vlm.generate_text(message, image):
        #     yield {"text": text_chunk, ...}

        # Step 4: Generate video prediction
        # TODO: Implement actual video generation
        # query_latents = self._vlm.get_query_predictions()
        # fused = self._adapter.fuse(vlm_latents, dino_latents, query_latents)
        # video = self._decoder.generate(fused, self.gen_config["num_frames"])

        # Step 5: Final result with metrics
        latency = time.time() - start_time
        yield {
            "text": "Prediction complete.",
            "video": None,  # Path to generated video
            "confidence": 0.85,
            "lpips": 0.18,
            "latency": latency,
            "frames": [],
            "final_frame": None,
        }


class MockPipeline:
    """
    Mock pipeline for UI development without models.

    Simulates the prediction pipeline with configurable delays
    and placeholder content.
    """

    def __init__(self, config: dict):
        """
        Initialize mock pipeline.

        Args:
            config: Demo configuration dictionary
        """
        self.config = config
        self.dev_config = config.get("dev", {})
        self.mock_latency = self.dev_config.get("mock_latency_ms", 2000) / 1000.0
        self.sample_video = self.dev_config.get("sample_video_path")

    def predict_streaming(
        self,
        message: str,
        image: Optional[Any] = None,
        state: Optional[dict] = None,
    ) -> Generator[dict, None, None]:
        """
        Generate a mock streaming prediction.

        Simulates the prediction pipeline with realistic delays
        for UI development and testing.

        Args:
            message: User's text message
            image: Optional input image
            state: Session state dictionary

        Yields:
            Dict with simulated prediction results
        """
        start_time = time.time()

        # Simulate processing stages
        stages = [
            ("Analyzing your image...", 0.3),
            ("Understanding the scene...", 0.3),
            ("Predicting future states...", 0.4),
        ]

        accumulated_text = ""
        for text, duration_frac in stages:
            accumulated_text += text + " "
            time.sleep(self.mock_latency * duration_frac)
            yield {
                "text": accumulated_text,
                "video": None,
                "confidence": None,
                "lpips": None,
                "latency": f"{time.time() - start_time:.1f}s",
                "frames": [],
                "final_frame": None,
            }

        # Generate mock response
        mock_responses = {
            "pour": "Based on the visual analysis, the liquid will flow into the container, filling it gradually. The surface will show ripples initially, then settle.",
            "push": "The object will move in the direction of applied force. It may slide, roll, or tip depending on its shape and the surface friction.",
            "drop": "The object will fall due to gravity, accelerating at approximately 9.8 m/s^2 until it impacts the surface below.",
            "default": "I can see the current scene. Based on the visual information, I predict the following sequence of events will occur...",
        }

        # Select appropriate response
        message_lower = message.lower()
        response_text = mock_responses["default"]
        for key, text in mock_responses.items():
            if key in message_lower:
                response_text = text
                break

        latency = time.time() - start_time

        # Final result
        yield {
            "text": response_text,
            "video": self.sample_video if self.sample_video and Path(self.sample_video).exists() else None,
            "confidence": 0.87,
            "lpips": 0.162,
            "latency": f"{latency:.2f}s",
            "frames": [],  # Would contain frame thumbnails
            "final_frame": image,  # In mock mode, just return input as "prediction"
        }
