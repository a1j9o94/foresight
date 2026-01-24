"""Configuration for the Foresight demo backend."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")

    # CORS settings
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins",
    )

    # Inference mode settings
    inference_mode: Literal["mock", "local", "modal"] = Field(
        default="mock",
        description=(
            "Inference mode: "
            "'mock' = placeholder responses (no GPU), "
            "'local' = local GPU inference, "
            "'modal' = remote GPU inference via Modal"
        ),
    )
    modal_app_name: str = Field(
        default="foresight-inference",
        description="Modal app name for inference endpoint",
    )
    modal_gpu: str = Field(
        default="A100-80GB",
        description="GPU type for Modal inference",
    )

    # Legacy mock_mode for backwards compatibility
    @property
    def mock_mode(self) -> bool:
        """Backwards compatibility: mock_mode is True when inference_mode is 'mock'."""
        return self.inference_mode == "mock"

    @property
    def modal_mode(self) -> bool:
        """True when using Modal remote inference."""
        return self.inference_mode == "modal"

    # Model settings
    vlm_model: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="Vision-language model to use",
    )
    video_decoder: str = Field(
        default="Lightricks/LTX-Video",
        description="Video decoder model to use",
    )
    dino_model: str = Field(
        default="facebook/dinov2-vitl14",
        description="DINOv2 model for hybrid encoder (P2)",
    )
    use_hybrid_encoder: bool = Field(
        default=True,
        description="Enable DINOv2 hybrid encoder (P2 experiment)",
    )
    use_adapter: bool = Field(
        default=True,
        description="Enable QueryAdapter for conditioning video generation",
    )

    # Generation settings
    num_frames: int = Field(default=30, description="Number of frames to generate")
    fps: int = Field(default=15, description="Frames per second for video output")
    video_width: int = Field(default=512, description="Video frame width")
    video_height: int = Field(default=512, description="Video frame height")

    @property
    def resolution(self) -> tuple[int, int]:
        """Video resolution as (width, height) tuple."""
        return (self.video_width, self.video_height)

    guidance_scale: float = Field(
        default=7.5,
        description="Classifier-free guidance scale",
    )
    num_inference_steps: int = Field(
        default=30,
        description="Number of diffusion inference steps",
    )

    # Performance settings
    use_fp16: bool = Field(default=True, description="Use FP16 for inference")
    cache_encodings: bool = Field(default=True, description="Cache image encodings")
    max_history: int = Field(
        default=10,
        description="Maximum number of predictions to keep in history",
    )
    max_new_tokens: int = Field(
        default=512,
        description="Maximum tokens for VLM text generation",
    )

    # Paths
    static_dir: Path = Field(
        default=Path(__file__).parent / "static",
        description="Directory for static files",
    )
    cache_dir: Path = Field(
        default=Path(__file__).parent / ".cache",
        description="Directory for cached files",
    )

    model_config = {
        "env_prefix": "FORESIGHT_",
        "env_file": ".env",
        "extra": "ignore",
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings."""
    return settings
