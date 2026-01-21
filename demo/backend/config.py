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

    # Model settings
    mock_mode: bool = Field(
        default=True,
        description="Use mock responses instead of real models (no GPU required)",
    )
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

    # Generation settings
    num_frames: int = Field(default=30, description="Number of frames to generate")
    fps: int = Field(default=15, description="Frames per second for video output")
    resolution: tuple[int, int] = Field(
        default=(512, 512),
        description="Video resolution (width, height)",
    )
    guidance_scale: float = Field(
        default=7.5,
        description="Classifier-free guidance scale",
    )

    # Performance settings
    use_fp16: bool = Field(default=True, description="Use FP16 for inference")
    cache_encodings: bool = Field(default=True, description="Cache image encodings")
    max_history: int = Field(
        default=10,
        description="Maximum number of predictions to keep in history",
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
