"""Pydantic schemas for API request/response models."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictionMetrics(BaseModel):
    """Metrics for a prediction."""

    lpips: float = Field(..., ge=0, le=1, description="LPIPS perceptual similarity score")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    latency_ms: float = Field(..., ge=0, description="Generation latency in milliseconds")
    spatial_iou: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Spatial IoU score (if available)",
    )


class PredictionFrame(BaseModel):
    """A single frame in a prediction timeline."""

    index: int = Field(..., ge=0, description="Frame index")
    timestamp: float = Field(..., ge=0, description="Timestamp in seconds")
    image_url: str = Field(..., description="URL to the frame image")


class PredictionResponse(BaseModel):
    """Response containing prediction results."""

    prediction_id: str = Field(..., description="Unique prediction ID")
    video_url: str = Field(..., description="URL to the generated video")
    thumbnail_url: str = Field(..., description="URL to the video thumbnail")
    frames: list[PredictionFrame] = Field(
        default_factory=list,
        description="Individual frames of the prediction",
    )
    metrics: PredictionMetrics = Field(..., description="Prediction quality metrics")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when prediction was created",
    )


class ChatMessage(BaseModel):
    """A single chat message."""

    id: str = Field(..., description="Unique message ID")
    role: Literal["user", "assistant"] = Field(..., description="Message sender role")
    content: str = Field(..., description="Message text content")
    image_url: str | None = Field(default=None, description="Optional attached image URL")
    prediction_id: str | None = Field(
        default=None,
        description="ID of associated prediction (for assistant messages)",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Message timestamp",
    )


class ChatRequest(BaseModel):
    """Request to send a chat message."""

    message: str = Field(..., min_length=1, description="User message text")
    message_id: str = Field(..., description="Client-generated message ID")
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded image data",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation continuity",
    )


class ChatResponse(BaseModel):
    """Response to a chat request (non-streaming)."""

    message_id: str = Field(..., description="Message ID")
    content: str = Field(..., description="Assistant response text")
    prediction_id: str | None = Field(
        default=None,
        description="ID of generated prediction",
    )


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    code: str | None = Field(default=None, description="Error code")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error details",
    )


class SystemStatus(BaseModel):
    """System status information."""

    ready: bool = Field(..., description="Whether the system is ready")
    model_loaded: bool = Field(..., description="Whether models are loaded")
    mock_mode: bool = Field(..., description="Whether mock mode is enabled")
    vram_usage_gb: float | None = Field(
        default=None,
        description="Current VRAM usage in GB",
    )
    vram_total_gb: float | None = Field(
        default=None,
        description="Total VRAM available in GB",
    )
    current_model: str | None = Field(
        default=None,
        description="Currently loaded model name",
    )


class WebSocketMessageType(str, Enum):
    """WebSocket message types."""

    TEXT_CHUNK = "text_chunk"
    PREDICTION_START = "prediction_start"
    PREDICTION_PROGRESS = "prediction_progress"
    PREDICTION_COMPLETE = "prediction_complete"
    ERROR = "error"


class TextChunkData(BaseModel):
    """Data for text chunk WebSocket messages."""

    message_id: str = Field(..., description="Message ID this chunk belongs to")
    chunk: str = Field(..., description="Text chunk content")
    done: bool = Field(default=False, description="Whether this is the final chunk")


class PredictionStartData(BaseModel):
    """Data for prediction start WebSocket messages."""

    prediction_id: str = Field(..., description="Prediction ID")


class PredictionProgressData(BaseModel):
    """Data for prediction progress WebSocket messages."""

    prediction_id: str = Field(..., description="Prediction ID")
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    current_frame: int | None = Field(default=None, description="Current frame being generated")
    total_frames: int | None = Field(default=None, description="Total frames to generate")


class PredictionCompleteData(BaseModel):
    """Data for prediction complete WebSocket messages."""

    prediction_id: str = Field(..., description="Prediction ID")
    video_url: str = Field(..., description="URL to the generated video")
    thumbnail_url: str = Field(..., description="URL to the thumbnail")
    frames: list[PredictionFrame] = Field(..., description="Frame data")
    metrics: PredictionMetrics = Field(..., description="Prediction metrics")


class ErrorData(BaseModel):
    """Data for error WebSocket messages."""

    message: str = Field(..., description="Error message")
    code: str | None = Field(default=None, description="Error code")


class WebSocketMessage(BaseModel):
    """WebSocket message envelope."""

    type: WebSocketMessageType = Field(..., description="Message type")
    data: (
        TextChunkData
        | PredictionStartData
        | PredictionProgressData
        | PredictionCompleteData
        | ErrorData
    ) = Field(..., description="Message payload")
