"""Pydantic schemas for API request/response models."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def to_camel(string: str) -> str:
    """Convert snake_case to camelCase."""
    components = string.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


class CamelCaseModel(BaseModel):
    """Base model that outputs camelCase JSON."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


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
    model_loaded: bool = Field(default=False, description="Whether models are loaded")
    mock_mode: bool = Field(default=False, description="Whether mock mode is enabled")
    modal_mode: bool = Field(default=False, description="Whether Modal remote inference is enabled")
    inference_mode: str | None = Field(
        default=None,
        description="Current inference mode: mock, modal, or local",
    )
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
    error: str | None = Field(
        default=None,
        description="Error message if service is not ready",
    )


class WebSocketMessageType(str, Enum):
    """WebSocket message types."""

    TEXT_CHUNK = "text_chunk"
    PREDICTION_START = "prediction_start"
    PREDICTION_PROGRESS = "prediction_progress"
    PREDICTION_COMPLETE = "prediction_complete"
    VIDEO_PROMPT = "video_prompt"
    ERROR = "error"


class TextChunkData(CamelCaseModel):
    """Data for text chunk WebSocket messages."""

    message_id: str = Field(..., description="Message ID this chunk belongs to")
    chunk: str = Field(..., description="Text chunk content")
    done: bool = Field(default=False, description="Whether this is the final chunk")


class PredictionStartData(CamelCaseModel):
    """Data for prediction start WebSocket messages."""

    prediction_id: str = Field(..., description="Prediction ID")


class VideoPromptData(CamelCaseModel):
    """Data for video prompt WebSocket messages.

    Sent before video generation to show the user what conditioning
    information is being used.
    """

    prediction_id: str = Field(..., description="Prediction ID")
    text_prompt: str = Field(..., description="Text prompt used for generation")
    conditioning_type: str = Field(
        ...,
        description="Type of conditioning: 'text_only', 'first_frame_insert', 'ltx_condition', or 'hybrid_encoder'"
    )
    image_used: bool = Field(default=False, description="Whether an image was used for conditioning")
    hybrid_encoder_used: bool = Field(
        default=False,
        description="Whether the hybrid encoder (DINOv2 + VLM) was used"
    )
    image_dimensions: tuple[int, int] | None = Field(
        default=None,
        description="Dimensions of conditioning image (width, height)"
    )


class PredictionProgressData(CamelCaseModel):
    """Data for prediction progress WebSocket messages."""

    prediction_id: str = Field(..., description="Prediction ID")
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    current_frame: int | None = Field(default=None, description="Current frame being generated")
    total_frames: int | None = Field(default=None, description="Total frames to generate")


class PredictionCompleteData(CamelCaseModel):
    """Data for prediction complete WebSocket messages."""

    prediction_id: str = Field(..., description="Prediction ID")
    video_url: str = Field(..., description="URL to the generated video")
    thumbnail_url: str = Field(..., description="URL to the thumbnail")
    frames: list["PredictionFrameData"] = Field(..., description="Frame data")
    metrics: "PredictionMetricsData" = Field(..., description="Prediction metrics")


class PredictionFrameData(CamelCaseModel):
    """Frame data for WebSocket messages (camelCase)."""

    index: int = Field(..., ge=0, description="Frame index")
    timestamp: float = Field(..., ge=0, description="Timestamp in seconds")
    image_url: str = Field(..., description="URL to the frame image")


class PredictionMetricsData(CamelCaseModel):
    """Metrics data for WebSocket messages (camelCase)."""

    lpips: float = Field(..., ge=0, le=1, description="LPIPS perceptual similarity score")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    latency_ms: float = Field(..., ge=0, description="Generation latency in milliseconds")
    spatial_iou: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Spatial IoU score (if available)",
    )


class ErrorData(CamelCaseModel):
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
