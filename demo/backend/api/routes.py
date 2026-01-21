"""REST API routes for the Foresight demo."""

import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from ..config import get_settings
from ..models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    SystemStatus,
)
from ..services import get_inference_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/status", response_model=SystemStatus)
async def get_status() -> SystemStatus:
    """Get the current system status."""
    service = get_inference_service()
    status = await service.get_status()
    return SystemStatus(**status)


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a chat message and get a response.

    This is a non-streaming endpoint for simpler clients.
    For streaming responses, use the WebSocket endpoint at /ws/chat.
    """
    service = get_inference_service()
    settings = get_settings()

    try:
        # Decode image if provided
        image = None
        if request.image_base64:
            from ..services.inference import decode_image
            image = await decode_image(request.image_base64)

        # Generate text response (collect all chunks)
        full_response = ""
        async for chunk in service.generate_text(request.message, image):
            full_response += chunk.content

        # Generate prediction
        prediction_id = None
        if image:
            try:
                result = await service.generate_prediction(request.message, image)
                prediction_id = result.prediction_id
            except Exception as e:
                logger.warning(f"Prediction generation failed: {e}")
                # Continue without prediction

        return ChatResponse(
            message_id=request.message_id,
            content=full_response,
            prediction_id=prediction_id,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "code": "CHAT_ERROR"},
        )


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
) -> dict:
    """
    Upload an image for later use in chat.

    Returns the URL where the image can be accessed.
    """
    settings = get_settings()

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail={"error": "File must be an image", "code": "INVALID_FILE_TYPE"},
        )

    # Read file content
    content = await file.read()

    # Validate file size (max 10MB)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail={"error": "File too large (max 10MB)", "code": "FILE_TOO_LARGE"},
        )

    # For now, just return a data URL
    # In production, you'd save to disk or cloud storage
    import base64
    data_url = f"data:{file.content_type};base64,{base64.b64encode(content).decode()}"

    return {"url": data_url, "filename": file.filename}


@router.get("/config")
async def get_config() -> dict:
    """Get the current configuration (non-sensitive values only)."""
    settings = get_settings()
    return {
        "mock_mode": settings.mock_mode,
        "num_frames": settings.num_frames,
        "fps": settings.fps,
        "resolution": settings.resolution,
        "use_hybrid_encoder": settings.use_hybrid_encoder,
    }
