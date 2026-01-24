"""WebSocket endpoint for streaming chat and predictions.

This module handles real-time communication with the frontend for:
- Streaming text responses from the VLM
- Streaming video generation progress and frames
- Concurrent text + video generation (Modal mode)

Protocol:
- Client sends: {"message": str, "messageId": str, "imageBase64"?: str}
- Server sends:
  - {"type": "text_chunk", "data": {"messageId": str, "chunk": str, "done": bool}}
  - {"type": "prediction_start", "data": {"predictionId": str}}
  - {"type": "prediction_progress", "data": {"predictionId": str, "progress": float, ...}}
  - {"type": "prediction_frame", "data": {"predictionId": str, "index": int, "frameBase64": str}}
  - {"type": "prediction_complete", "data": {"predictionId": str, "videoUrl": str, ...}}
  - {"type": "error", "data": {"message": str, "code"?: str}}
"""

import asyncio
import logging
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from ..config import get_settings
from ..models.schemas import (
    WebSocketMessage,
    WebSocketMessageType,
    TextChunkData,
    PredictionStartData,
    PredictionProgressData,
    PredictionCompleteData,
    ErrorData,
    PredictionFrameData,
    PredictionMetricsData,
)
from ..services import get_inference_service
from ..services.inference import decode_image, ModalInferenceService

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connected: {connection_id}")
        return connection_id

    def disconnect(self, connection_id: str) -> None:
        """Remove a WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_message(
        self,
        connection_id: str,
        message_type: WebSocketMessageType,
        data: dict[str, Any],
    ) -> None:
        """Send a message to a specific connection."""
        if connection_id not in self.active_connections:
            return

        websocket = self.active_connections[connection_id]
        # Send with camelCase field names for frontend compatibility
        message = {"type": message_type.value, "data": data}
        await websocket.send_json(message)


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time chat with streaming responses.

    Protocol:
    - Client sends: {"message": str, "messageId": str, "imageBase64"?: str}
    - Server sends:
      - {"type": "text_chunk", "data": {"messageId": str, "chunk": str, "done": bool}}
      - {"type": "prediction_start", "data": {"predictionId": str}}
      - {"type": "prediction_progress", "data": {"predictionId": str, "progress": float, ...}}
      - {"type": "prediction_complete", "data": {"predictionId": str, "videoUrl": str, ...}}
      - {"type": "error", "data": {"message": str, "code"?: str}}
    """
    connection_id = await manager.connect(websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            # Validate required fields
            message = data.get("message", "")
            message_id = data.get("messageId")
            image_base64 = data.get("imageBase64")
            images_base64 = data.get("imagesBase64")  # Multiple images for context

            if not message_id:
                await manager.send_message(
                    connection_id,
                    WebSocketMessageType.ERROR,
                    ErrorData(message="Missing messageId", code="MISSING_MESSAGE_ID").model_dump(by_alias=True),
                )
                continue

            # Process the message
            await process_chat_message(
                connection_id=connection_id,
                message=message,
                message_id=message_id,
                image_base64=image_base64,
                images_base64=images_base64,
            )

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await manager.send_message(
                connection_id,
                WebSocketMessageType.ERROR,
                ErrorData(message=str(e), code="INTERNAL_ERROR").model_dump(by_alias=True),
            )
        except Exception:
            pass
    finally:
        manager.disconnect(connection_id)


async def process_chat_message(
    connection_id: str,
    message: str,
    message_id: str,
    image_base64: str | None = None,
    images_base64: list[str] | None = None,
) -> None:
    """Process a chat message and send streaming responses.

    For Modal inference, uses concurrent generation for optimal performance.
    For other modes, runs text and prediction generation in parallel tasks.

    Args:
        images_base64: Optional list of all images in conversation for multi-image context
    """
    service = get_inference_service()
    settings = get_settings()

    try:
        # Decode images - support multi-image context
        images = []
        if images_base64:
            # Multiple images provided for context
            for img_b64 in images_base64:
                images.append(await decode_image(img_b64))
        elif image_base64:
            # Single image (backwards compatibility)
            images.append(await decode_image(image_base64))

        # Use most recent image for video generation, all images for VLM context
        primary_image = images[-1] if images else None

        # Use concurrent generation for Modal mode with images
        if isinstance(service, ModalInferenceService) and primary_image:
            await stream_concurrent_generation(
                connection_id=connection_id,
                message_id=message_id,
                service=service,
                message=message,
                image=primary_image,
                images=images if len(images) > 1 else None,
            )
        else:
            # Standard parallel execution for other modes
            text_task = asyncio.create_task(
                stream_text_response(connection_id, message_id, service, message, primary_image)
            )

            prediction_task = None
            if primary_image:
                prediction_task = asyncio.create_task(
                    stream_prediction(connection_id, service, message, primary_image)
                )

            # Wait for both tasks
            await text_task
            if prediction_task:
                await prediction_task

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await manager.send_message(
            connection_id,
            WebSocketMessageType.ERROR,
            ErrorData(message=str(e), code="PROCESSING_ERROR").model_dump(by_alias=True),
        )


async def stream_text_response(
    connection_id: str,
    message_id: str,
    service,
    message: str,
    image,
) -> None:
    """Stream text response chunks to the client."""
    try:
        async for chunk in service.generate_text(message, image):
            await manager.send_message(
                connection_id,
                WebSocketMessageType.TEXT_CHUNK,
                TextChunkData(
                    message_id=message_id,
                    chunk=chunk.content,
                    done=chunk.done,
                ).model_dump(by_alias=True),
            )
    except Exception as e:
        logger.error(f"Error streaming text: {e}")
        await manager.send_message(
            connection_id,
            WebSocketMessageType.ERROR,
            ErrorData(message=f"Text generation error: {e}", code="TEXT_ERROR").model_dump(by_alias=True),
        )


async def stream_prediction(
    connection_id: str,
    service,
    message: str,
    image,
) -> None:
    """Generate and stream prediction progress to the client."""
    prediction_id = str(uuid.uuid4())

    try:
        # Send prediction start
        await manager.send_message(
            connection_id,
            WebSocketMessageType.PREDICTION_START,
            PredictionStartData(prediction_id=prediction_id).model_dump(by_alias=True),
        )

        # Progress callback
        async def on_progress(progress: float, current_frame: int, total_frames: int):
            await manager.send_message(
                connection_id,
                WebSocketMessageType.PREDICTION_PROGRESS,
                PredictionProgressData(
                    prediction_id=prediction_id,
                    progress=progress,
                    current_frame=current_frame,
                    total_frames=total_frames,
                ).model_dump(by_alias=True),
            )

        # Generate prediction
        result = await service.generate_prediction(
            message,
            image,
            on_progress=on_progress,
        )

        # Send prediction complete
        await manager.send_message(
            connection_id,
            WebSocketMessageType.PREDICTION_COMPLETE,
            PredictionCompleteData(
                prediction_id=result.prediction_id,
                video_url=result.video_url,
                thumbnail_url=result.thumbnail_url,
                frames=[f.model_dump(by_alias=True) for f in result.frames],
                metrics=result.metrics.model_dump(by_alias=True),
            ).model_dump(by_alias=True),
        )

    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        await manager.send_message(
            connection_id,
            WebSocketMessageType.ERROR,
            ErrorData(
                message=f"Prediction error: {e}",
                code="PREDICTION_ERROR",
            ).model_dump(by_alias=True),
        )


async def stream_concurrent_generation(
    connection_id: str,
    message_id: str,
    service: ModalInferenceService,
    message: str,
    image,
    images: list | None = None,
) -> None:
    """Stream concurrent text and video generation from Modal.

    This uses Modal's concurrent generation endpoint which produces
    both text tokens and video frames in an interleaved stream.

    Args:
        image: Primary image for video generation
        images: Optional list of all images for multi-image VLM context
    """
    prediction_id = str(uuid.uuid4())
    text_complete = False
    video_started = False
    frames_received: list[dict] = []
    settings = get_settings()

    try:
        async for item in service.generate_concurrent(message, image, images=images):
            item_type = item.get("type")

            if item_type == "text_chunk":
                # Send text chunk
                await manager.send_message(
                    connection_id,
                    WebSocketMessageType.TEXT_CHUNK,
                    TextChunkData(
                        message_id=message_id,
                        chunk=item.get("token", ""),
                        done=False,
                    ).model_dump(by_alias=True),
                )

            elif item_type == "text_done":
                # Send final text chunk
                text_complete = True
                await manager.send_message(
                    connection_id,
                    WebSocketMessageType.TEXT_CHUNK,
                    TextChunkData(
                        message_id=message_id,
                        chunk="",
                        done=True,
                    ).model_dump(by_alias=True),
                )

            elif item_type == "video_start":
                # Send prediction start
                video_started = True
                await manager.send_message(
                    connection_id,
                    WebSocketMessageType.PREDICTION_START,
                    PredictionStartData(prediction_id=prediction_id).model_dump(by_alias=True),
                )

            elif item_type == "video_progress":
                # Send progress update
                await manager.send_message(
                    connection_id,
                    WebSocketMessageType.PREDICTION_PROGRESS,
                    PredictionProgressData(
                        prediction_id=prediction_id,
                        progress=item.get("progress", 0),
                        current_frame=None,
                        total_frames=None,
                    ).model_dump(by_alias=True),
                )

            elif item_type == "video_frame":
                # Store frame and send progress
                frame_data = {
                    "index": item.get("index", len(frames_received)),
                    "frame_base64": item.get("frame_base64", ""),
                    "total": item.get("total", 30),
                }
                frames_received.append(frame_data)

                await manager.send_message(
                    connection_id,
                    WebSocketMessageType.PREDICTION_PROGRESS,
                    PredictionProgressData(
                        prediction_id=prediction_id,
                        progress=(frame_data["index"] + 1) / frame_data["total"] * 100,
                        current_frame=frame_data["index"],
                        total_frames=frame_data["total"],
                    ).model_dump(by_alias=True),
                )

            elif item_type == "video_done":
                # Send prediction complete with all frames
                frames = [
                    PredictionFrameData(
                        index=f["index"],
                        timestamp=f["index"] / settings.fps,
                        image_url=f"data:image/png;base64,{f['frame_base64']}",
                    )
                    for f in frames_received
                ]

                metrics = PredictionMetricsData(
                    lpips=0.20,  # Placeholder
                    confidence=0.85,
                    latency_ms=0,  # TODO: Track actual latency
                    spatial_iou=0.75 if settings.use_hybrid_encoder else None,
                )

                await manager.send_message(
                    connection_id,
                    WebSocketMessageType.PREDICTION_COMPLETE,
                    PredictionCompleteData(
                        prediction_id=prediction_id,
                        video_url=f"/api/static/video_{prediction_id}.mp4",
                        thumbnail_url=frames[0].image_url if frames else "",
                        frames=[f.model_dump(by_alias=True) for f in frames],
                        metrics=metrics.model_dump(by_alias=True),
                    ).model_dump(by_alias=True),
                )

            elif item_type == "error":
                # Handle error
                error_source = item.get("source", "unknown")
                error_msg = item.get("error", "Unknown error")
                logger.error(f"Concurrent generation error ({error_source}): {error_msg}")

                await manager.send_message(
                    connection_id,
                    WebSocketMessageType.ERROR,
                    ErrorData(
                        message=f"{error_source} error: {error_msg}",
                        code=f"{error_source.upper()}_ERROR",
                    ).model_dump(by_alias=True),
                )

            elif item_type == "complete":
                # Generation complete
                logger.info(f"Concurrent generation complete for {message_id}")

    except Exception as e:
        logger.error(f"Error in concurrent generation: {e}")
        await manager.send_message(
            connection_id,
            WebSocketMessageType.ERROR,
            ErrorData(
                message=f"Concurrent generation error: {e}",
                code="CONCURRENT_ERROR",
            ).model_dump(by_alias=True),
        )
