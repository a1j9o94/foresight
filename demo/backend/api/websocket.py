"""WebSocket endpoint for streaming chat and predictions."""

import asyncio
import logging
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from ..models.schemas import (
    WebSocketMessage,
    WebSocketMessageType,
    TextChunkData,
    PredictionStartData,
    PredictionProgressData,
    PredictionCompleteData,
    ErrorData,
)
from ..services import get_inference_service
from ..services.inference import decode_image

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
        message = WebSocketMessage(type=message_type, data=data)
        await websocket.send_json(message.model_dump())


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

            if not message_id:
                await manager.send_message(
                    connection_id,
                    WebSocketMessageType.ERROR,
                    ErrorData(message="Missing messageId", code="MISSING_MESSAGE_ID").model_dump(),
                )
                continue

            # Process the message
            await process_chat_message(
                connection_id=connection_id,
                message=message,
                message_id=message_id,
                image_base64=image_base64,
            )

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await manager.send_message(
                connection_id,
                WebSocketMessageType.ERROR,
                ErrorData(message=str(e), code="INTERNAL_ERROR").model_dump(),
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
) -> None:
    """Process a chat message and send streaming responses."""
    service = get_inference_service()

    try:
        # Decode image if provided
        image = None
        if image_base64:
            image = await decode_image(image_base64)

        # Start text generation and prediction in parallel
        text_task = asyncio.create_task(
            stream_text_response(connection_id, message_id, service, message, image)
        )

        prediction_task = None
        if image:
            prediction_task = asyncio.create_task(
                stream_prediction(connection_id, service, message, image)
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
            ErrorData(message=str(e), code="PROCESSING_ERROR").model_dump(),
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
                ).model_dump(),
            )
    except Exception as e:
        logger.error(f"Error streaming text: {e}")
        await manager.send_message(
            connection_id,
            WebSocketMessageType.ERROR,
            ErrorData(message=f"Text generation error: {e}", code="TEXT_ERROR").model_dump(),
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
            PredictionStartData(prediction_id=prediction_id).model_dump(),
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
                ).model_dump(),
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
                frames=[f.model_dump() for f in result.frames],
                metrics=result.metrics.model_dump(),
            ).model_dump(),
        )

    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        await manager.send_message(
            connection_id,
            WebSocketMessageType.ERROR,
            ErrorData(
                message=f"Prediction error: {e}",
                code="PREDICTION_ERROR",
            ).model_dump(),
        )
