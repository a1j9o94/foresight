"""API routes for the Foresight demo backend."""

from .routes import router
from .websocket import websocket_endpoint

__all__ = ["router", "websocket_endpoint"]
