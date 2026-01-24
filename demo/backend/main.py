"""FastAPI application for the Foresight demo backend."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import router, websocket_endpoint
from .config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()
    logger.info("Starting Foresight demo backend...")
    logger.info(f"Inference mode: {settings.inference_mode}")
    if settings.modal_mode:
        logger.info(f"Modal app: {settings.modal_app_name}")
    logger.info(f"Debug mode: {settings.debug}")

    # Create cache directory if needed
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize inference service
    from .services import get_inference_service
    service = get_inference_service()
    status = await service.get_status()
    logger.info(f"Inference service status: {status}")

    yield

    logger.info("Shutting down Foresight demo backend...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Foresight Demo API",
        description="Backend API for the Foresight demo - AI that sees the future",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router)

    # WebSocket endpoint
    app.websocket("/ws/chat")(websocket_endpoint)

    # Serve static files if directory exists
    if settings.static_dir.exists():
        app.mount(
            "/static",
            StaticFiles(directory=settings.static_dir),
            name="static",
        )

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "demo.backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
