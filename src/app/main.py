"""Main FastAPI application entrypoint."""

from fastapi import FastAPI
from .core.config import settings
from .core.model_manager import model_manager
from .api.api import api_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
    )

    # Include API router
    app.include_router(api_router)

    @app.on_event("startup")
    def startup_event():
        """Load the model on application startup."""
        model_manager.load_model()

    return app


# Create the app instance
app = create_app()
