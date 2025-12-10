"""Health check endpoint."""

from fastapi import APIRouter
from ...core.model_manager import model_manager

router = APIRouter()


@router.get("/health")
def health_check():
    """Check service health and model status."""
    return model_manager.get_health_info()
