"""Dependencies for API endpoints."""

from ..core.model_manager import model_manager


def get_embed_batch():
    """Dependency to get batch embedding function."""
    return model_manager.embed_batch


def get_embed_single():
    """Dependency to get single embedding function."""
    return model_manager.embed_single
