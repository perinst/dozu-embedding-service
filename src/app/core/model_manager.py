"""Model management for embedding service."""

import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
from .config import settings

logging.basicConfig(level=logging.INFO)


class ModelManager:
    """Manages the loading and access of the sentence transformer model."""

    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.loaded_model_name: Optional[str] = None
        self.loaded_device: Optional[str] = None

    def _resolve_device(self) -> str:
        """Resolve device setting."""
        return "cpu"

    def load_model(self):
        """Load the embedding model at startup."""
        primary = settings.MODEL_NAME
        device = self._resolve_device()
        self.loaded_device = device
        fallback = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        try:
            logging.info(f"Loading model: {primary}")
            self.model = SentenceTransformer(
                primary, token=settings.HF_TOKEN, device=device
            )
            self.loaded_model_name = primary
            logging.info(f"Loaded model: {primary}")
        except Exception as e:
            logging.warning(f"Failed to load {primary}: {e}")
            logging.info(f"Falling back to: {fallback}")
            self.model = SentenceTransformer(fallback, device=device)
            self.loaded_model_name = fallback
            logging.info(f"Loaded fallback model: {fallback}")

    def ensure_model(self):
        """Ensure model is loaded."""
        if self.model is None:
            raise RuntimeError("Model not loaded yet")

    def embed_batch(self, texts: list[str]):
        """Embed a batch of texts."""
        self.ensure_model()
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_single(self, text: str):
        """Embed a single text."""
        self.ensure_model()
        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    def get_health_info(self) -> dict:
        """Get health information about the model."""
        return {
            "status": "ok",
            "model": self.loaded_model_name,
            "device": self.loaded_device,
            "device_requested": settings.DEVICE_SETTING,
            "requires_token": settings.MODEL_NAME == "google/embeddinggemma-300m",
        }


# Global model manager instance
model_manager = ModelManager()
