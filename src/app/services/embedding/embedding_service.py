"""Business logic for embedding operations."""

import logging
import numpy as np
from typing import Callable, List, Dict, Any

from src.app.services.youtube.youtube_pipeline import embed_sentences


logging.basicConfig(level=logging.INFO)


class EmbeddingService:
    """Service for text embedding operations."""

    @staticmethod
    def embed_single_text(text: str, embed_single: Callable) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed
            embed_single: Function to embed single text

        Returns:
            Embedding vector as list of floats
        """
        return embed_single(text)

    @staticmethod
    def embed_batch_texts(
        segments: List[Dict[str, Any]], embed_batch: Callable
    ) -> Dict[str, Any]:
        """
        Embed a batch of text segments.

        Args:
            segments: List of text segments
            embed_batch: Function to embed batch of texts

        Returns:
            Dictionary with embeddings
        """
        embeddings = embed_sentences(segments, embed_batch)
        return embeddings

    @staticmethod
    def compare_embeddings(
        query: str,
        pattern: str,
        embed_single: Callable,
    ) -> Dict[str, Any]:
        """
        Compare two texts by calculating cosine similarity between their embeddings.

        Args:
            query: First text to compare
            pattern: Second text to compare
            embed_single: Function to embed single text

        Returns:
            Dictionary with similarity score and both embeddings
        """
        query_embedding = embed_single(query)
        pattern_embedding = embed_single(pattern)

        # Calculate cosine similarity (embeddings are already normalized)
        similarity_score = float(np.dot(query_embedding, pattern_embedding))

        return {
            "similarity": similarity_score,
            "queryEmbedding": query_embedding,
            "patternEmbedding": pattern_embedding,
        }
