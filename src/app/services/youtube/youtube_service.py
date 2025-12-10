"""Business logic for YouTube transcript processing."""

import logging
from typing import Optional, Dict, Any, Callable, List

from src.app.services.youtube.youtube_pipeline import (
    full_pipeline,
    search_sentences,
    segment_fit_sentence,
)


logging.basicConfig(level=logging.INFO)


class YouTubeService:
    """Service for processing YouTube transcripts and embeddings."""

    @staticmethod
    def process_video_transcript(
        video_id: str,
        embed_batch: Callable,
        embed_single: Callable,
        languages: Optional[List[str]] = None,
        max_gap: float = 1.5,
        min_length: int = 5,
        refine: bool = False,
        proxy: Optional[Dict[str, Any]] = None,
        preserve_formatting: bool = False,
        query: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Process a YouTube video transcript through the full pipeline.

        Args:
            video_id: YouTube video ID or URL
            embed_batch: Function to embed batch of texts
            embed_single: Function to embed single text
            languages: List of language codes
            max_gap: Maximum time gap between segments
            min_length: Minimum sentence length
            refine: Whether to refine sentences
            proxy: Proxy configuration
            preserve_formatting: Whether to preserve formatting
            query: Optional search query
            top_k: Number of top results to return

        Returns:
            Dictionary with processed sentences and optional search results
        """
        proxy_dict = proxy if proxy else None

        result = full_pipeline(
            video_id=video_id,
            embed_batch=embed_batch,
            embed_single=embed_single,
            languages=languages,
            max_gap=max_gap,
            min_length=min_length,
            refine=refine,
            proxy=proxy_dict,
            preserve_formatting=preserve_formatting,
        )

        query_results = None
        if query:
            query_results = search_sentences(
                query, result["sentences"], embed_single, top_k=top_k
            )

        return {
            "video_id": result["video_id"],
            "sentence_count": result["sentence_count"],
            "sentences": result["sentences"],
            "query_results": query_results,
        }

    @staticmethod
    def process_segments_embedding(
        segments: List[Dict[str, Any]],
        embed_batch: Callable,
        embed_single: Callable,
        languages: Optional[List[str]] = None,
        max_gap: float = 1.5,
        min_length: int = 5,
        refine: bool = False,
    ) -> Dict[str, Any]:
        """
        Process raw segments and generate embeddings.

        Args:
            segments: List of segment dictionaries
            embed_batch: Function to embed batch of texts
            embed_single: Function to embed single text
            languages: List of language codes
            max_gap: Maximum time gap between segments
            min_length: Minimum sentence length
            refine: Whether to refine sentences

        Returns:
            Dictionary with sentence count and embeddings
        """
        result = segment_fit_sentence(
            raw=segments,
            embed_batch=embed_batch,
            embed_single=embed_single,
            languages=languages,
            max_gap=max_gap,
            min_length=min_length,
            refine=refine,
        )

        return {
            "count": result["sentence_count"],
            "embeddings": result["sentences"],
        }

    @staticmethod
    def process_segments_similarity(
        segments: List[Dict[str, Any]],
        embed_batch: Callable,
        embed_single: Callable,
        query: Optional[str],
        languages: Optional[List[str]] = None,
        max_gap: float = 1.5,
        min_length: int = 5,
        refine: bool = False,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Process segments and calculate similarity with query.

        Args:
            segments: List of segment dictionaries
            embed_batch: Function to embed batch of texts
            embed_single: Function to embed single text
            query: Search query
            languages: List of language codes
            max_gap: Maximum time gap between segments
            min_length: Minimum sentence length
            refine: Whether to refine sentences
            top_k: Number of top results to return

        Returns:
            Dictionary with sentence count and query results
        """
        result = segment_fit_sentence(
            raw=segments,
            embed_batch=embed_batch,
            embed_single=embed_single,
            languages=languages,
            max_gap=max_gap,
            min_length=min_length,
            refine=refine,
        )

        query_results = None
        if query:
            query_results = search_sentences(
                query, result["sentences"], embed_single, top_k=top_k
            )

        return {
            "sentence_count": result["sentence_count"],
            "query_results": query_results,
        }
