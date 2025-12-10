"""Schemas for YouTube transcript processing endpoints."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from .common_schema import ProxyConfig


class YouTubePipelineRequest(BaseModel):
    """Request schema for YouTube transcript pipeline."""

    video_id: str  # can be full URL; will be parsed
    languages: Optional[List[str]] = None
    refine: bool = False
    max_gap: float = 1.5
    min_length: int = 5
    top_k: Optional[int] = None
    query: Optional[str] = None
    preserve_formatting: bool = False
    proxy: Optional[ProxyConfig] = None


class YouTubeSegmentSentenceCalSimilarityRequest(BaseModel):
    """Request schema for YouTube segment sentence similarity calculation."""

    segments: List[Dict[str, Any]]
    languages: Optional[List[str]] = None
    refine: bool = False
    max_gap: float = 1.5
    min_length: int = 5
    top_k: Optional[int] = None
    query: Optional[str] = None


class YouTubeSentence(BaseModel):
    """Schema for a YouTube sentence with embedding."""

    start: float
    text: str
    embedding: List[float]


class YouTubePipelineResponse(BaseModel):
    """Response schema for YouTube pipeline."""

    video_id: str
    sentence_count: int
    sentences: List[YouTubeSentence]
    query_results: Optional[List[dict]] = None


class YouTubePipelineSentenceCalSimilarityResponse(BaseModel):
    """Response schema for YouTube sentence similarity calculation."""

    sentence_count: int
    sentences: List[YouTubeSentence]
    query_results: Optional[List[dict]] = None
