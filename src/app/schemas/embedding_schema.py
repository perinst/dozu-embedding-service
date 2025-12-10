"""Schemas for embedding endpoints."""

from typing import List, Dict, Any
from pydantic import BaseModel


class SingleTextEmbeddingRequest(BaseModel):
    """Request schema for single text embedding."""

    query: str


class ListTextForEmbeddingRequest(BaseModel):
    """Request schema for batch text embedding."""

    segments: List[Dict[str, Any]]


class CompareEmbeddingRequest(BaseModel):
    """Request schema for comparing two embeddings."""

    pattern: str
    query: str
