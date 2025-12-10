"""Schemas for PDF processing endpoints."""

from pydantic import BaseModel


class PDFEmbeddingRequest(BaseModel):
    """Request schema for PDF embedding."""

    fileUrl: str
