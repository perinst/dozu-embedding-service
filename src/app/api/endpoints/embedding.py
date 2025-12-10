"""Embedding endpoints."""

import logging
from typing import Dict
from fastapi import APIRouter, HTTPException, Depends

from src.app.services.embedding.embedding_service import EmbeddingService

from ...schemas.embedding_schema import (
    SingleTextEmbeddingRequest,
    ListTextForEmbeddingRequest,
    CompareEmbeddingRequest,
)
from ..deps import get_embed_batch, get_embed_single

router = APIRouter(tags=["embedding"])
logging.basicConfig(level=logging.INFO)


@router.post("/single/text/embedding", response_model=Dict)
def embedding_single(
    req: SingleTextEmbeddingRequest,
    embed_single=Depends(get_embed_single),
):
    """
    Generate embedding for a single text.
    """
    try:
        embedding = EmbeddingService.embed_single_text(req.query, embed_single)
        return {"embedding": embedding}

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/segments/embedding", response_model=Dict)
def embedding_segments(
    req: ListTextForEmbeddingRequest,
    embed_batch=Depends(get_embed_batch),
):
    """
    Generate embeddings for a list of text segments.
    """
    try:
        embeddings = EmbeddingService.embed_batch_texts(req.segments, embed_batch)
        return embeddings

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/embedding/compare", response_model=Dict)
def embedding_compare(
    req: CompareEmbeddingRequest,
    embed_single=Depends(get_embed_single),
):
    """
    Compare two texts by calculating cosine similarity between their embeddings.
    """
    try:
        result = EmbeddingService.compare_embeddings(
            query=req.query,
            pattern=req.pattern,
            embed_single=embed_single,
        )
        return result

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error in embedding comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
