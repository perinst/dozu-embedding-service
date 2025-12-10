"""YouTube transcript processing endpoints."""

from typing import Dict
from fastapi import APIRouter, HTTPException, Depends

from ...schemas.youtube_schema import (
    YouTubePipelineRequest,
    YouTubePipelineResponse,
    YouTubeSegmentSentenceCalSimilarityRequest,
    YouTubeSentence,
)
from ...services.youtube.youtube_service import YouTubeService
from ..deps import get_embed_batch, get_embed_single

router = APIRouter(prefix="/youtube", tags=["youtube"])


@router.post("/segments", response_model=YouTubePipelineResponse)
def youtube_segments(
    req: YouTubePipelineRequest,
    embed_batch=Depends(get_embed_batch),
    embed_single=Depends(get_embed_single),
):
    """
    Fetch YouTube transcript, merge into sentences, embed, and optionally search.
    """
    try:
        proxy_dict = req.proxy.dict() if req.proxy else None

        result = YouTubeService.process_video_transcript(
            video_id=req.video_id,
            embed_batch=embed_batch,
            embed_single=embed_single,
            languages=req.languages,
            max_gap=req.max_gap,
            min_length=req.min_length,
            refine=req.refine,
            proxy=proxy_dict,
            preserve_formatting=req.preserve_formatting,
            query=req.query,
            top_k=req.top_k or 5,
        )

        sentences = [YouTubeSentence(**s) for s in result["sentences"]]

        return YouTubePipelineResponse(
            video_id=result["video_id"],
            sentence_count=result["sentence_count"],
            sentences=sentences,
            query_results=result["query_results"],
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/segments/embedding", response_model=Dict)
def embedding_youtube_segments(
    req: YouTubeSegmentSentenceCalSimilarityRequest,
    embed_batch=Depends(get_embed_batch),
    embed_single=Depends(get_embed_single),
):
    """
    Process YouTube segments and generate embeddings.
    """
    try:
        result = YouTubeService.process_segments_embedding(
            segments=req.segments,
            embed_batch=embed_batch,
            embed_single=embed_single,
            languages=req.languages,
            max_gap=req.max_gap,
            min_length=req.min_length,
            refine=req.refine,
        )

        sentences = [YouTubeSentence(**s) for s in result["embeddings"]]

        return {
            "count": result["count"],
            "embeddings": sentences,
        }

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/segments/similarity", response_model=Dict)
def youtube_segments_similarity(
    req: YouTubeSegmentSentenceCalSimilarityRequest,
    embed_batch=Depends(get_embed_batch),
    embed_single=Depends(get_embed_single),
):
    """
    Process YouTube segments and calculate similarity with query.
    """
    try:
        result = YouTubeService.process_segments_similarity(
            segments=req.segments,
            embed_batch=embed_batch,
            embed_single=embed_single,
            query=req.query,
            languages=req.languages,
            max_gap=req.max_gap,
            min_length=req.min_length,
            refine=req.refine,
            top_k=req.top_k or 5,
        )

        return {
            "sentence_count": result["sentence_count"],
            "query_results": result["query_results"],
        }

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
