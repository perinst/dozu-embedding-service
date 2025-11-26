import os
import numpy as np
import logging
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


from youtube_pipeline import (
    embed_sentences,
    full_pipeline,
    search_sentences,
    segment_fit_sentence,
)
from file_pdf_pipeline import process_pdf_from_url


logging.basicConfig(level=logging.INFO)

MODEL_NAME = os.getenv("MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
DEVICE_SETTING = os.getenv("DEVICE", "auto")

model: SentenceTransformer | None = None
loaded_model_name: str | None = None
loaded_device: str | None = None

app = FastAPI(title="Dozu Embedding Service", version="1.0.0")


class TextIn(BaseModel):
    text: str


class EmbeddingOut(BaseModel):
    embedding: list[float]


class SimilarityIn(BaseModel):
    query: str
    documents: List[str]


class SimilarityOut(BaseModel):
    scores: list[float]
    best_index: int | None
    best_score: float | None


class ProxyConfig(BaseModel):
    kind: str | None = None  # 'webshare' or 'generic'
    proxy_username: str | None = None
    proxy_password: str | None = None
    filter_ip_locations: list[str] | None = None
    http_url: str | None = None
    https_url: str | None = None


class YouTubePipelineRequest(BaseModel):
    video_id: str  # can be full URL; will be parsed
    languages: list[str] | None = None
    refine: bool = False
    max_gap: float = 1.5
    min_length: int = 5
    top_k: int | None = None
    query: str | None = None
    preserve_formatting: bool = False
    proxy: ProxyConfig | None = None


class YouTubeSegmentSentenceCalSimilarityRequest(BaseModel):
    segments: List[Dict[str, Any]]
    languages: list[str] | None = None
    refine: bool = False
    max_gap: float = 1.5
    min_length: int = 5
    top_k: int | None = None
    query: str | None = None


class ListTextForEmbeddingRequest(BaseModel):
    segments: List[Dict[str, Any]]


class SingleTextEmbeddingRequest(BaseModel):
    query: str


class CompareEmbeddingRequest(BaseModel):
    pattern: str
    query: str


class PDFEmbeddingRequest(BaseModel):
    fileUrl: str


class YouTubeSentence(BaseModel):
    start: float
    text: str
    embedding: list[float]


class YouTubePipelineResponse(BaseModel):
    video_id: str
    sentence_count: int
    sentences: list[YouTubeSentence]
    query_results: list[dict] | None = None


class YouTubePipelineSentenceCalSimilarityResponse(BaseModel):
    sentence_count: int
    sentences: list[YouTubeSentence]
    query_results: list[dict] | None = None


def _resolve_device() -> str:
    return "cpu"


@app.on_event("startup")
def load_model():
    global model, loaded_model_name, loaded_device
    primary = MODEL_NAME
    device = _resolve_device()
    loaded_device = device
    fallback = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    try:
        logging.info(f"Loading model: {primary}")
        model = SentenceTransformer(primary, token=HF_TOKEN, device=device)
        loaded_model_name = primary
        logging.info(f"Loaded model: {primary}")
    except Exception as e:
        logging.warning(f"Failed to load {primary}: {e}")
        logging.info(f"Falling back to: {fallback}")
        model = SentenceTransformer(fallback, device=device)
        loaded_model_name = fallback
        logging.info(f"Loaded fallback model: {fallback}")


def ensure_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": loaded_model_name,
        "device": loaded_device,
        "device_requested": DEVICE_SETTING,
        "requires_token": MODEL_NAME == "google/embeddinggemma-300m",
    }


def _embed_batch(texts: list[str]):
    ensure_model()
    # Using model.encode for batch with normalization for cosine similarity
    return model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )


def _embed_single(text: str):
    ensure_model()
    return model.encode(
        text, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).tolist()


@app.post("/youtube/segments", response_model=YouTubePipelineResponse)
def youtube_segments(req: YouTubePipelineRequest):
    """Fetch YouTube transcript, merge into sentences, embed, optionally search."""
    ensure_model()
    try:
        proxy_dict = req.proxy.dict() if req.proxy else None
        result = full_pipeline(
            video_id=req.video_id,
            embed_batch=_embed_batch,
            embed_single=_embed_single,
            languages=req.languages,
            max_gap=req.max_gap,
            min_length=req.min_length,
            refine=req.refine,
            proxy=proxy_dict,
            preserve_formatting=req.preserve_formatting,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sentences = [YouTubeSentence(**s) for s in result["sentences"]]
    query_results = None
    if req.query:
        # Perform search over sentences
        query_results = search_sentences(
            req.query, result["sentences"], _embed_single, top_k=req.top_k or 5
        )

    return YouTubePipelineResponse(
        video_id=result["video_id"],
        sentence_count=result["sentence_count"],
        sentences=sentences,
        query_results=query_results,
    )


@app.post("/youtube/segments/embedding", response_model=Dict)
def embedding_youtube_segments(req: YouTubeSegmentSentenceCalSimilarityRequest):
    ensure_model()

    try:
        result = segment_fit_sentence(
            raw=req.segments,
            embed_batch=_embed_batch,
            embed_single=_embed_single,
            languages=req.languages,
            max_gap=req.max_gap,
            min_length=req.min_length,
            refine=req.refine,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sentences = [YouTubeSentence(**s) for s in result["sentences"]]

    return {
        "count": result["sentence_count"],
        "embeddings": sentences,
    }


@app.post(
    "/youtube/segments/similarity",
    response_model=Dict,
)
def youtube_segments(req: YouTubeSegmentSentenceCalSimilarityRequest):
    ensure_model()

    try:
        result = segment_fit_sentence(
            raw=req.segments,
            embed_batch=_embed_batch,
            embed_single=_embed_single,
            languages=req.languages,
            max_gap=req.max_gap,
            min_length=req.min_length,
            refine=req.refine,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    query_results = None

    if req.query:
        query_results = search_sentences(
            req.query, result["sentences"], _embed_single, top_k=req.top_k or 5
        )

    return {
        "sentence_count": result["sentence_count"],
        "query_results": query_results,
    }


@app.post(
    "/single/text/embedding",
    response_model=Dict,
)
def embedding_single(req: SingleTextEmbeddingRequest):
    try:
        query = req.query

        embedding = _embed_single(query)

        return {"embedding": embedding}

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/embedding/compare", response_model=Dict)
def embedding_compare(req: CompareEmbeddingRequest):
    """
    Compare two texts by calculating cosine similarity between their embeddings.

    Args:
        req: Request containing pattern and query texts

    Returns:
        Dictionary with similarity score and both embedding vectors
    """
    ensure_model()

    try:
        query = req.query
        pattern = req.pattern

        query_embedding = _embed_single(query)
        pattern_embedding = _embed_single(pattern)

        similarity_score = float(np.dot(query_embedding, pattern_embedding))

        return {
            "similarity": similarity_score,
            "queryEmbedding": query_embedding,
            "patternEmbedding": pattern_embedding,
        }

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error in embedding comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post(
    "/segments/embedding",
    response_model=Dict,
)
def embedding_segments(req: ListTextForEmbeddingRequest):
    try:
        listToEmbed = req.segments

        embeddings = embed_sentences(listToEmbed, _embed_batch)

        return embeddings

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/pdf/page/embedding",
    response_model=Dict,
)
def pdf_page_embedding(req: PDFEmbeddingRequest):
    """
    Process PDF from URL and generate embeddings for each page.

    Args:
        req: Request containing fileUrl

    Returns:
        Dictionary with embeddings array containing page-by-page embeddings
    """
    ensure_model()

    try:
        # Process PDF using the pipeline
        result = process_pdf_from_url(req.fileUrl, _embed_batch)

        # Format response to match requirement: { embeddings: [...] }
        embeddings = [
            {
                "pageNumber": page["page_number"],
                "charCount": page["char_count"],
                "embedding": page["embedding"],
            }
            for page in result["pages"]
        ]

        return {
            "embeddings": embeddings,
            "pageCount": result["page_count"],
        }

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error in PDF embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
