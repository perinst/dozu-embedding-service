import os
import numpy as np
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch


logging.basicConfig(level=logging.INFO)

MODEL_NAME = os.getenv("MODEL_NAME", "google/embeddinggemma-300m")
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
DEVICE_SETTING = os.getenv("DEVICE", "auto")

model: SentenceTransformer | None = None
loaded_model_name: str | None = None
loaded_device: str | None = None

app = FastAPI(title="Gemma Embedding Service", version="1.0.0")


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


def _resolve_device() -> str:
    req = DEVICE_SETTING.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return req
        logging.warning("Requested CUDA but not available. Falling back to CPU.")
        return "cpu"
    if req == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        logging.warning("Requested MPS but not available. Falling back to CPU.")
        return "cpu"
    return "cpu"


@app.on_event("startup")
def load_model():
    global model, loaded_model_name
    primary = MODEL_NAME
    device = _resolve_device()
    loaded_device = device
    fallback = "sentence-transformers/all-MiniLM-L6-v2"
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


def _encode(text: str):
    # Normalize so cosine similarity works directly
    emb = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return emb.tolist()


def _batch_encode(texts: list[str]):
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embs


@app.post("/embed/query", response_model=EmbeddingOut)
def embed_query(item: TextIn):
    ensure_model()
    return EmbeddingOut(embedding=_encode(item.text))


@app.post("/embed/document", response_model=EmbeddingOut)
def embed_document(item: TextIn):
    ensure_model()
    return EmbeddingOut(embedding=_encode(item.text))


@app.post("/similarity", response_model=SimilarityOut)
def similarity(payload: SimilarityIn):
    ensure_model()
    if not payload.documents:
        raise HTTPException(status_code=400, detail="documents list is empty")
    # Batch encode: first is query, rest are documents
    embs = _batch_encode([payload.query] + payload.documents)
    query_vec = embs[0]
    doc_vecs = embs[1:]
    # Since normalized, cosine = dot
    scores = (doc_vecs @ query_vec).tolist()
    best_idx = int(np.argmax(scores)) if scores else None
    best_score = scores[best_idx] if best_idx is not None else None
    return SimilarityOut(scores=scores, best_index=best_idx, best_score=best_score)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": loaded_model_name,
        "device": loaded_device,
        "device_requested": DEVICE_SETTING,
        "requires_token": MODEL_NAME == "google/embeddinggemma-300m",
    }
