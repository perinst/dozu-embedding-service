"""YouTube transcript processing and sentence embedding pipeline.

Single-responsibility functions to:
1. Fetch raw transcript segments.
2. Clean segments (remove empty/malformed).
3. Merge short fragments into full sentences using time gap + punctuation.
4. (Optional) Refine sentences via spaCy if available.
5. Embed sentences using a provided embedding function.
6. Search sentences by cosine similarity (assuming normalized embeddings).

Design notes:
- No direct dependency on global FastAPI app; pure functions for testability.
- Embedding is injected (dependency inversion) via `embed_batch` callable.
- Timestamps: we carry start and end seconds from merged ranges.
"""

from __future__ import annotations
from typing import List, Callable, Optional, Sequence, Dict, Any

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

try:
    import spacy

    _NLP = None
except ImportError:
    spacy = None
    _NLP = None


def get_transcript_segments(
    video_id: str, languages: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Fetch transcript segments for a YouTube video.

    Each segment has keys: text, start (seconds), duration.
    Raises RuntimeError if library not installed or transcript unavailable.
    """
    if YouTubeTranscriptApi is None:
        raise RuntimeError(
            "youtube-transcript-api not installed. Please add it to requirements and install."
        )
    langs = languages or ["en"]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to fetch transcript: {e}")
    return transcript


def clean_segments(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove empty text entries and normalize whitespace."""
    cleaned = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        cleaned.append(
            {
                "text": text,
                "start": float(seg.get("start", 0.0)),
                "duration": float(seg.get("duration", 0.0)),
            }
        )
    return cleaned


def merge_segments(
    segments: Sequence[Dict[str, Any]], max_gap: float = 1.5, min_length: int = 5
) -> List[Dict[str, Any]]:
    """Merge short transcript fragments into sentence-like units.

    Rules:
    - Consecutive segments appended until punctuation (., ?, !) or time gap > max_gap.
    - Ensures minimum word count (min_length); if below, will try to absorb next segment even if punctuation.
    - Each merged item: {start, end, text}
    """
    segs = list(segments)
    if not segs:
        return []
    merged: List[Dict[str, Any]] = []
    current_text: List[str] = []
    current_start = float(segs[0]["start"])

    def flush(end_time: float):
        if not current_text:
            return
        merged.append(
            {
                "start": current_start,
                "end": end_time,
                "startMs": int(round(current_start * 1000)),
                "endMs": int(round(end_time * 1000)),
                "text": " ".join(current_text).strip(),
            }
        )

    for i, seg in enumerate(segs):
        text = seg["text"].strip()
        start_time = float(seg["start"])
        end_time = start_time + float(seg.get("duration", 0.0))
        if not current_text:
            current_start = start_time
        current_text.append(text)

        # Compute gap to next
        if i + 1 < len(segs):
            next_start = float(segs[i + 1]["start"])
            gap = next_start - start_time
        else:
            gap = 0.0

        sentence_so_far = " ".join(current_text)
        word_count = len(sentence_so_far.split())
        is_terminal_punct = text.endswith((".", "?", "!"))
        should_break = False
        if is_terminal_punct and word_count >= min_length:
            should_break = True
        if gap > max_gap:
            should_break = True

        if should_break:
            flush(end_time)
            current_text = []

    # Flush any remainder using last segment end
    if current_text:
        last_seg = segs[-1]
        last_end = float(last_seg["start"]) + float(last_seg.get("duration", 0.0))
        flush(last_end)

    return merged


def refine_sentences(merged: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Optionally split merged text further using spaCy sentence boundary detection.

    If spaCy or model unavailable, returns merged as-is.
    """
    if spacy is None:
        return list(merged)
    global _NLP
    if _NLP is None:  # lazy load
        try:  # pragma: no cover
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            return list(merged)
    refined: List[Dict[str, Any]] = []
    for m in merged:
        doc = _NLP(m["text"])  # type: ignore
        for sent in doc.sents:
            refined.append(
                {
                    "start": m["start"],
                    "end": m["end"],
                    "startMs": m.get("startMs"),
                    "endMs": m.get("endMs"),
                    "text": sent.text.strip(),
                }
            )
    return refined


def embed_sentences(
    sentences: Sequence[Dict[str, Any]], embed_batch: Callable[[List[str]], Any]
) -> List[Dict[str, Any]]:
    """Add embeddings to each sentence dict using provided batch embedding function.

    Assumes embed_batch returns a numpy array or list of lists already normalized.
    """
    texts = [s["text"] for s in sentences]
    vectors = embed_batch(texts)
    # Convert to list of floats per sentence (handle numpy array)
    try:
        vectors_list = vectors.tolist()  # type: ignore[attr-defined]
    except AttributeError:
        vectors_list = vectors
    enriched: List[Dict[str, Any]] = []
    for sent, vec in zip(sentences, vectors_list):
        enriched.append({**sent, "embedding": vec})
    return enriched


def search_sentences(
    query: str,
    sentences_with_embeddings: Sequence[Dict[str, Any]],
    embed_query: Callable[[str], Any],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Return top_k most similar sentences to the query.

    Assumes embeddings are normalized so cosine similarity reduces to dot product.
    Returns list of {index, score, start, end, text} sorted by score desc.
    """
    if not sentences_with_embeddings:
        return []
    q_vec = embed_query(query)
    try:
        # q_vec could be list; convert
        import numpy as np  # local import

        q_arr = np.array(q_vec, dtype=float)
        sent_arr = np.array(
            [s["embedding"] for s in sentences_with_embeddings], dtype=float
        )
        scores = sent_arr @ q_arr
        top_indices = scores.argsort()[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            s = sentences_with_embeddings[idx]
            results.append(
                {
                    "index": int(idx),
                    "score": float(scores[idx]),
                    "start": s["start"],
                    "end": s["end"],
                    "startMs": s.get("startMs"),
                    "endMs": s.get("endMs"),
                    "text": s["text"],
                }
            )
        return results
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Similarity computation failed: {e}")


def full_pipeline(
    video_id: str,
    embed_batch: Callable[[List[str]], Any],
    embed_single: Callable[[str], Any],
    languages: Optional[List[str]] = None,
    max_gap: float = 1.5,
    min_length: int = 5,
    refine: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end pipeline and return structured result.

    Returns dict with keys:
    - video_id
    - sentences: list of {start, end, text, embedding}
    Metadata: counts etc.
    """
    raw = get_transcript_segments(video_id, languages=languages)
    cleaned = clean_segments(raw)
    merged = merge_segments(cleaned, max_gap=max_gap, min_length=min_length)
    if refine:
        merged = refine_sentences(merged)
    sentences_with_embeddings = embed_sentences(merged, embed_batch)
    return {
        "video_id": video_id,
        "sentence_count": len(sentences_with_embeddings),
        "sentences": sentences_with_embeddings,
    }


__all__ = [
    "get_transcript_segments",
    "clean_segments",
    "merge_segments",
    "refine_sentences",
    "embed_sentences",
    "search_sentences",
    "full_pipeline",
]
