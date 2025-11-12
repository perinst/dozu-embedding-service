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
import re
import numpy as np

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


def extract_video_id(video_or_url: str) -> str:
    """Extract a YouTube video ID from a URL or return the input if it already looks like an ID."""
    s = video_or_url.strip()
    m = re.search(r"youtu\.be/([\w-]{11})", s)
    if m:
        return m.group(1)
    m = re.search(r"v=([\w-]{11})", s)
    if m:
        return m.group(1)
    m = re.search(r"/embed/([\w-]{11})", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"[\w-]{11}", s):
        return s
    m = re.search(r"([\w-]{11})(?:\?.*)?$", s)
    if m:
        return m.group(1)
    return s


def get_transcript_segments(
    video_id_or_url: str,
    languages: Optional[List[str]] = None,
    proxy: Optional[Dict[str, Any]] = None,
    preserve_formatting: bool = False,
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
    video_id = extract_video_id(video_id_or_url)

    # Configure proxy if provided
    api_kwargs: Dict[str, Any] = {}
    if proxy:
        try:
            from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig  # type: ignore

            kind = (proxy.get("kind") or "").lower()
            if kind == "webshare":
                api_kwargs["proxy_config"] = WebshareProxyConfig(
                    proxy_username=proxy.get("proxy_username"),
                    proxy_password=proxy.get("proxy_password"),
                    filter_ip_locations=proxy.get("filter_ip_locations"),
                )
            elif kind == "generic":
                api_kwargs["proxy_config"] = GenericProxyConfig(
                    http_url=proxy.get("http_url"),
                    https_url=proxy.get("https_url"),
                )
        except Exception:
            api_kwargs = {}

    try:
        api = YouTubeTranscriptApi(**api_kwargs)
        fetched = api.fetch(
            video_id, languages=langs, preserve_formatting=preserve_formatting
        )
        if hasattr(fetched, "to_raw_data"):
            return fetched.to_raw_data()  # list of dicts
        # If no to_raw_data, try to iterate and build dicts
        return [
            {
                "text": getattr(snippet, "text", ""),
                "start": float(getattr(snippet, "start", 0.0)),
                "duration": float(getattr(snippet, "duration", 0.0)),
            }
            for snippet in fetched
        ]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch transcript: {e}")


def clean_segments(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove empty text entries and normalize whitespace, while preserving timing.
    We assume input provides only start times and estimate end times as:
        end ≈ start + (word_count * ms_per_word) / 1000
    where ms_per_word defaults to 400ms. Any provided end/duration fields are ignored.
    Produces normalized dicts with: text, start, startMs, end, endMs.
    """
    cleaned: List[Dict[str, Any]] = []

    MS_PER_WORD_DEFAULT = 400.0

    for seg in segments:
        if isinstance(seg, dict):
            text = (seg.get("text") or "").strip()
            if not text:
                continue

            # Resolve start (prefer seconds)
            start_val = seg.get("start")
            if start_val is None:
                start_val = seg.get("startSecond")
            if start_val is None and seg.get("startMs") is not None:
                start_val = float(seg.get("startMs", 0.0)) / 1000.0
            start = float(start_val or 0.0)

            # Estimate end time purely from word count and ms-per-word
            word_count = len(text.split())
            duration_sec = (word_count * MS_PER_WORD_DEFAULT) / 1000.0
            end: float = start + float(duration_sec)

            item: Dict[str, Any] = {
                "text": text,
                "start": start,
                "startMs": int(round(start * 1000)),
            }
            item["end"] = end
            item["endMs"] = int(round(end * 1000))
            cleaned.append(item)
        else:
            # Object-like segment
            text = getattr(seg, "text", "").strip()
            if not text:
                continue
            start_val = getattr(seg, "start", None)
            if start_val is None:
                start_val = getattr(seg, "startSecond", 0.0)
            start = float(start_val or 0.0)
            # Estimate end time from word count and ms-per-word
            word_count = len(text.split())
            duration_sec = (word_count * MS_PER_WORD_DEFAULT) / 1000.0
            end: float = start + float(duration_sec)
            item: Dict[str, Any] = {
                "text": text,
                "start": start,
                "startMs": int(round(start * 1000)),
            }
            item["end"] = end
            item["endMs"] = int(round(end * 1000))
            cleaned.append(item)
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

    current_end: Optional[float] = float(segs[0].get("end", current_start))

    def flush():
        if not current_text:
            return
        out: Dict[str, Any] = {
            "start": current_start,
            "startMs": int(round(current_start * 1000)),
            "text": " ".join(current_text).strip(),
        }
        if current_end is not None:
            out["end"] = current_end
            out["endMs"] = int(round(current_end * 1000))
        merged.append(out)

    for i, seg in enumerate(segs):
        text = seg["text"].strip()

        start_time = float(seg["start"])
        end_time = float(seg.get("end", start_time))

        if not current_text:
            current_start = start_time
            current_end = end_time
        current_text.append(text)
        # advance current_end to the end of this piece (when available)
        if end_time is not None:
            current_end = end_time

        # Compute gap to next
        if i + 1 < len(segs):
            next_start = float(segs[i + 1]["start"])
            # If we know the end_time of the current piece, compute actual gap
            # between the end of current and the next start; otherwise fall back
            # to start-to-start difference.
            if end_time is not None:
                gap = next_start - end_time
            else:
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
            flush()
            current_text = []

    # Flush any remainder using last segment end
    if current_text:
        flush()

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
        # local import

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
                    "startMs": s.get("startMs"),
                    "text": s["text"],
                }
            )
        return results
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Similarity computation failed: {e}")


def segment_fit_sentence(
    raw: List[Dict[str, Any]],
    embed_batch: Callable[[List[str]], Any],
    embed_single: Callable[[str], Any],
    languages: Optional[List[str]] = None,
    max_gap: float = 2.5,
    min_length: int = 5,
    refine: bool = False,
) -> Dict[str, Any]:

    cleaned = clean_segments(raw)

    merged = merge_segments(cleaned, max_gap=max_gap, min_length=min_length)

    if refine:
        merged = refine_sentences(merged)

    sentences_with_embeddings = embed_sentences(merged, embed_batch)

    return {
        "sentence_count": len(sentences_with_embeddings),
        "sentences": sentences_with_embeddings,
    }


def full_pipeline(
    video_id: str,
    embed_batch: Callable[[List[str]], Any],
    embed_single: Callable[[str], Any],
    languages: Optional[List[str]] = None,
    max_gap: float = 2.5,
    min_length: int = 5,
    refine: bool = False,
    proxy: Optional[Dict[str, Any]] = None,
    preserve_formatting: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end pipeline and return structured result.

    Returns dict with keys:
    - video_id
    - sentences: list of {start, end, text, embedding}
    Metadata: counts etc.
    """
    raw = get_transcript_segments(
        video_id,
        languages=languages,
        proxy=proxy,
        preserve_formatting=preserve_formatting,
    )
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
    "extract_video_id",
    "get_transcript_segments",
    "clean_segments",
    "merge_segments",
    "refine_sentences",
    "embed_sentences",
    "search_sentences",
    "full_pipeline",
    "segment_fit_sentence",
]
