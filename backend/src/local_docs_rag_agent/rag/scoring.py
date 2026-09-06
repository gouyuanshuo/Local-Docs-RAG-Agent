"""Scores a chunk against a query for the local retrieval backend.

Three signals are blended: cosine similarity over embeddings, lexical overlap
with the chunk text, and overlap with the chunk's title and section metadata.
The final score is the maximum of the dense score, the lexical score, and their
weighted mix, so a query that matches strongly on one signal is not dragged down
by a weak showing on another — which matters most when embeddings have degraded
to the hash fallback.
"""

from __future__ import annotations

import math
import re

from local_docs_rag_agent import models

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def build_retrieval_hit(
    chunk: models.DocumentChunk, score: float
) -> models.RetrievalHit:
    return models.RetrievalHit(
        chunk=chunk,
        score=score,
        citation_span=models.CitationSpan(
            source_path=chunk.source_path,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            text=chunk.text,
        ),
    )


def score_local_chunk(
    *,
    query_embedding: list[float],
    query_terms: set[str],
    chunk: models.DocumentChunk,
) -> float:
    dense_score = cosine_similarity(query_embedding, chunk.embedding or [])
    lexical_score = lexical_overlap_score(query_terms, tokenize(chunk.text))
    metadata_score = metadata_overlap_score(query_terms, chunk)
    weighted_mix = (
        (dense_score * 0.65) + (lexical_score * 0.25) + (metadata_score * 0.10)
    )
    return max(dense_score, lexical_score, weighted_mix)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def tokenize_terms(text: str) -> list[str]:
    """Return every term in order, keeping duplicates.

    BM25 weights a term by how often it occurs, so it needs the counts that
    :func:`tokenize` discards. Both shapes come from one regex so the lexical
    signals can never disagree about what a term is.
    """

    return [token.lower() for token in TOKEN_RE.findall(text)]


def tokenize(text: str) -> set[str]:
    return set(tokenize_terms(text))


def lexical_overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def metadata_overlap_score(
    query_terms: set[str], chunk: models.DocumentChunk
) -> float:
    metadata_terms = tokenize(chunk.title)
    for key in ("section_title", "source_title"):
        value = chunk.metadata.get(key)
        if isinstance(value, str):
            metadata_terms |= tokenize(value)
    return lexical_overlap_score(query_terms, metadata_terms)
