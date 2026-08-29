from __future__ import annotations

import math
import re

from local_docs_rag_agent.models import CitationSpan, DocumentChunk, RetrievalHit

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def build_retrieval_hit(chunk: DocumentChunk, score: float) -> RetrievalHit:
    return RetrievalHit(
        chunk=chunk,
        score=score,
        citation_span=CitationSpan(
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
    chunk: DocumentChunk,
) -> float:
    dense_score = cosine_similarity(query_embedding, chunk.embedding or [])
    lexical_score = lexical_overlap_score(query_terms, tokenize(chunk.text))
    metadata_score = metadata_overlap_score(query_terms, chunk)
    weighted_mix = (dense_score * 0.65) + (lexical_score * 0.25) + (metadata_score * 0.10)
    return max(dense_score, lexical_score, weighted_mix)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def lexical_overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def metadata_overlap_score(query_terms: set[str], chunk: DocumentChunk) -> float:
    metadata_terms = tokenize(chunk.title)
    for key in ("section_title", "source_title"):
        value = chunk.metadata.get(key)
        if isinstance(value, str):
            metadata_terms |= tokenize(value)
    return lexical_overlap_score(query_terms, metadata_terms)
