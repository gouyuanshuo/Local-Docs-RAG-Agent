"""Framework-free domain records shared by every layer of the backend.

These dataclasses are the vocabulary the whole backend speaks: a chunk and where
in its source file it came from, a scored retrieval hit, an answer with its
citations, and the honest provider/runtime diagnostics attached to both. Nothing
here knows about FastAPI, Pydantic, or argparse, so the same records serialize
to an HTTP response, a CLI payload, and an eval report without change.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from local_docs_rag_agent import constants, exceptions

ProviderMode = Literal["ready", "live", "fallback", "unknown"]


@dataclasses.dataclass(slots=True)
class DocumentChunk:
    """One chunk of a source document, with the span it came from."""

    chunk_id: str
    source_path: str
    title: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    embedding: list[float] | None = None
    metadata: dict[str, object] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return the chunk as a JSON-serializable mapping."""
        return {
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "title": self.title,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DocumentChunk:
        """Rebuild a chunk from a stored mapping.

        Args:
          payload: A mapping previously produced by `to_dict`.

        Returns:
          The reconstructed chunk.

        Raises:
          DataFormatError: If a field is missing, has the wrong type, or
            the character span is inconsistent with the chunk index.
        """
        try:
            chunk_id = _required_string(payload, "chunk_id")
            source_path = _required_string(payload, "source_path")
            title = _required_string(payload, "title")
            text = _required_string(payload, "text")
            chunk_index = _required_integer(payload, "chunk_index")
            start_char = _required_integer(payload, "start_char")
            end_char = _required_integer(payload, "end_char")
            metadata_value = payload.get("metadata", {})
            if not isinstance(metadata_value, dict):
                raise TypeError("metadata must be an object")
            metadata = {
                str(key): value for key, value in metadata_value.items()
            }
            embedding_value = payload.get("embedding")
            embedding = _optional_float_list(embedding_value)
        except (KeyError, TypeError, ValueError) as exc:
            raise exceptions.DataFormatError(
                f"Invalid document chunk payload: {exc}"
            ) from exc
        if chunk_index < 0 or start_char < 0 or end_char < start_char:
            raise exceptions.DataFormatError(
                "Invalid document chunk payload: indexes and character "
                "span are inconsistent"
            )
        return cls(
            chunk_id=chunk_id,
            source_path=source_path,
            title=title,
            text=text,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            embedding=embedding,
            metadata=metadata,
        )


@dataclasses.dataclass(slots=True)
class CitationSpan:
    """The exact source offsets backing one citation."""

    source_path: str
    chunk_id: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str


@dataclasses.dataclass(slots=True)
class RetrievalHit:
    """One retrieved chunk, its score, and the span that cites it.

    Scores are comparable only within the strategy that produced them: an
    RRF score is a sum of reciprocal ranks, a dense score is a cosine
    similarity, and a reranked score is `1 / position`.
    """

    chunk: DocumentChunk
    score: float
    citation_span: CitationSpan


@dataclasses.dataclass(slots=True)
class ProviderStatus:
    """Whether a provider ran, was skipped, or degraded, and why."""

    provider: str
    mode: ProviderMode
    reason: str | None = None


@dataclasses.dataclass(slots=True)
class RetrievalOutcome:
    """What one retrieval produced, and the health of each stage.

    Retrieval runs in two stages that can degrade independently — embeddings and
    the reranker — so the hits alone are not a complete answer to "what happened
    here". Returning them together is what lets a runtime attach both statuses
    to its diagnostics without knowing how retrieval is assembled.
    """

    hits: list[RetrievalHit]
    embedding_status: ProviderStatus
    reranker_status: ProviderStatus


@dataclasses.dataclass(slots=True)
class AnswerDiagnostics:
    """Truthful record of how an answer was actually produced.

    `requested_runtime` and `actual_runtime` differ whenever a runtime degraded,
    and the provider statuses stay `fallback` rather than being smoothed into
    success, so a caller can reject a degraded run instead of trusting it.
    """

    requested_runtime: constants.RuntimeName
    actual_runtime: constants.RuntimeName
    vector_backend: constants.VectorBackendName
    chat_provider: ProviderStatus
    embedding_provider: ProviderStatus
    reranker: ProviderStatus


@dataclasses.dataclass(slots=True)
class AgentAnswer:
    """A complete answer: the text, its citations, and its diagnostics."""

    question: str
    answer: str
    citations: list[str]
    citation_spans: list[CitationSpan]
    retrieved_chunks: list[RetrievalHit]
    diagnostics: AnswerDiagnostics


@dataclasses.dataclass(slots=True)
class EvalCase:
    """One eval question and everything a correct answer must contain."""

    question: str
    expected_answer_keywords: list[str]
    expected_source_paths: list[str]
    expected_span_keywords: list[str] = dataclasses.field(default_factory=list)
    expected_retrieval_keywords: list[str] = dataclasses.field(
        default_factory=list
    )
    notes: str | None = None


@dataclasses.dataclass(slots=True)
class EvalResult:
    """One eval case's measured outcome and the expectations behind it."""

    question: str
    answer: str
    citations: list[str]
    retrieved_sources: list[str]
    answer_keyword_hit_rate: float
    retrieval_source_hit_rate: float
    retrieval_span_hit_rate: float
    citation_source_hit_rate: float
    citation_span_hit_rate: float
    response_time_ms: float
    diagnostics: AnswerDiagnostics | None = None
    failure_reasons: list[str] = dataclasses.field(default_factory=list)
    expected_source_paths: list[str] = dataclasses.field(default_factory=list)
    expected_answer_keywords: list[str] = dataclasses.field(
        default_factory=list
    )
    expected_span_keywords: list[str] = dataclasses.field(default_factory=list)
    expected_retrieval_keywords: list[str] = dataclasses.field(
        default_factory=list
    )


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _required_integer(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value


def _optional_float_list(value: object) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("embedding must be a list or null")
    if not all(
        isinstance(item, int | float) and not isinstance(item, bool)
        for item in value
    ):
        raise TypeError("embedding values must be numbers")
    return [float(item) for item in value]
