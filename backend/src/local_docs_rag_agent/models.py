from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from local_docs_rag_agent.exceptions import DataFormatError

ProviderMode = Literal["ready", "live", "fallback", "unknown"]


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    source_path: str
    title: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    embedding: list[float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
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
            metadata = {str(key): value for key, value in metadata_value.items()}
            embedding_value = payload.get("embedding")
            embedding = _optional_float_list(embedding_value)
        except (KeyError, TypeError, ValueError) as exc:
            raise DataFormatError(f"Invalid document chunk payload: {exc}") from exc
        if chunk_index < 0 or start_char < 0 or end_char < start_char:
            raise DataFormatError(
                "Invalid document chunk payload: indexes and character span are inconsistent"
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


@dataclass(slots=True)
class CitationSpan:
    source_path: str
    chunk_id: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str


@dataclass(slots=True)
class RetrievalHit:
    chunk: DocumentChunk
    score: float
    citation_span: CitationSpan


@dataclass(slots=True)
class ProviderStatus:
    provider: str
    mode: ProviderMode
    reason: str | None = None


@dataclass(slots=True)
class AnswerDiagnostics:
    requested_runtime: str
    actual_runtime: str
    vector_backend: str
    chat_provider: ProviderStatus
    embedding_provider: ProviderStatus


@dataclass(slots=True)
class AgentAnswer:
    question: str
    answer: str
    citations: list[str]
    citation_spans: list[CitationSpan]
    retrieved_chunks: list[RetrievalHit]
    diagnostics: AnswerDiagnostics


@dataclass(slots=True)
class EvalCase:
    question: str
    expected_answer_keywords: list[str]
    expected_source_paths: list[str]
    expected_span_keywords: list[str] = field(default_factory=list)
    expected_retrieval_keywords: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass(slots=True)
class EvalResult:
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
    failure_reasons: list[str] = field(default_factory=list)
    expected_source_paths: list[str] = field(default_factory=list)
    expected_answer_keywords: list[str] = field(default_factory=list)
    expected_span_keywords: list[str] = field(default_factory=list)
    expected_retrieval_keywords: list[str] = field(default_factory=list)


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
    if not all(isinstance(item, int | float) and not isinstance(item, bool) for item in value):
        raise TypeError("embedding values must be numbers")
    return [float(item) for item in value]
