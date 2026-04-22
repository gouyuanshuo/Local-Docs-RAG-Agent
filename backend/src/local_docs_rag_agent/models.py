from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentChunk":
        return cls(**payload)


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
class AgentAnswer:
    question: str
    answer: str
    citations: list[str]
    citation_spans: list[CitationSpan]
    retrieved_chunks: list[RetrievalHit]


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
