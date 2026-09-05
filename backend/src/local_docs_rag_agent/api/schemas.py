"""Public request and response contracts for the HTTP API.

These models are the only place the backend promises a wire shape. The option types
are imported from `constants` rather than re-declared, so a new runtime or chunk
strategy is accepted by request validation the moment it is supported by the
configuration layer, and can never drift out of sync with it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, StringConstraints

from local_docs_rag_agent.constants import (
    AGENT_RUNTIMES,
    CHUNK_STRATEGIES,
    RETRIEVAL_STRATEGIES,
    VECTOR_BACKENDS,
    ChunkStrategyName,
    RetrievalStrategyName,
    RuntimeName,
    VectorBackendName,
)
from local_docs_rag_agent.models import ProviderMode

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
RuntimeList = Annotated[list[RuntimeName], Field(min_length=1, max_length=len(AGENT_RUNTIMES))]
ChunkStrategyList = Annotated[
    list[ChunkStrategyName],
    Field(min_length=1, max_length=len(CHUNK_STRATEGIES)),
]
VectorBackendList = Annotated[
    list[VectorBackendName],
    Field(min_length=1, max_length=len(VECTOR_BACKENDS)),
]
RetrievalStrategyList = Annotated[
    list[RetrievalStrategyName],
    Field(min_length=1, max_length=len(RETRIEVAL_STRATEGIES)),
]
PositiveIntList = Annotated[list[PositiveInt], Field(min_length=1, max_length=8)]
NonNegativeIntList = Annotated[list[NonNegativeInt], Field(min_length=1, max_length=8)]


class AskRequest(BaseModel):
    question: NonEmptyString
    runtime: RuntimeName | None = None


class EvalRequest(BaseModel):
    runtime: RuntimeName | None = None


class EvalCompareRequest(BaseModel):
    runtimes: RuntimeList | None = None
    chunk_strategies: ChunkStrategyList | None = None
    vector_backends: VectorBackendList | None = None
    top_ks: PositiveIntList | None = None
    chunk_sizes: PositiveIntList | None = None
    chunk_overlaps: NonNegativeIntList | None = None
    retrieval_strategies: RetrievalStrategyList | None = None


class IngestResponse(BaseModel):
    num_chunks: NonNegativeInt
    vector_backend: VectorBackendName


class HealthResponse(BaseModel):
    status: Literal["ok"]
    backend_time_utc: str


class AppInfoResponse(BaseModel):
    name: str
    runtime: RuntimeName
    vector_backend: VectorBackendName
    docs_dir: str
    docs_exclude_patterns: list[str]
    docs_count: NonNegativeInt
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    top_k: PositiveInt
    retrieval_strategy: RetrievalStrategyName
    chunk_strategy: ChunkStrategyName
    chunk_size: PositiveInt
    chunk_overlap: NonNegativeInt
    qdrant_collection: str
    external_http_trust_env: bool


class DocumentsResponse(BaseModel):
    count: NonNegativeInt
    documents: list[str]


class ProviderStatusResponse(BaseModel):
    provider: str
    mode: ProviderMode
    reason: str | None = None


class CitationSpanResponse(BaseModel):
    source_path: str
    chunk_id: str
    chunk_index: NonNegativeInt
    start_char: NonNegativeInt
    end_char: NonNegativeInt
    text: str


class AnswerDiagnosticsResponse(BaseModel):
    requested_runtime: RuntimeName
    actual_runtime: RuntimeName
    vector_backend: VectorBackendName
    chat_provider: ProviderStatusResponse
    embedding_provider: ProviderStatusResponse


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    citation_spans: list[CitationSpanResponse]
    runtime: RuntimeName
    diagnostics: AnswerDiagnosticsResponse


class RetrievalConfigResponse(BaseModel):
    vector_backend: VectorBackendName
    chunk_strategy: ChunkStrategyName
    chunk_size: PositiveInt
    chunk_overlap: NonNegativeInt
    top_k: PositiveInt
    retrieval_strategy: RetrievalStrategyName
    retrieval_candidate_k: PositiveInt
    rrf_k: PositiveInt
    docs_dir: str
    docs_exclude_patterns: list[str]


class EvalResultResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    retrieved_sources: list[str]
    answer_keyword_hit_rate: float = Field(ge=0, le=1)
    retrieval_source_hit_rate: float = Field(ge=0, le=1)
    retrieval_span_hit_rate: float = Field(ge=0, le=1)
    citation_source_hit_rate: float = Field(ge=0, le=1)
    citation_span_hit_rate: float = Field(ge=0, le=1)
    response_time_ms: float = Field(ge=0)
    diagnostics: AnswerDiagnosticsResponse | None = None
    expected_source_paths: list[str]
    expected_answer_keywords: list[str]
    expected_span_keywords: list[str]
    expected_retrieval_keywords: list[str]
    failure_reasons: list[str]


class EvalSummaryResponse(BaseModel):
    num_cases: NonNegativeInt
    runtime: RuntimeName
    retrieval_config: RetrievalConfigResponse | None
    answer_keyword_hit_rate: float
    retrieval_source_hit_rate: float
    retrieval_span_hit_rate: float
    citation_source_hit_rate: float
    citation_span_hit_rate: float
    avg_response_time_ms: float
    avg_keyword_hit_rate: float
    source_hit_rate: float
    avg_citation_span_hit_rate: float
    results: list[EvalResultResponse]


class EvalLeaderboardRowResponse(BaseModel):
    label: str
    answer_keyword_hit_rate: float
    retrieval_source_hit_rate: float
    retrieval_span_hit_rate: float
    citation_span_hit_rate: float
    avg_response_time_ms: float


class EvalMatrixRunResponse(BaseModel):
    label: str
    status: Literal["ok", "skipped", "error"]
    reason: str | None = None
    error: str | None = None
    retrieval_config: RetrievalConfigResponse | None = None
    runtime: RuntimeName | None = None
    summary: EvalSummaryResponse | None = None


class EvalCompareResponse(BaseModel):
    num_runs: NonNegativeInt
    runtimes: list[RuntimeName]
    chunk_strategies: list[ChunkStrategyName]
    vector_backends: list[VectorBackendName]
    top_ks: list[PositiveInt]
    chunk_sizes: list[PositiveInt]
    chunk_overlaps: list[NonNegativeInt]
    retrieval_strategies: list[RetrievalStrategyName]
    leaderboard: list[EvalLeaderboardRowResponse]
    runs: list[EvalMatrixRunResponse]


class ErrorResponse(BaseModel):
    code: str
    detail: str
    action_hint: str | None = None
