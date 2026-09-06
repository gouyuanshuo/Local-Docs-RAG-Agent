"""Public request and response contracts for the HTTP API.

These models are the only place the backend promises a wire shape. The option
types are imported from `constants` rather than re-declared, so a new runtime or
chunk strategy is accepted by request validation the moment it is supported by
the configuration layer, and can never drift out of sync with it.
"""

from __future__ import annotations

from typing import Annotated, Literal

import pydantic

from local_docs_rag_agent import constants, models

NonEmptyString = Annotated[
    str, pydantic.StringConstraints(strip_whitespace=True, min_length=1)
]
RuntimeList = Annotated[
    list[constants.RuntimeName],
    pydantic.Field(min_length=1, max_length=len(constants.AGENT_RUNTIMES)),
]
ChunkStrategyList = Annotated[
    list[constants.ChunkStrategyName],
    pydantic.Field(min_length=1, max_length=len(constants.CHUNK_STRATEGIES)),
]
VectorBackendList = Annotated[
    list[constants.VectorBackendName],
    pydantic.Field(min_length=1, max_length=len(constants.VECTOR_BACKENDS)),
]
RetrievalStrategyList = Annotated[
    list[constants.RetrievalStrategyName],
    pydantic.Field(
        min_length=1, max_length=len(constants.RETRIEVAL_STRATEGIES)
    ),
]
RerankerList = Annotated[
    list[constants.RerankerName],
    pydantic.Field(min_length=1, max_length=len(constants.RERANKERS)),
]
PositiveIntList = Annotated[
    list[pydantic.PositiveInt], pydantic.Field(min_length=1, max_length=8)
]
NonNegativeIntList = Annotated[
    list[pydantic.NonNegativeInt], pydantic.Field(min_length=1, max_length=8)
]


class AskRequest(pydantic.BaseModel):
    """One question to answer, optionally overriding the runtime."""

    question: NonEmptyString
    runtime: constants.RuntimeName | None = None


class EvalRequest(pydantic.BaseModel):
    """A request to run the eval harness, optionally overriding the runtime."""

    runtime: constants.RuntimeName | None = None


class EvalCompareRequest(pydantic.BaseModel):
    """The axes to sweep in a comparison run.

    An omitted axis holds the configured value steady rather than
    sweeping it, so a request cannot accidentally expand into the full
    Cartesian product.
    """

    runtimes: RuntimeList | None = None
    chunk_strategies: ChunkStrategyList | None = None
    vector_backends: VectorBackendList | None = None
    top_ks: PositiveIntList | None = None
    chunk_sizes: PositiveIntList | None = None
    chunk_overlaps: NonNegativeIntList | None = None
    retrieval_strategies: RetrievalStrategyList | None = None
    rerankers: RerankerList | None = None


class IngestResponse(pydantic.BaseModel):
    """How many chunks an ingest produced, and where they were stored."""

    num_chunks: pydantic.NonNegativeInt
    vector_backend: constants.VectorBackendName


class HealthResponse(pydantic.BaseModel):
    """A liveness answer with the backend's clock, for drift diagnosis."""

    status: Literal["ok"]
    backend_time_utc: str


class AppInfoResponse(pydantic.BaseModel):
    """The configuration the server is actually running under.

    This is what the frontend's config panel reads, so it reports the
    resolved values rather than the requested ones.
    """

    name: str
    runtime: constants.RuntimeName
    vector_backend: constants.VectorBackendName
    docs_dir: str
    docs_exclude_patterns: list[str]
    docs_count: pydantic.NonNegativeInt
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    top_k: pydantic.PositiveInt
    retrieval_strategy: constants.RetrievalStrategyName
    reranker: constants.RerankerName
    chunk_strategy: constants.ChunkStrategyName
    chunk_size: pydantic.PositiveInt
    chunk_overlap: pydantic.NonNegativeInt
    qdrant_collection: str
    external_http_trust_env: bool


class DocumentsResponse(pydantic.BaseModel):
    """Every document the agent can currently read."""

    count: pydantic.NonNegativeInt
    documents: list[str]


class ProviderStatusResponse(pydantic.BaseModel):
    """One provider's health, including why it degraded.

    `mode` stays `fallback` rather than being smoothed into success, so a
    client can reject a degraded answer instead of trusting it.
    """

    provider: str
    mode: models.ProviderMode
    reason: str | None = None


class CitationSpanResponse(pydantic.BaseModel):
    """The exact source offsets a citation points at."""

    source_path: str
    chunk_id: str
    chunk_index: pydantic.NonNegativeInt
    start_char: pydantic.NonNegativeInt
    end_char: pydantic.NonNegativeInt
    text: str


class AnswerDiagnosticsResponse(pydantic.BaseModel):
    """How an answer was actually produced, stage by stage.

    `requested_runtime` and `actual_runtime` differ whenever a runtime
    degraded, and each of the three provider statuses can degrade
    independently of the others.
    """

    requested_runtime: constants.RuntimeName
    actual_runtime: constants.RuntimeName
    vector_backend: constants.VectorBackendName
    chat_provider: ProviderStatusResponse
    embedding_provider: ProviderStatusResponse
    reranker: ProviderStatusResponse


class AskResponse(pydantic.BaseModel):
    """An answer with its citations and the diagnostics behind it."""

    question: str
    answer: str
    citations: list[str]
    citation_spans: list[CitationSpanResponse]
    runtime: constants.RuntimeName
    diagnostics: AnswerDiagnosticsResponse


class RetrievalConfigResponse(pydantic.BaseModel):
    """The retrieval settings a result was produced under.

    Attaching this to eval output is what makes two runs comparable after
    the fact, instead of leaving the reader to guess which settings
    produced which numbers.
    """

    vector_backend: constants.VectorBackendName
    chunk_strategy: constants.ChunkStrategyName
    chunk_size: pydantic.PositiveInt
    chunk_overlap: pydantic.NonNegativeInt
    top_k: pydantic.PositiveInt
    retrieval_strategy: constants.RetrievalStrategyName
    retrieval_candidate_k: pydantic.PositiveInt
    rrf_k: pydantic.PositiveInt
    reranker: constants.RerankerName
    rerank_candidate_k: pydantic.PositiveInt
    docs_dir: str
    docs_exclude_patterns: list[str]


class EvalResultResponse(pydantic.BaseModel):
    """One eval case: its metrics, its expectations, and why it failed."""

    question: str
    answer: str
    citations: list[str]
    retrieved_sources: list[str]
    answer_keyword_hit_rate: float = pydantic.Field(ge=0, le=1)
    retrieval_source_hit_rate: float = pydantic.Field(ge=0, le=1)
    retrieval_span_hit_rate: float = pydantic.Field(ge=0, le=1)
    citation_source_hit_rate: float = pydantic.Field(ge=0, le=1)
    citation_span_hit_rate: float = pydantic.Field(ge=0, le=1)
    response_time_ms: float = pydantic.Field(ge=0)
    diagnostics: AnswerDiagnosticsResponse | None = None
    expected_source_paths: list[str]
    expected_answer_keywords: list[str]
    expected_span_keywords: list[str]
    expected_retrieval_keywords: list[str]
    failure_reasons: list[str]


class EvalSummaryResponse(pydantic.BaseModel):
    """Per-case eval results aggregated into one reportable summary."""

    num_cases: pydantic.NonNegativeInt
    runtime: constants.RuntimeName
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


class EvalLeaderboardRowResponse(pydantic.BaseModel):
    """One comparison cell's headline metrics, for ranking cells."""

    label: str
    answer_keyword_hit_rate: float
    retrieval_source_hit_rate: float
    retrieval_span_hit_rate: float
    citation_span_hit_rate: float
    avg_response_time_ms: float


class EvalMatrixRunResponse(pydantic.BaseModel):
    """One cell of a comparison matrix.

    A cell that could not run is reported as `skipped` with a reason and a
    cell that failed as `error`; neither is dropped, because a leaderboard
    that hides its gaps is worse than no leaderboard.
    """

    label: str
    status: Literal["ok", "skipped", "error"]
    reason: str | None = None
    error: str | None = None
    retrieval_config: RetrievalConfigResponse | None = None
    runtime: constants.RuntimeName | None = None
    summary: EvalSummaryResponse | None = None


class EvalCompareResponse(pydantic.BaseModel):
    """Every comparison cell that ran, plus the leaderboard built from them."""

    num_runs: pydantic.NonNegativeInt
    runtimes: list[constants.RuntimeName]
    chunk_strategies: list[constants.ChunkStrategyName]
    vector_backends: list[constants.VectorBackendName]
    top_ks: list[pydantic.PositiveInt]
    chunk_sizes: list[pydantic.PositiveInt]
    chunk_overlaps: list[pydantic.NonNegativeInt]
    retrieval_strategies: list[constants.RetrievalStrategyName]
    rerankers: list[constants.RerankerName]
    leaderboard: list[EvalLeaderboardRowResponse]
    runs: list[EvalMatrixRunResponse]


class ErrorResponse(pydantic.BaseModel):
    """A failure with a stable code and, where one exists, a way to fix it."""

    code: str
    detail: str
    action_hint: str | None = None
