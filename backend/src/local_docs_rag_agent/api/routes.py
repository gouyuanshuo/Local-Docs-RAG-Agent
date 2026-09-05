"""Maps HTTP requests to application operations.

Handlers stay thin on purpose: they read a fresh `AppConfig`, call one application
function, and hand the result to a presenter. Retrieval and provider policy live
behind those calls, so the CLI can reach the same behaviour without going through HTTP.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from local_docs_rag_agent.agent import LocalDocsAgent
from local_docs_rag_agent.api.schemas import (
    AppInfoResponse,
    AskRequest,
    AskResponse,
    DocumentsResponse,
    EvalCompareRequest,
    EvalCompareResponse,
    EvalRequest,
    EvalSummaryResponse,
    HealthResponse,
    IngestResponse,
)
from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.evals.comparison import (
    default_chunk_strategies,
    default_rerankers,
    default_retrieval_strategies,
    default_vector_backends,
    run_eval_matrix,
)
from local_docs_rag_agent.evals.harness import run_eval
from local_docs_rag_agent.presenters import serialize_answer, serialize_eval_summary
from local_docs_rag_agent.rag import ensure_index, ingest_documents
from local_docs_rag_agent.tools import list_documents

router = APIRouter(prefix="/api")
API_NAME = "Local Docs RAG Agent"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        backend_time_utc=datetime.now(UTC).isoformat(),
    )


@router.get("/info", response_model=AppInfoResponse)
def info() -> AppInfoResponse:
    config = AppConfig.from_env()
    documents = list_documents(config)
    return AppInfoResponse.model_validate(
        {
            "name": API_NAME,
            "runtime": config.agent_runtime,
            "vector_backend": config.vector_backend,
            "docs_dir": str(config.docs_dir),
            "docs_exclude_patterns": config.docs_exclude_patterns,
            "docs_count": len(documents),
            "llm_provider": config.llm_provider,
            "llm_model": config.llm_model,
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "top_k": config.top_k,
            "retrieval_strategy": config.retrieval_strategy,
            "reranker": config.reranker,
            "chunk_strategy": config.chunk_strategy,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "qdrant_collection": config.qdrant_collection,
            "external_http_trust_env": config.external_http_trust_env,
        }
    )


@router.get("/documents", response_model=DocumentsResponse)
def documents() -> DocumentsResponse:
    docs = list_documents(AppConfig.from_env())
    return DocumentsResponse(count=len(docs), documents=docs)


@router.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    config = AppConfig.from_env()
    chunks = ingest_documents(config)
    return IngestResponse.model_validate(
        {"num_chunks": len(chunks), "vector_backend": config.vector_backend}
    )


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    config = AppConfig.from_env().with_runtime(payload.runtime)
    ensure_index(config)
    answer = LocalDocsAgent(config).answer(payload.question)
    return AskResponse.model_validate(serialize_answer(answer))


@router.post("/eval", response_model=EvalSummaryResponse)
def evaluate(payload: EvalRequest) -> EvalSummaryResponse:
    config = AppConfig.from_env().with_runtime(payload.runtime)
    ensure_index(config)
    results = run_eval(config)
    return EvalSummaryResponse.model_validate(
        serialize_eval_summary(results, runtime=config.agent_runtime, config=config)
    )


@router.post("/eval/compare", response_model=EvalCompareResponse)
def compare_eval(payload: EvalCompareRequest) -> EvalCompareResponse:
    config = AppConfig.from_env()
    comparison = run_eval_matrix(
        config=config,
        runtimes=list(payload.runtimes or [config.agent_runtime]),
        chunk_strategies=list(payload.chunk_strategies or default_chunk_strategies()),
        vector_backends=list(payload.vector_backends or default_vector_backends(config)),
        top_ks=list(payload.top_ks or [config.top_k]),
        chunk_sizes=list(payload.chunk_sizes or [config.chunk_size]),
        chunk_overlaps=list(payload.chunk_overlaps or [config.chunk_overlap]),
        retrieval_strategies=list(
            payload.retrieval_strategies or default_retrieval_strategies()
        ),
        rerankers=list(payload.rerankers or default_rerankers(config)),
    )
    return EvalCompareResponse.model_validate(comparison)
