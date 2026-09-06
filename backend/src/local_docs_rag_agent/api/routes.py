"""Maps HTTP requests to application operations.

Handlers stay thin on purpose: they read a fresh `AppConfig`, call one
application function, and hand the result to a presenter. Retrieval and provider
policy live behind those calls, so the CLI can reach the same behaviour without
going through HTTP.
"""

from __future__ import annotations

import datetime

import fastapi

from local_docs_rag_agent import agent, presenters, rag, tools
from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.api import schemas
from local_docs_rag_agent.evals import comparison, harness

router = fastapi.APIRouter(prefix="/api")
API_NAME = "Local Docs RAG Agent"


@router.get("/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    """Report liveness and the backend clock.

    Returns:
      The health payload.
    """
    return schemas.HealthResponse(
        status="ok",
        backend_time_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    )


@router.get("/info", response_model=schemas.AppInfoResponse)
def info() -> schemas.AppInfoResponse:
    """Report the configuration the server is running under.

    Returns:
      The resolved settings, plus how many documents are visible.
    """
    config = app_config.AppConfig.from_env()
    documents = tools.list_documents(config)
    return schemas.AppInfoResponse.model_validate(
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


@router.get("/documents", response_model=schemas.DocumentsResponse)
def documents() -> schemas.DocumentsResponse:
    """List the documents the agent can currently read.

    Returns:
      Every discovered document path, with a count.
    """
    docs = tools.list_documents(app_config.AppConfig.from_env())
    return schemas.DocumentsResponse(count=len(docs), documents=docs)


@router.post("/ingest", response_model=schemas.IngestResponse)
def ingest() -> schemas.IngestResponse:
    """Rebuild the retrieval index from the documents on disk.

    Returns:
      How many chunks were written, and to which backend.
    """
    config = app_config.AppConfig.from_env()
    chunks = rag.ingest_documents(config)
    return schemas.IngestResponse.model_validate(
        {"num_chunks": len(chunks), "vector_backend": config.vector_backend}
    )


@router.post("/ask", response_model=schemas.AskResponse)
def ask(payload: schemas.AskRequest) -> schemas.AskResponse:
    """Answer one question against the indexed documents.

    Args:
      payload: The question, and an optional runtime override.

    Returns:
      The answer with its citations and diagnostics. The index is
      built first if it is missing or was built under other settings.
    """
    config = app_config.AppConfig.from_env().with_runtime(payload.runtime)
    rag.ensure_index(config)
    answer = agent.LocalDocsAgent(config).answer(payload.question)
    return schemas.AskResponse.model_validate(
        presenters.serialize_answer(answer)
    )


@router.post("/eval", response_model=schemas.EvalSummaryResponse)
def evaluate(payload: schemas.EvalRequest) -> schemas.EvalSummaryResponse:
    """Run the eval harness over the configured eval file.

    Args:
      payload: An optional runtime override.

    Returns:
      Per-case results and their aggregate summary.
    """
    config = app_config.AppConfig.from_env().with_runtime(payload.runtime)
    rag.ensure_index(config)
    results = harness.run_eval(config)
    return schemas.EvalSummaryResponse.model_validate(
        presenters.serialize_eval_summary(
            results, runtime=config.agent_runtime, config=config
        )
    )


@router.post("/eval/compare", response_model=schemas.EvalCompareResponse)
def compare_eval(
    payload: schemas.EvalCompareRequest,
) -> schemas.EvalCompareResponse:
    """Evaluate a matrix of configurations and rank the results.

    Args:
      payload: The axes to sweep. An omitted axis holds the configured
        value steady rather than sweeping it.

    Returns:
      Every cell that ran, plus the leaderboard built from them.
    """
    config = app_config.AppConfig.from_env()
    report = comparison.run_eval_matrix(
        config=config,
        runtimes=list(payload.runtimes or [config.agent_runtime]),
        chunk_strategies=list(
            payload.chunk_strategies or comparison.default_chunk_strategies()
        ),
        vector_backends=list(
            payload.vector_backends
            or comparison.default_vector_backends(config)
        ),
        top_ks=list(payload.top_ks or [config.top_k]),
        chunk_sizes=list(payload.chunk_sizes or [config.chunk_size]),
        chunk_overlaps=list(payload.chunk_overlaps or [config.chunk_overlap]),
        retrieval_strategies=list(
            payload.retrieval_strategies
            or comparison.default_retrieval_strategies()
        ),
        rerankers=list(
            payload.rerankers or comparison.default_rerankers(config)
        ),
    )
    return schemas.EvalCompareResponse.model_validate(report)
