"""Runs the eval harness across a matrix of retrieval and runtime configurations.

Each cell re-ingests and re-evaluates under one `AppConfig` variant so the results are
comparable. A cell that cannot run is reported as `skipped` with a reason, and a cell
that fails is reported as `error`; neither is silently dropped, because a leaderboard
that hides its gaps is worse than no leaderboard.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import product
from math import prod
from pathlib import Path

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.constants import (
    CHUNK_STRATEGIES,
    RETRIEVAL_STRATEGIES,
    ChunkStrategyName,
    RerankerName,
    RetrievalStrategyName,
    VectorBackendName,
)
from local_docs_rag_agent.evals.harness import run_eval
from local_docs_rag_agent.exceptions import ConfigurationError, VectorStoreError
from local_docs_rag_agent.presenters import serialize_eval_summary, serialize_retrieval_config
from local_docs_rag_agent.rag import ingest_documents
from local_docs_rag_agent.rag.file_io import atomic_write_text

# Guards API and CLI input from expanding into an unbounded Cartesian workload.
MAX_MATRIX_RUNS = 128


def default_chunk_strategies() -> list[ChunkStrategyName]:
    """Return the chunk strategies compared when a caller does not choose any."""

    return list(CHUNK_STRATEGIES)


def default_retrieval_strategies() -> list[RetrievalStrategyName]:
    """Return the retrieval strategies compared when a caller does not choose any."""

    return list(RETRIEVAL_STRATEGIES)


def default_rerankers(config: AppConfig) -> list[RerankerName]:
    """Return the rerankers compared when a caller does not choose any.

    Unlike every other axis this one does not sweep, it holds the configured value.
    Each `llm` cell spends one model call per eval case, so sweeping by default would
    turn a routine comparison into an unrequested bill; opting in with `--reranker llm`
    keeps that a decision.
    """

    return [config.reranker]


def default_vector_backends(config: AppConfig) -> list[VectorBackendName]:
    """Return the vector backends worth comparing for `config`.

    Qdrant is only included when a URL is configured; otherwise every Qdrant cell
    would report the same `missing_qdrant_url` skip.
    """

    return ["local", "qdrant"] if config.qdrant_url else ["local"]


def run_eval_matrix(
    config: AppConfig,
    runtimes: Sequence[str],
    chunk_strategies: Sequence[str],
    vector_backends: Sequence[str],
    top_ks: Sequence[int],
    chunk_sizes: Sequence[int],
    chunk_overlaps: Sequence[int],
    retrieval_strategies: Sequence[str] | None = None,
    rerankers: Sequence[str] | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Evaluate every combination of the requested axes and rank the results.

    Raises `ConfigurationError` when an axis is empty or when the Cartesian product
    would exceed `MAX_MATRIX_RUNS`, so a single request cannot start an unbounded run.
    """

    # An omitted axis holds the configured value steady rather than sweeping, so an
    # existing caller keeps producing the run count it produced before.
    strategies: Sequence[str] = retrieval_strategies or [config.retrieval_strategy]
    reranker_names: Sequence[str] = rerankers or [config.reranker]
    axis_lengths = [
        len(runtimes),
        len(vector_backends),
        len(chunk_strategies),
        len(top_ks),
        len(chunk_sizes),
        len(chunk_overlaps),
        len(strategies),
        len(reranker_names),
    ]
    num_combinations = prod(axis_lengths)
    if 0 in axis_lengths:
        raise ConfigurationError("Eval comparison axes must not be empty")
    if num_combinations > MAX_MATRIX_RUNS:
        raise ConfigurationError(
            f"Eval comparison requested {num_combinations} runs; maximum is {MAX_MATRIX_RUNS}",
            action_hint="Reduce one or more comparison axes.",
        )
    runs: list[dict[str, object]] = []

    combinations = product(
        runtimes,
        vector_backends,
        chunk_strategies,
        top_ks,
        chunk_sizes,
        chunk_overlaps,
        strategies,
        reranker_names,
    )
    for (
        runtime,
        vector_backend,
        chunk_strategy,
        top_k,
        chunk_size,
        chunk_overlap,
        retrieval_strategy,
        reranker,
    ) in combinations:
        run_config = config.with_overrides(
            agent_runtime=runtime,
            vector_backend=vector_backend,
            chunk_strategy=chunk_strategy,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_strategy=retrieval_strategy,
            reranker=reranker,
        )
        runs.append(_run_matrix_case(run_config))

    payload = {
        "num_runs": len(runs),
        "runtimes": list(runtimes),
        "chunk_strategies": list(chunk_strategies),
        "vector_backends": list(vector_backends),
        "top_ks": list(top_ks),
        "chunk_sizes": list(chunk_sizes),
        "chunk_overlaps": list(chunk_overlaps),
        "retrieval_strategies": list(strategies),
        "rerankers": list(reranker_names),
        "leaderboard": _build_leaderboard(runs),
        "runs": runs,
    }
    if output_path is not None:
        atomic_write_text(output_path, _dump_json(payload))
    return payload


def _dump_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _run_matrix_case(config: AppConfig) -> dict[str, object]:
    label = _run_label(config)
    common = {
        "label": label,
        "retrieval_config": serialize_retrieval_config(config),
        "runtime": config.agent_runtime,
    }
    skip_reason = _skip_reason(config)
    if skip_reason is not None:
        return {**common, "status": "skipped", "reason": skip_reason}

    try:
        ingest_documents(config)
        results = run_eval(config)
        summary = serialize_eval_summary(
            results,
            runtime=config.agent_runtime,
            config=config,
        )
        return {"label": label, "status": "ok", "summary": summary}
    except Exception as exc:
        skip_reason = _qdrant_runtime_skip_reason(config, exc)
        if skip_reason is not None:
            return {**common, "status": "skipped", "reason": skip_reason}
        return {
            **common,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_label(config: AppConfig) -> str:
    return (
        f"{config.agent_runtime}:{config.vector_backend}:{config.chunk_strategy}"
        f":{config.retrieval_strategy}:rr-{config.reranker}"
        f":k{config.top_k}:s{config.chunk_size}:o{config.chunk_overlap}"
    )


def _skip_reason(config: AppConfig) -> str | None:
    if config.vector_backend == "qdrant" and not config.qdrant_url:
        return "missing_qdrant_url"
    return None


def _qdrant_runtime_skip_reason(config: AppConfig, exc: Exception) -> str | None:
    if config.vector_backend != "qdrant":
        return None

    if isinstance(exc, VectorStoreError):
        if exc.reason_code == "unreachable":
            return "qdrant_unreachable"
        if exc.reason_code == "dependency_missing":
            return "missing_qdrant_client"
    return None


def _build_leaderboard(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    leaderboard: list[dict[str, object]] = []
    for run in runs:
        if run.get("status") != "ok":
            continue
        summary = run.get("summary")
        if not isinstance(summary, dict):
            continue
        leaderboard.append(
            {
                "label": str(run["label"]),
                "answer_keyword_hit_rate": summary.get("answer_keyword_hit_rate", 0.0),
                "retrieval_source_hit_rate": summary.get("retrieval_source_hit_rate", 0.0),
                "retrieval_span_hit_rate": summary.get("retrieval_span_hit_rate", 0.0),
                "citation_span_hit_rate": summary.get("citation_span_hit_rate", 0.0),
                "avg_response_time_ms": summary.get("avg_response_time_ms", 0.0),
            }
        )
    leaderboard.sort(
        key=lambda row: (
            _as_float(row["retrieval_span_hit_rate"]),
            _as_float(row["answer_keyword_hit_rate"]),
            _as_float(row["citation_span_hit_rate"]),
            -_as_float(row["avg_response_time_ms"]),
        ),
        reverse=True,
    )
    return leaderboard


def _as_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0
