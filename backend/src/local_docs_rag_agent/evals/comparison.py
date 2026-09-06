"""Runs the eval harness across a matrix of retrieval configurations.

Each cell re-ingests and re-evaluates under one `AppConfig` variant so the
results are comparable. A cell that cannot run is reported as `skipped` with a
reason, and a cell that fails is reported as `error`; neither is silently
dropped, because a leaderboard that hides its gaps is worse than no leaderboard.
"""

from __future__ import annotations

import itertools
import json
import math
import pathlib
from collections.abc import Sequence

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import constants, exceptions, presenters, rag
from local_docs_rag_agent.evals import harness
from local_docs_rag_agent.rag import file_io

# Guards API and CLI input from expanding into an unbounded Cartesian workload.
MAX_MATRIX_RUNS = 128


def default_chunk_strategies() -> list[constants.ChunkStrategyName]:
    """Return the chunk strategies compared when none are chosen."""
    return list(constants.CHUNK_STRATEGIES)


def default_retrieval_strategies() -> list[constants.RetrievalStrategyName]:
    """Return the retrieval strategies compared when none are chosen."""
    return list(constants.RETRIEVAL_STRATEGIES)


def default_rerankers(
    config: app_config.AppConfig,
) -> list[constants.RerankerName]:
    """Return the rerankers compared when a caller does not choose any.

    Unlike every other axis this one does not sweep, it holds the configured
    value. Each `llm` cell spends one model call per eval case, so sweeping by
    default would turn a routine comparison into an unrequested bill; opting in
    with `--reranker llm` keeps that a decision.
    """
    return [config.reranker]


def default_vector_backends(
    config: app_config.AppConfig,
) -> list[constants.VectorBackendName]:
    """Return the vector backends worth comparing for `config`.

    Qdrant is only included when a URL is configured; otherwise every Qdrant
    cell would report the same `missing_qdrant_url` skip.
    """
    return ["local", "qdrant"] if config.qdrant_url else ["local"]


def run_eval_matrix(
    config: app_config.AppConfig,
    runtimes: Sequence[str],
    chunk_strategies: Sequence[str],
    vector_backends: Sequence[str],
    top_ks: Sequence[int],
    chunk_sizes: Sequence[int],
    chunk_overlaps: Sequence[int],
    retrieval_strategies: Sequence[str] | None = None,
    rerankers: Sequence[str] | None = None,
    output_path: pathlib.Path | None = None,
) -> dict[str, object]:
    """Evaluate every combination of the requested axes and rank the results.

    Args:
      config: The baseline settings each cell varies from.
      runtimes: Runtimes to sweep.
      chunk_strategies: Chunk strategies to sweep.
      vector_backends: Vector backends to sweep.
      top_ks: Result-window sizes to sweep.
      chunk_sizes: Chunk sizes to sweep.
      chunk_overlaps: Chunk overlaps to sweep.
      retrieval_strategies: Ranking strategies to sweep, or None to
        hold the configured one steady.
      rerankers: Rerankers to sweep, or None to hold the configured
        one steady.
      output_path: Where to write the report, or None to skip
        writing it.

    Returns:
      The report: every cell that ran, and the leaderboard built from
      the ones that succeeded.

    Raises:
      ConfigurationError: If an axis is empty, or the Cartesian
        product would exceed `MAX_MATRIX_RUNS`, so one request cannot
        start an unbounded run.
    """
    # An omitted axis holds the configured value steady rather than sweeping, so
    # an existing caller keeps producing the run count it produced before.
    strategies: Sequence[str] = retrieval_strategies or [
        config.retrieval_strategy
    ]
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
    num_combinations = math.prod(axis_lengths)
    if 0 in axis_lengths:
        raise exceptions.ConfigurationError(
            "Eval comparison axes must not be empty"
        )
    if num_combinations > MAX_MATRIX_RUNS:
        raise exceptions.ConfigurationError(
            f"Eval comparison requested {num_combinations} runs; "
            f"maximum is {MAX_MATRIX_RUNS}",
            action_hint="Reduce one or more comparison axes.",
        )
    runs: list[dict[str, object]] = []

    combinations = itertools.product(
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
        file_io.atomic_write_text(output_path, _dump_json(payload))
    return payload


def _dump_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _run_matrix_case(config: app_config.AppConfig) -> dict[str, object]:
    label = _run_label(config)
    common = {
        "label": label,
        "retrieval_config": presenters.serialize_retrieval_config(config),
        "runtime": config.agent_runtime,
    }
    skip_reason = _skip_reason(config)
    if skip_reason is not None:
        return {**common, "status": "skipped", "reason": skip_reason}

    try:
        rag.ingest_documents(config)
        results = harness.run_eval(config)
        summary = presenters.serialize_eval_summary(
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


def _run_label(config: app_config.AppConfig) -> str:
    return (
        f"{config.agent_runtime}:{config.vector_backend}"
        f":{config.chunk_strategy}"
        f":{config.retrieval_strategy}:rr-{config.reranker}"
        f":k{config.top_k}:s{config.chunk_size}:o{config.chunk_overlap}"
    )


def _skip_reason(config: app_config.AppConfig) -> str | None:
    if config.vector_backend == "qdrant" and not config.qdrant_url:
        return "missing_qdrant_url"
    return None


def _qdrant_runtime_skip_reason(
    config: app_config.AppConfig, exc: Exception
) -> str | None:
    if config.vector_backend != "qdrant":
        return None

    if isinstance(exc, exceptions.VectorStoreError):
        if exc.reason_code == "unreachable":
            return "qdrant_unreachable"
        if exc.reason_code == "dependency_missing":
            return "missing_qdrant_client"
    return None


def _build_leaderboard(
    runs: list[dict[str, object]],
) -> list[dict[str, object]]:
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
                "answer_keyword_hit_rate": summary.get(
                    "answer_keyword_hit_rate", 0.0
                ),
                "retrieval_source_hit_rate": summary.get(
                    "retrieval_source_hit_rate", 0.0
                ),
                "retrieval_span_hit_rate": summary.get(
                    "retrieval_span_hit_rate", 0.0
                ),
                "citation_span_hit_rate": summary.get(
                    "citation_span_hit_rate", 0.0
                ),
                "avg_response_time_ms": summary.get(
                    "avg_response_time_ms", 0.0
                ),
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
