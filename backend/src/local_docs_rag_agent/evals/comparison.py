from __future__ import annotations

import json
from itertools import product
from math import prod
from pathlib import Path

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.evals.harness import run_eval
from local_docs_rag_agent.exceptions import ConfigurationError, VectorStoreError
from local_docs_rag_agent.presenters import serialize_eval_summary
from local_docs_rag_agent.rag.file_io import atomic_write_text
from local_docs_rag_agent.rag.ingest import ingest_documents

MAX_MATRIX_RUNS = 128


def run_eval_matrix(
    config: AppConfig,
    runtimes: list[str],
    chunk_strategies: list[str],
    vector_backends: list[str],
    top_ks: list[int],
    chunk_sizes: list[int],
    chunk_overlaps: list[int],
    output_path: Path | None = None,
) -> dict[str, object]:
    axis_lengths = [
        len(runtimes),
        len(vector_backends),
        len(chunk_strategies),
        len(top_ks),
        len(chunk_sizes),
        len(chunk_overlaps),
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
    )
    for runtime, vector_backend, chunk_strategy, top_k, chunk_size, chunk_overlap in combinations:
        run_config = config.with_overrides(
            agent_runtime=runtime,
            vector_backend=vector_backend,
            chunk_strategy=chunk_strategy,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        runs.append(_run_matrix_case(run_config))

    payload = {
        "num_runs": len(runs),
        "runtimes": runtimes,
        "chunk_strategies": chunk_strategies,
        "vector_backends": vector_backends,
        "top_ks": top_ks,
        "chunk_sizes": chunk_sizes,
        "chunk_overlaps": chunk_overlaps,
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
        "retrieval_config": _retrieval_config_snapshot(config),
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


def _retrieval_config_snapshot(config: AppConfig) -> dict[str, object]:
    return {
        "vector_backend": config.vector_backend,
        "chunk_strategy": config.chunk_strategy,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "docs_dir": str(config.docs_dir),
        "docs_exclude_patterns": list(config.docs_exclude_patterns),
    }


def _run_label(config: AppConfig) -> str:
    return (
        f"{config.agent_runtime}:{config.vector_backend}:{config.chunk_strategy}"
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
