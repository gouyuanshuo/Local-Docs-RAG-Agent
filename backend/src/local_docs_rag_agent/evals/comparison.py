from __future__ import annotations

from pathlib import Path

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.evals.harness import run_eval
from local_docs_rag_agent.presenters import serialize_eval_summary
from local_docs_rag_agent.rag.ingest import ingest_documents


def run_eval_matrix(
    config: AppConfig,
    runtimes: list[str],
    chunk_strategies: list[str],
    vector_backends: list[str],
    output_path: Path | None = None,
) -> dict[str, object]:
    runs: list[dict[str, object]] = []

    for runtime in runtimes:
        for vector_backend in vector_backends:
            for chunk_strategy in chunk_strategies:
                run_config = config.with_overrides(
                    agent_runtime=runtime,
                    vector_backend=vector_backend,
                    chunk_strategy=chunk_strategy,
                )
                run_label = f"{runtime}:{vector_backend}:{chunk_strategy}"
                skip_reason = _skip_reason(run_config)
                if skip_reason is not None:
                    runs.append(
                        {
                            "label": run_label,
                            "status": "skipped",
                            "reason": skip_reason,
                            "retrieval_config": _retrieval_config_snapshot(run_config),
                            "runtime": run_config.agent_runtime,
                        }
                    )
                    continue

                try:
                    ingest_documents(run_config)
                    results = run_eval(run_config)
                    summary = serialize_eval_summary(results, runtime=run_config.agent_runtime, config=run_config)
                    runs.append(
                        {
                            "label": run_label,
                            "status": "ok",
                            "summary": summary,
                        }
                    )
                except Exception as exc:
                    skip_reason = _qdrant_runtime_skip_reason(run_config, exc)
                    if skip_reason is not None:
                        runs.append(
                            {
                                "label": run_label,
                                "status": "skipped",
                                "reason": skip_reason,
                                "retrieval_config": _retrieval_config_snapshot(run_config),
                                "runtime": run_config.agent_runtime,
                            }
                        )
                        continue
                    runs.append(
                        {
                            "label": run_label,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "retrieval_config": _retrieval_config_snapshot(run_config),
                            "runtime": run_config.agent_runtime,
                        }
                    )

    payload = {
        "num_runs": len(runs),
        "runtimes": runtimes,
        "chunk_strategies": chunk_strategies,
        "vector_backends": vector_backends,
        "leaderboard": _build_leaderboard(runs),
        "runs": runs,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_dump_json(payload), encoding="utf-8")
    return payload


def _dump_json(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=True, indent=2)


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


def _skip_reason(config: AppConfig) -> str | None:
    if config.vector_backend == "qdrant" and not config.qdrant_url:
        return "missing_qdrant_url"
    return None


def _qdrant_runtime_skip_reason(config: AppConfig, exc: Exception) -> str | None:
    if config.vector_backend != "qdrant":
        return None

    error_name = type(exc).__name__
    error_text = f"{error_name}: {exc}"
    if error_name in {"ResponseHandlingException", "UnexpectedResponse"}:
        return "qdrant_unreachable"
    if "Connection refused" in error_text or "WinError 10061" in error_text:
        return "qdrant_unreachable"
    if "qdrant-client is not installed" in error_text:
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
                "label": run["label"],
                "answer_keyword_hit_rate": summary.get("answer_keyword_hit_rate", 0.0),
                "retrieval_source_hit_rate": summary.get("retrieval_source_hit_rate", 0.0),
                "retrieval_span_hit_rate": summary.get("retrieval_span_hit_rate", 0.0),
                "citation_span_hit_rate": summary.get("citation_span_hit_rate", 0.0),
                "avg_response_time_ms": summary.get("avg_response_time_ms", 0.0),
            }
        )
    leaderboard.sort(
        key=lambda row: (
            row["retrieval_span_hit_rate"],
            row["answer_keyword_hit_rate"],
            row["citation_span_hit_rate"],
            -row["avg_response_time_ms"],
        ),
        reverse=True,
    )
    return leaderboard
