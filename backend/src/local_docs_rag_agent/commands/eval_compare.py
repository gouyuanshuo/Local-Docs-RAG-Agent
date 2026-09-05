"""Runs the eval comparison matrix and writes its report."""

from __future__ import annotations

import json
from pathlib import Path

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.evals.comparison import (
    default_chunk_strategies,
    default_retrieval_strategies,
    default_vector_backends,
    run_eval_matrix,
)

DEFAULT_COMPARE_OUTPUT_PATH = Path("data/evals/compare_latest.json")


def run_eval_compare_command(
    config: AppConfig,
    runtimes: list[str] | None = None,
    chunk_strategies: list[str] | None = None,
    vector_backends: list[str] | None = None,
    top_ks: list[int] | None = None,
    chunk_sizes: list[int] | None = None,
    chunk_overlaps: list[int] | None = None,
    retrieval_strategies: list[str] | None = None,
    output_path: str | None = None,
) -> None:
    """Compare eval results across the requested axes and print the report as JSON.

    An omitted axis falls back to the configured value, so the default invocation
    compares chunk strategies while holding everything else steady.
    """

    payload = run_eval_matrix(
        config=config,
        runtimes=runtimes or [config.agent_runtime],
        chunk_strategies=chunk_strategies or default_chunk_strategies(),
        vector_backends=vector_backends or default_vector_backends(config),
        top_ks=top_ks or [config.top_k],
        chunk_sizes=chunk_sizes or [config.chunk_size],
        chunk_overlaps=chunk_overlaps or [config.chunk_overlap],
        retrieval_strategies=retrieval_strategies or default_retrieval_strategies(),
        output_path=Path(output_path) if output_path else DEFAULT_COMPARE_OUTPUT_PATH,
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))
