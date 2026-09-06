"""Runs the eval comparison matrix and writes its report."""

from __future__ import annotations

import json
import pathlib

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.evals import comparison

DEFAULT_COMPARE_OUTPUT_PATH = pathlib.Path("data/evals/compare_latest.json")


def run_eval_compare_command(
    config: app_config.AppConfig,
    runtimes: list[str] | None = None,
    chunk_strategies: list[str] | None = None,
    vector_backends: list[str] | None = None,
    top_ks: list[int] | None = None,
    chunk_sizes: list[int] | None = None,
    chunk_overlaps: list[int] | None = None,
    retrieval_strategies: list[str] | None = None,
    rerankers: list[str] | None = None,
    output_path: str | None = None,
) -> None:
    """Compare eval results across the requested axes and print the report as JSON.

    An omitted axis falls back to the configured value, so the default
    invocation compares chunk strategies while holding everything else steady.
    """

    payload = comparison.run_eval_matrix(
        config=config,
        runtimes=runtimes or [config.agent_runtime],
        chunk_strategies=chunk_strategies
        or comparison.default_chunk_strategies(),
        vector_backends=vector_backends
        or comparison.default_vector_backends(config),
        top_ks=top_ks or [config.top_k],
        chunk_sizes=chunk_sizes or [config.chunk_size],
        chunk_overlaps=chunk_overlaps or [config.chunk_overlap],
        retrieval_strategies=retrieval_strategies
        or comparison.default_retrieval_strategies(),
        rerankers=rerankers or comparison.default_rerankers(config),
        output_path=pathlib.Path(output_path)
        if output_path
        else DEFAULT_COMPARE_OUTPUT_PATH,
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))
