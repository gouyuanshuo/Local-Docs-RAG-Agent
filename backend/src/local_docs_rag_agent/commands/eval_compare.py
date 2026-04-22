from __future__ import annotations

import json
from pathlib import Path

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.evals.comparison import run_eval_matrix


DEFAULT_CHUNK_STRATEGIES = ["fixed", "paragraph", "markdown"]


def run_eval_compare_command(
    config: AppConfig,
    runtimes: list[str] | None = None,
    chunk_strategies: list[str] | None = None,
    vector_backends: list[str] | None = None,
    output_path: str | None = None,
) -> None:
    runtimes = runtimes or [config.agent_runtime]
    chunk_strategies = chunk_strategies or list(DEFAULT_CHUNK_STRATEGIES)
    vector_backends = vector_backends or _default_vector_backends(config)
    compare_output = Path(output_path) if output_path else Path("data/evals/compare_latest.json")

    payload = run_eval_matrix(
        config=config,
        runtimes=runtimes,
        chunk_strategies=chunk_strategies,
        vector_backends=vector_backends,
        output_path=compare_output,
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))


def _default_vector_backends(config: AppConfig) -> list[str]:
    backends = ["local"]
    if config.qdrant_url:
        backends.append("qdrant")
    return backends
