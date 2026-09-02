from __future__ import annotations

import json

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.evals.harness import run_eval
from local_docs_rag_agent.presenters import serialize_eval_summary
from local_docs_rag_agent.rag import ensure_index


def run_eval_command(config: AppConfig, runtime: str | None = None) -> None:
    config = config.with_runtime(runtime)
    ensure_index(config)
    results = run_eval(config)
    summary = serialize_eval_summary(results, runtime=config.agent_runtime, config=config)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
