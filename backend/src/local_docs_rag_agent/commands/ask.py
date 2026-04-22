from __future__ import annotations

import json

from local_docs_rag_agent.agent import LocalDocsAgent
from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.presenters import serialize_answer
from local_docs_rag_agent.rag.ingest import ensure_index


def run_ask(config: AppConfig, question: str, runtime: str | None = None) -> None:
    config = config.with_runtime(runtime)
    ensure_index(config)
    agent = LocalDocsAgent(config)
    answer = agent.answer(question)
    payload = serialize_answer(answer, runtime=answer.diagnostics.actual_runtime)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
