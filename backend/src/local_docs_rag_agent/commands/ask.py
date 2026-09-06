from __future__ import annotations

import json

from local_docs_rag_agent import agent, presenters, rag
from local_docs_rag_agent import config as app_config


def run_ask(
    config: app_config.AppConfig, question: str, runtime: str | None = None
) -> None:
    config = config.with_runtime(runtime)
    rag.ensure_index(config)
    docs_agent = agent.LocalDocsAgent(config)
    answer = docs_agent.answer(question)
    payload = presenters.serialize_answer(answer)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
