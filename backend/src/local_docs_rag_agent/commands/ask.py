"""Answers one question from the command line."""

from __future__ import annotations

import json

from local_docs_rag_agent import agent, presenters, rag
from local_docs_rag_agent import config as app_config


def run_ask(
    config: app_config.AppConfig, question: str, runtime: str | None = None
) -> None:
    """Answer one question and print the result as JSON.

    Args:
      config: The settings to answer under.
      question: The question to answer.
      runtime: Runtime override for this command, or None to use the
        configured one.
    """
    config = config.with_runtime(runtime)
    rag.ensure_index(config)
    docs_agent = agent.LocalDocsAgent(config)
    answer = docs_agent.answer(question)
    payload = presenters.serialize_answer(answer)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
