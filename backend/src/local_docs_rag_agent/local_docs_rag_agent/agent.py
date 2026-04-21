from __future__ import annotations

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import AgentAnswer
from local_docs_rag_agent.runtime import answer_question


class LocalDocsAgent:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def answer(self, question: str) -> AgentAnswer:
        return answer_question(self._config, question)
