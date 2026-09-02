"""One-question facade over the runtime layer.

Delivery code holds a `LocalDocsAgent` rather than calling the dispatcher directly, so
the CLI, the API, the eval harness, and the live verification script all enter the
system through the same door.
"""

from __future__ import annotations

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import AgentAnswer
from local_docs_rag_agent.runtime import answer_question


class LocalDocsAgent:
    """Answers questions about the local documents under one configuration."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def answer(self, question: str) -> AgentAnswer:
        """Answer `question` with citations and diagnostics for how it was produced."""

        return answer_question(self._config, question)
