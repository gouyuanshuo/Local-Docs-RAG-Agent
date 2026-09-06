"""One-question facade over the runtime layer.

Delivery code holds a `LocalDocsAgent` rather than calling the dispatcher
directly, so the CLI, the API, the eval harness, and the live verification
script all enter the system through the same door.
"""

from __future__ import annotations

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import models, runtime


class LocalDocsAgent:
    """Answers questions about the local documents under one configuration."""

    def __init__(self, config: app_config.AppConfig) -> None:
        self._config = config

    def answer(self, question: str) -> models.AgentAnswer:
        """Answer `question` with citations and diagnostics for how it was produced."""

        return runtime.answer_question(self._config, question)
