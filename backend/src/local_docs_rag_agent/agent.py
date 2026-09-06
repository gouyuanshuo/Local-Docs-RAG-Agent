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
        """Bind the agent to one configuration snapshot.

        Args:
          config: The settings every answer from this agent is produced
            under.
        """
        self._config = config

    def answer(self, question: str) -> models.AgentAnswer:
        """Return the answer to `question`, with citations and diagnostics."""
        return runtime.answer_question(self._config, question)
