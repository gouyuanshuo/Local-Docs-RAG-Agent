"""Selects the runtime that answers a question.

Runtime selection is the only decision made here; each runtime is responsible
for returning a complete `AgentAnswer`, including diagnostics that say whether
it ran or delegated to another runtime.
"""

from __future__ import annotations

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import models
from local_docs_rag_agent.runtime import agents_sdk
from local_docs_rag_agent.runtime import basic as basic_runtime


def answer_question(
    config: app_config.AppConfig, question: str
) -> models.AgentAnswer:
    if config.agent_runtime == "agents_sdk":
        return agents_sdk.answer_with_agents_sdk(config, question)
    return basic_runtime.answer_with_basic_runtime(config, question)
