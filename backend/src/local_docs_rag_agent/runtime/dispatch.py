"""Selects the runtime that answers a question.

Runtime selection is the only decision made here; each runtime is responsible for
returning a complete `AgentAnswer`, including diagnostics that say whether it ran or
delegated to another runtime.
"""

from __future__ import annotations

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import AgentAnswer
from local_docs_rag_agent.runtime.agents_sdk import answer_with_agents_sdk
from local_docs_rag_agent.runtime.basic import answer_with_basic_runtime


def answer_question(config: AppConfig, question: str) -> AgentAnswer:
    if config.agent_runtime == "agents_sdk":
        return answer_with_agents_sdk(config, question)
    return answer_with_basic_runtime(config, question)
