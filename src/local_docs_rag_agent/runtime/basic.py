from __future__ import annotations

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import AgentAnswer
from local_docs_rag_agent.providers.factory import build_chat_provider
from local_docs_rag_agent.runtime.shared import build_agent_answer, build_answer_context, retrieve_hits


def answer_with_basic_runtime(config: AppConfig, question: str) -> AgentAnswer:
    provider = build_chat_provider(config)
    hits = retrieve_hits(config, question)
    context = build_answer_context(hits)
    answer = provider.answer(question=question, context=context)
    return build_agent_answer(question=question, answer=answer, hits=hits)
