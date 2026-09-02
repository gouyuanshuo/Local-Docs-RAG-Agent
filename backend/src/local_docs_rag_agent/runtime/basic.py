"""Deterministic retrieve-then-answer runtime.

This is the baseline path and the fallback target for every other runtime: it always
retrieves first, answers only from that context, and records why it ran when another
runtime handed off to it.
"""

from __future__ import annotations

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.constants import RuntimeName
from local_docs_rag_agent.models import AgentAnswer, AnswerDiagnostics, ProviderStatus
from local_docs_rag_agent.providers.factory import build_chat_provider
from local_docs_rag_agent.runtime.shared import (
    build_agent_answer,
    build_answer_context,
    retrieve_hits,
)


def answer_with_basic_runtime(
    config: AppConfig,
    question: str,
    requested_runtime: RuntimeName | None = None,
    runtime_reason: str | None = None,
) -> AgentAnswer:
    provider = build_chat_provider(config)
    hits, embedding_status = retrieve_hits(config, question)
    context = build_answer_context(hits)
    answer = provider.answer(question=question, context=context)
    chat_status = provider.status
    if runtime_reason:
        chat_status = ProviderStatus(
            provider=chat_status.provider,
            mode=chat_status.mode,
            reason=runtime_reason
            if chat_status.reason is None
            else f"{chat_status.reason};{runtime_reason}",
        )

    diagnostics = AnswerDiagnostics(
        requested_runtime=requested_runtime or config.agent_runtime,
        actual_runtime="basic",
        vector_backend=config.vector_backend,
        chat_provider=chat_status,
        embedding_provider=embedding_status,
    )
    return build_agent_answer(question=question, answer=answer, hits=hits, diagnostics=diagnostics)
