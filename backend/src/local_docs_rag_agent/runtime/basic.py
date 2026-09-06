"""Deterministic retrieve-then-answer runtime.

This is the baseline path and the fallback target for every other runtime: it
always retrieves first, answers only from that context, and records why it ran
when another runtime handed off to it.
"""

from __future__ import annotations

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import constants, models
from local_docs_rag_agent.providers import factory as provider_factory
from local_docs_rag_agent.rag import pipeline
from local_docs_rag_agent.runtime import shared as runtime_shared


def answer_with_basic_runtime(
    config: app_config.AppConfig,
    question: str,
    requested_runtime: constants.RuntimeName | None = None,
    runtime_reason: str | None = None,
) -> models.AgentAnswer:
    provider = provider_factory.build_chat_provider(config)
    outcome = pipeline.retrieve(config, question)
    hits = outcome.hits
    context = runtime_shared.build_answer_context(hits)
    answer = provider.answer(question=question, context=context)
    chat_status = provider.status
    if runtime_reason:
        chat_status = models.ProviderStatus(
            provider=chat_status.provider,
            mode=chat_status.mode,
            reason=runtime_reason
            if chat_status.reason is None
            else f"{chat_status.reason};{runtime_reason}",
        )

    diagnostics = models.AnswerDiagnostics(
        requested_runtime=requested_runtime or config.agent_runtime,
        actual_runtime="basic",
        vector_backend=config.vector_backend,
        chat_provider=chat_status,
        embedding_provider=outcome.embedding_status,
        reranker=outcome.reranker_status,
    )
    return runtime_shared.build_agent_answer(
        question=question, answer=answer, hits=hits, diagnostics=diagnostics
    )
