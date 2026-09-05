"""Tool-capable runtime backed by the OpenAI Agents SDK.

The SDK is imported lazily so the package stays installable without the `agents`
extra. Any missing dependency, missing key, or run failure degrades to the basic
runtime with a `runtime_fallback:` reason recorded in the diagnostics rather than
raising, because a degraded answer is still useful as long as it is labelled as one.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any

from openai import AsyncOpenAI

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import AgentAnswer, AnswerDiagnostics, ProviderStatus, RetrievalHit
from local_docs_rag_agent.providers.openai_client import build_async_openai_client
from local_docs_rag_agent.rag.pipeline import retrieve
from local_docs_rag_agent.runtime.basic import answer_with_basic_runtime
from local_docs_rag_agent.runtime.shared import (
    build_agent_answer,
    format_tool_search_results,
    merge_hits,
)
from local_docs_rag_agent.tools import get_system_time, list_documents


@dataclass(slots=True)
class AgentRuntimeContext:
    config: AppConfig
    retrieved_hits: list[RetrievalHit] = field(default_factory=list)
    # The agent decides whether to search at all, so both retrieval statuses start
    # `unknown` and are only replaced if the search tool actually runs.
    embedding_status: ProviderStatus = field(
        default_factory=lambda: ProviderStatus(
            provider="embedding", mode="unknown", reason="search_not_run"
        )
    )
    reranker_status: ProviderStatus = field(
        default_factory=lambda: ProviderStatus(
            provider="reranker", mode="unknown", reason="search_not_run"
        )
    )


def answer_with_agents_sdk(config: AppConfig, question: str) -> AgentAnswer:
    agents_sdk_available = _supports_agents_sdk()
    if not agents_sdk_available or not config.llm_api_key:
        fallback_reason = (
            "agents_sdk_unavailable" if not agents_sdk_available else "missing_llm_api_key"
        )
        return answer_with_basic_runtime(
            config,
            question,
            requested_runtime="agents_sdk",
            runtime_reason=f"runtime_fallback:{fallback_reason}",
        )

    openai_client = build_async_openai_client(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        trust_env=config.external_http_trust_env,
    )
    run_context = AgentRuntimeContext(config=config)
    try:
        agent = _build_sdk_agent(config, openai_client)
        result = asyncio.run(
            _run_agent(
                agent=agent,
                question=question,
                run_context=run_context,
                max_turns=config.agents_max_turns,
                openai_client=openai_client,
            )
        )
    except Exception as exc:
        if not openai_client.is_closed():
            asyncio.run(openai_client.close())
        return answer_with_basic_runtime(
            config,
            question,
            requested_runtime="agents_sdk",
            runtime_reason=f"runtime_fallback:agents_sdk_error:{exc.__class__.__name__}",
        )

    final_output = str(result.final_output).strip()
    if not final_output:
        return answer_with_basic_runtime(
            config,
            question,
            requested_runtime="agents_sdk",
            runtime_reason="runtime_fallback:empty_agent_output",
        )

    diagnostics = AnswerDiagnostics(
        requested_runtime="agents_sdk",
        actual_runtime="agents_sdk",
        vector_backend=config.vector_backend,
        chat_provider=ProviderStatus(provider=config.llm_provider, mode="live"),
        embedding_provider=run_context.embedding_status,
        reranker=run_context.reranker_status,
    )
    return build_agent_answer(
        question=question,
        answer=final_output,
        hits=run_context.retrieved_hits,
        diagnostics=diagnostics,
    )


def _build_sdk_agent(config: AppConfig, openai_client: AsyncOpenAI) -> Any:
    from agents import Agent, RunContextWrapper, function_tool

    # function_tool resolves postponed annotations from the module global namespace.
    globals()["RunContextWrapper"] = RunContextWrapper

    @function_tool
    def list_local_documents(ctx: RunContextWrapper[AgentRuntimeContext]) -> str:
        """List the local documents that are available for question answering."""
        documents = list_documents(ctx.context.config)
        if not documents:
            return "No local documents were found."
        return "\n".join(documents)

    @function_tool
    def search_local_documents(
        ctx: RunContextWrapper[AgentRuntimeContext],
        query: str,
        top_k: int | None = None,
    ) -> str:
        """Search local documents for evidence before answering a docs question."""
        # The tool's own `top_k` is applied by retrieval rather than by truncating
        # afterwards, so the reranker reorders the window the agent actually asked for.
        outcome = retrieve(ctx.context.config, query, top_k)
        ctx.context.embedding_status = outcome.embedding_status
        ctx.context.reranker_status = outcome.reranker_status
        merge_hits(ctx.context.retrieved_hits, outcome.hits)
        return format_tool_search_results(outcome.hits)

    @function_tool
    def get_current_time() -> str:
        """Return the current UTC time."""
        return get_system_time()

    return Agent(
        name="Local Docs RAG Agent",
        model=_build_agent_model(config, openai_client),
        instructions=dedent(
            """
            You are a local-document QA agent.
            For document questions, call `search_local_documents` before answering.
            Call `list_local_documents` when the user asks what files are available.
            Only answer from evidence. If evidence is insufficient, say what is missing.
            Cite source ids inline like [S1], [S2].
            End with a `Sources:` line naming the relevant `source_path` values.
            """
        ).strip(),
        tools=[list_local_documents, search_local_documents, get_current_time],
    )


async def _run_agent(
    *,
    agent: Any,
    question: str,
    run_context: AgentRuntimeContext,
    max_turns: int,
    openai_client: AsyncOpenAI,
) -> Any:
    from agents import Runner

    try:
        return await Runner.run(
            agent,
            question,
            context=run_context,
            max_turns=max_turns,
        )
    finally:
        await openai_client.close()


def _supports_agents_sdk() -> bool:
    try:
        import agents  # noqa: F401
    except Exception:
        return False
    return True


def _build_agent_model(config: AppConfig, openai_client: AsyncOpenAI) -> Any:
    from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel

    if config.llm_api_style == "chat_completions":
        return OpenAIChatCompletionsModel(
            model=config.llm_model,
            openai_client=openai_client,
        )
    return OpenAIResponsesModel(
        model=config.llm_model,
        openai_client=openai_client,
    )
