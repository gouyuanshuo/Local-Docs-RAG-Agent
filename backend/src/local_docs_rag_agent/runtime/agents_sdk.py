from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import AgentAnswer, AnswerDiagnostics, ProviderStatus, RetrievalHit
from local_docs_rag_agent.runtime.basic import answer_with_basic_runtime
from local_docs_rag_agent.runtime.shared import (
    build_agent_answer,
    format_tool_search_results,
    merge_hits,
    retrieve_hits,
)
from local_docs_rag_agent.tools import get_system_time, list_documents


@dataclass(slots=True)
class AgentRuntimeContext:
    config: AppConfig
    retrieved_hits: list[RetrievalHit] = field(default_factory=list)
    embedding_status: ProviderStatus = field(
        default_factory=lambda: ProviderStatus(provider="embedding", mode="unknown", reason="search_not_run")
    )


def answer_with_agents_sdk(config: AppConfig, question: str) -> AgentAnswer:
    if not _supports_agents_sdk() or not config.llm_api_key:
        fallback_reason = "agents_sdk_unavailable" if not _supports_agents_sdk() else "missing_llm_api_key"
        return answer_with_basic_runtime(
            config,
            question,
            requested_runtime="agents_sdk",
            runtime_reason=f"runtime_fallback:{fallback_reason}",
        )

    from agents import Agent, RunContextWrapper, Runner, function_tool, set_default_openai_client
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient

    globals()["RunContextWrapper"] = RunContextWrapper

    client_kwargs: dict[str, object] = {
        "api_key": config.llm_api_key,
        "base_url": config.llm_base_url,
    }
    if not config.external_http_trust_env:
        client_kwargs["http_client"] = DefaultAsyncHttpxClient(trust_env=False)
    set_default_openai_client(AsyncOpenAI(**client_kwargs), use_for_tracing=False)

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
        """Search local documents for relevant evidence before answering a docs question."""
        hits, embedding_status = retrieve_hits(ctx.context.config, query)
        ctx.context.embedding_status = embedding_status
        if top_k is not None:
            hits = hits[:top_k]
        merge_hits(ctx.context.retrieved_hits, hits)
        return format_tool_search_results(hits)

    @function_tool
    def get_current_time() -> str:
        """Return the current UTC time."""
        return get_system_time()

    agent = Agent(
        name="Local Docs RAG Agent",
        model=config.llm_model,
        instructions=dedent(
            """
            You are a local-document QA agent.
            For questions about the user's documents, call `search_local_documents` before answering.
            You may call `list_local_documents` if the user is asking what files are available.
            Only answer from retrieved evidence. If the retrieved evidence is insufficient, say what is missing.
            Cite source ids inline like [S1], [S2].
            End with a short `Sources:` line that names the relevant `source_path` values.
            """
        ).strip(),
        tools=[list_local_documents, search_local_documents, get_current_time],
    )

    run_context = AgentRuntimeContext(config=config)
    try:
        result = Runner.run_sync(
            agent,
            question,
            context=run_context,
            max_turns=config.agents_max_turns,
        )
    except Exception as exc:
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
    )
    return build_agent_answer(
        question=question,
        answer=final_output,
        hits=run_context.retrieved_hits,
        diagnostics=diagnostics,
    )


def _supports_agents_sdk() -> bool:
    try:
        import agents  # noqa: F401
    except Exception:
        return False
    return True
