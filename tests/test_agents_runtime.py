from __future__ import annotations

import asyncio
import logging
import socket
import sys
import types
from typing import Any, ClassVar, NoReturn

import pytest

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import rag, tools
from local_docs_rag_agent.core import models
from local_docs_rag_agent.providers import openai_client
from local_docs_rag_agent.rag import scoring
from local_docs_rag_agent.runtime import agents_sdk
from local_docs_rag_agent.runtime import basic as basic_runtime


class FakeAsyncOpenAI:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    def is_closed(self) -> bool:
        return self.closed


class FakeRunContextWrapper:
    def __init__(self, context: agents_sdk.AgentRuntimeContext) -> None:
        self.context = context

    @classmethod
    def __class_getitem__(cls, item: object) -> type[FakeRunContextWrapper]:
        del item
        return cls


class FakeAgent:
    created: ClassVar[list[FakeAgent]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.name = kwargs["name"]
        self.model = kwargs["model"]
        self.instructions = kwargs["instructions"]
        self.tools = kwargs["tools"]
        self.created.append(self)


class FakeResponsesModel:
    def __init__(self, **kwargs: Any) -> None:
        self.model = kwargs["model"]
        self.openai_client = kwargs["openai_client"]


class FakeChatCompletionsModel(FakeResponsesModel):
    pass


class FakeRunner:
    calls: ClassVar[list[dict[str, object]]] = []

    @classmethod
    async def run(
        cls,
        agent: FakeAgent,
        question: str,
        *,
        context: agents_sdk.AgentRuntimeContext,
        max_turns: int,
    ) -> types.SimpleNamespace:
        wrapper = FakeRunContextWrapper(context)
        tool_output = agent.tools[1](wrapper, question, 1)
        cls.calls.append(
            {
                "question": question,
                "max_turns": max_turns,
                "tool_output": tool_output,
            }
        )
        return types.SimpleNamespace(
            final_output="Attention uses queries and keys. [S1]"
        )


def _install_fake_agents(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_module = types.ModuleType("agents")
    # Populated through the module namespace rather than by attribute
    # assignment: a synthetic module has no declared attributes for a type
    # checker to accept.
    fake_module.__dict__.update(
        Agent=FakeAgent,
        RunContextWrapper=FakeRunContextWrapper,
        function_tool=lambda function: function,
        OpenAIResponsesModel=FakeResponsesModel,
        OpenAIChatCompletionsModel=FakeChatCompletionsModel,
        Runner=FakeRunner,
    )
    monkeypatch.setitem(sys.modules, "agents", fake_module)
    FakeAgent.created.clear()
    FakeRunner.calls.clear()
    return fake_module


def _config(api_style: str) -> app_config.AppConfig:
    return app_config.AppConfig.from_env().with_overrides(
        agent_runtime="agents_sdk",
        vector_backend="local",
        llm_api_key="fake-key",
        llm_base_url="https://provider.invalid/v1",
        llm_model="fake-model",
        llm_api_style=api_style,
    )


def _retrieval_outcome() -> models.RetrievalOutcome:
    chunk = models.DocumentChunk(
        chunk_id="attention::0",
        source_path="docs/attention.md",
        title="Attention",
        text="Attention uses queries, keys, and values.",
        chunk_index=0,
        start_char=0,
        end_char=41,
        embedding=[1.0],
    )
    return models.RetrievalOutcome(
        hits=[scoring.build_retrieval_hit(chunk, 0.95)],
        embedding_status=models.ProviderStatus(
            provider="fake-embedding", mode="live"
        ),
        reranker_status=models.ProviderStatus(
            provider="none", mode="ready", reason="reranker_disabled"
        ),
    )


def _search_outcome(
    *,
    chunk_id: str,
    source_path: str,
    embedding_status: models.ProviderStatus,
    reranker_status: models.ProviderStatus,
) -> models.RetrievalOutcome:
    chunk = models.DocumentChunk(
        chunk_id=chunk_id,
        source_path=source_path,
        title=chunk_id.upper(),
        text=f"Evidence for {chunk_id}.",
        chunk_index=0,
        start_char=0,
        end_char=len(f"Evidence for {chunk_id}."),
        embedding=[1.0],
    )
    return models.RetrievalOutcome(
        hits=[scoring.build_retrieval_hit(chunk, 0.9)],
        embedding_status=embedding_status,
        reranker_status=reranker_status,
    )


@pytest.mark.parametrize("api_style", ["responses", "chat_completions"])
def test_real_sdk_registers_tools(api_style: str) -> None:
    config = _config(api_style)
    client = openai_client.build_async_openai_client(
        api_key="fake-key",
        base_url="https://provider.invalid/v1",
        trust_env=False,
    )
    try:
        built = agents_sdk._build_sdk_agent(config, client)
        assert {tool.name for tool in built.tools} == {
            "list_local_documents",
            "search_local_documents",
            "get_current_time",
        }
    finally:
        asyncio.run(client.close())


def test_real_sdk_public_runtime_uses_fake_runner_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agents_module = pytest.importorskip("agents")
    registered_tools: set[str] = set()

    def block_network(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("The Agents SDK runtime test must stay offline.")

    async def fake_run(
        agent: Any,
        question: str,
        *,
        context: agents_sdk.AgentRuntimeContext,
        max_turns: int,
    ) -> types.SimpleNamespace:
        del question, context, max_turns
        registered_tools.update(tool.name for tool in agent.tools)
        return types.SimpleNamespace(final_output="Offline SDK answer.")

    monkeypatch.setattr(socket, "create_connection", block_network)
    monkeypatch.setattr(
        asyncio.BaseEventLoop, "create_connection", block_network
    )
    monkeypatch.setattr(agents_module.Runner, "run", fake_run)

    answer = agents_sdk.answer_with_agents_sdk(_config("responses"), "question")

    assert answer.answer == "Offline SDK answer."
    assert answer.diagnostics.actual_runtime == "agents_sdk"
    assert answer.diagnostics.embedding_provider.mode == "unknown"
    assert answer.diagnostics.embedding_provider.reason == "search_not_run"
    assert answer.diagnostics.reranker.mode == "unknown"
    assert answer.diagnostics.reranker.reason == "search_not_run"
    assert registered_tools == {
        "list_local_documents",
        "search_local_documents",
        "get_current_time",
    }


@pytest.mark.parametrize(
    ("api_style", "expected_model_type"),
    [
        ("responses", FakeResponsesModel),
        ("chat_completions", FakeChatCompletionsModel),
    ],
)
def test_agents_runtime_uses_fake_runner_and_preserves_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    api_style: str,
    expected_model_type: type[FakeResponsesModel],
) -> None:
    _install_fake_agents(monkeypatch)
    client = FakeAsyncOpenAI()
    monkeypatch.setattr(agents_sdk, "_supports_agents_sdk", lambda: True)
    monkeypatch.setattr(
        openai_client, "build_async_openai_client", lambda **kwargs: client
    )
    monkeypatch.setattr(
        tools,
        "search_documents",
        lambda config, query, top_k: _retrieval_outcome(),
    )

    config = _config(api_style)
    answer = agents_sdk.answer_with_agents_sdk(config, "Explain attention")

    assert answer.answer == "Attention uses queries and keys. [S1]"
    assert answer.citations == ["docs/attention.md"]
    assert [span.chunk_id for span in answer.citation_spans] == ["attention::0"]
    assert [hit.chunk.chunk_id for hit in answer.retrieved_chunks] == [
        "attention::0"
    ]
    assert answer.diagnostics.requested_runtime == "agents_sdk"
    assert answer.diagnostics.actual_runtime == "agents_sdk"
    assert answer.diagnostics.chat_provider.mode == "live"
    assert answer.diagnostics.embedding_provider.mode == "live"
    assert answer.diagnostics.reranker.reason == "reranker_disabled"
    assert client.closed is True
    assert len(FakeRunner.calls) == 1
    assert FakeRunner.calls[0]["max_turns"] == config.agents_max_turns
    assert "docs/attention.md" in str(FakeRunner.calls[0]["tool_output"])
    assert isinstance(FakeAgent.created[0].model, expected_model_type)


@pytest.mark.parametrize(
    (
        "first_mode",
        "first_reason",
        "second_mode",
        "second_reason",
        "expected_reason",
    ),
    [
        (
            "fallback",
            "primary_failed",
            "live",
            "provider_recovered",
            "primary_failed;provider_recovered",
        ),
        (
            "live",
            "provider_available",
            "fallback",
            "later_failed",
            "provider_available;later_failed",
        ),
    ],
)
def test_agents_searches_keep_source_ids_and_conservative_statuses(
    monkeypatch: pytest.MonkeyPatch,
    first_mode: models.ProviderMode,
    first_reason: str,
    second_mode: models.ProviderMode,
    second_reason: str,
    expected_reason: str,
) -> None:
    fake_module = _install_fake_agents(monkeypatch)
    client = FakeAsyncOpenAI()
    tool_outputs: list[str] = []

    class ThreeSearchRunner:
        @staticmethod
        async def run(
            agent: FakeAgent,
            question: str,
            *,
            context: agents_sdk.AgentRuntimeContext,
            max_turns: int,
        ) -> types.SimpleNamespace:
            del question, max_turns
            wrapper = FakeRunContextWrapper(context)
            tool_outputs.extend(
                [
                    agent.tools[1](wrapper, "find a", 1),
                    agent.tools[1](wrapper, "find b", 1),
                    agent.tools[1](wrapper, "find a again", 1),
                ]
            )
            return types.SimpleNamespace(
                final_output="Combined evidence. [S1] [S2]"
            )

    fake_module.__dict__["Runner"] = ThreeSearchRunner
    monkeypatch.setattr(agents_sdk, "_supports_agents_sdk", lambda: True)
    monkeypatch.setattr(
        openai_client, "build_async_openai_client", lambda **kwargs: client
    )
    statuses = [
        models.ProviderStatus(
            provider="test-provider", mode=first_mode, reason=first_reason
        ),
        models.ProviderStatus(
            provider="test-provider", mode=second_mode, reason=second_reason
        ),
        models.ProviderStatus(
            provider="test-provider", mode=first_mode, reason=first_reason
        ),
    ]
    outcomes = iter(
        [
            _search_outcome(
                chunk_id="a",
                source_path="a.md",
                embedding_status=statuses[0],
                reranker_status=statuses[0],
            ),
            _search_outcome(
                chunk_id="b",
                source_path="b.md",
                embedding_status=statuses[1],
                reranker_status=statuses[1],
            ),
            _search_outcome(
                chunk_id="a",
                source_path="a.md",
                embedding_status=statuses[2],
                reranker_status=statuses[2],
            ),
        ]
    )
    monkeypatch.setattr(
        tools, "search_documents", lambda config, query, top_k: next(outcomes)
    )

    answer = agents_sdk.answer_with_agents_sdk(
        _config("responses"), "combine a and b"
    )

    assert [output.splitlines()[0] for output in tool_outputs] == [
        "[S1]",
        "[S2]",
        "[S1]",
    ]
    assert [hit.chunk.chunk_id for hit in answer.retrieved_chunks] == ["a", "b"]
    assert [span.source_id for span in answer.citation_spans] == ["S1", "S2"]
    assert [hit.citation_span.source_id for hit in answer.retrieved_chunks] == [
        None,
        None,
    ]
    assert answer.diagnostics.embedding_provider.mode == "fallback"
    assert answer.diagnostics.embedding_provider.reason == expected_reason
    assert answer.diagnostics.reranker.mode == "fallback"
    assert answer.diagnostics.reranker.reason == expected_reason
    assert client.closed is True


@pytest.mark.parametrize(
    ("failure_kind", "expected_runtime_reason"),
    [
        (
            "runner_error",
            "runtime_fallback:agents_sdk_error:RuntimeError",
        ),
        ("empty_output", "runtime_fallback:empty_agent_output"),
        ("none_output", "runtime_fallback:empty_agent_output"),
        (
            "raising_output",
            "runtime_fallback:agents_sdk_error:RuntimeError",
        ),
    ],
)
def test_agents_fallback_preserves_prior_search_degradation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure_kind: str,
    expected_runtime_reason: str,
) -> None:
    fake_module = _install_fake_agents(monkeypatch)
    client = FakeAsyncOpenAI()
    output_error_sentinel = "sensitive-final-output-sentinel"

    class RaisingFinalOutput:
        def __str__(self) -> str:
            raise RuntimeError(output_error_sentinel)

    class SearchThenFallbackRunner:
        @staticmethod
        async def run(
            agent: FakeAgent,
            question: str,
            *,
            context: agents_sdk.AgentRuntimeContext,
            max_turns: int,
        ) -> types.SimpleNamespace:
            del question, max_turns
            agent.tools[1](FakeRunContextWrapper(context), "find a", 1)
            if failure_kind == "runner_error":
                raise RuntimeError("runner stopped after search")
            final_output: object = "   "
            if failure_kind == "none_output":
                final_output = None
            elif failure_kind == "raising_output":
                final_output = RaisingFinalOutput()
            return types.SimpleNamespace(final_output=final_output)

    fake_module.__dict__["Runner"] = SearchThenFallbackRunner
    monkeypatch.setattr(agents_sdk, "_supports_agents_sdk", lambda: True)
    monkeypatch.setattr(
        openai_client, "build_async_openai_client", lambda **kwargs: client
    )
    monkeypatch.setattr(
        tools,
        "search_documents",
        lambda config, query, top_k: _search_outcome(
            chunk_id="a",
            source_path="a.md",
            embedding_status=models.ProviderStatus(
                provider="embedding",
                mode="fallback",
                reason="tool_embedding_failed",
            ),
            reranker_status=models.ProviderStatus(
                provider="reranker",
                mode="fallback",
                reason="tool_reranker_failed",
            ),
        ),
    )
    basic_answer = models.AgentAnswer(
        question="combine a and b",
        answer="Basic fallback answer.",
        citations=[],
        citation_spans=[],
        retrieved_chunks=[],
        diagnostics=models.AnswerDiagnostics(
            requested_runtime="agents_sdk",
            actual_runtime="basic",
            vector_backend="local",
            chat_provider=models.ProviderStatus(
                provider="chat",
                mode="fallback",
                reason="runtime_fallback:agents_sdk_error:RuntimeError",
            ),
            embedding_provider=models.ProviderStatus(
                provider="embedding",
                mode="live",
                reason="basic_embedding_recovered",
            ),
            reranker=models.ProviderStatus(
                provider="reranker",
                mode="live",
                reason="basic_reranker_recovered",
            ),
        ),
    )
    fallback_call: dict[str, object] = {}

    def fake_basic_runtime(
        *args: object, **kwargs: object
    ) -> models.AgentAnswer:
        del args
        fallback_call.update(kwargs)
        return basic_answer

    monkeypatch.setattr(
        basic_runtime, "answer_with_basic_runtime", fake_basic_runtime
    )
    caplog.set_level(logging.WARNING, logger=agents_sdk.__name__)

    answer = agents_sdk.answer_with_agents_sdk(
        _config("responses"), "combine a and b"
    )

    assert answer.answer == "Basic fallback answer."
    assert answer.diagnostics.actual_runtime == "basic"
    assert fallback_call["runtime_reason"] == expected_runtime_reason
    assert answer.diagnostics.embedding_provider.mode == "fallback"
    assert (
        answer.diagnostics.embedding_provider.reason
        == "tool_embedding_failed;basic_embedding_recovered"
    )
    assert answer.diagnostics.reranker.mode == "fallback"
    assert (
        answer.diagnostics.reranker.reason
        == "tool_reranker_failed;basic_reranker_recovered"
    )
    assert basic_answer.diagnostics.embedding_provider.mode == "live"
    assert basic_answer.diagnostics.reranker.mode == "live"
    assert client.closed is True
    assert output_error_sentinel not in "\n".join(
        record.getMessage() for record in caplog.records
    )


def test_agents_runtime_closes_client_and_exposes_runner_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = _install_fake_agents(monkeypatch)

    class FailingRunner:
        @staticmethod
        async def run(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise RuntimeError("fake runner failed")

    fake_module.__dict__["Runner"] = FailingRunner
    client = FakeAsyncOpenAI()
    fallback_answer = object()
    fallback_call: dict[str, object] = {}

    def fake_basic_runtime(*args: object, **kwargs: object) -> object:
        fallback_call.update(kwargs)
        return fallback_answer

    monkeypatch.setattr(agents_sdk, "_supports_agents_sdk", lambda: True)
    monkeypatch.setattr(
        openai_client, "build_async_openai_client", lambda **kwargs: client
    )
    monkeypatch.setattr(
        basic_runtime, "answer_with_basic_runtime", fake_basic_runtime
    )

    answer = agents_sdk.answer_with_agents_sdk(_config("responses"), "question")

    assert answer is fallback_answer
    assert client.closed is True
    assert fallback_call["requested_runtime"] == "agents_sdk"
    assert (
        fallback_call["runtime_reason"]
        == "runtime_fallback:agents_sdk_error:RuntimeError"
    )


def test_agents_runtime_logs_safe_registration_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeAsyncOpenAI()
    fallback_answer = object()
    fallback_call: dict[str, object] = {}
    sensitive_values = (
        "sdk-log-api-key-sentinel",
        "sdk-log-document-sentinel",
        "https://sdk-log-provider-sentinel.invalid/v1",
        "sdk-log-question-sentinel",
    )

    def raise_sensitive_registration_error(
        *args: object, **kwargs: object
    ) -> Any:
        del args, kwargs
        raise RuntimeError(" ".join(sensitive_values))

    def fake_basic_runtime(*args: object, **kwargs: object) -> object:
        del args
        fallback_call.update(kwargs)
        return fallback_answer

    monkeypatch.setattr(agents_sdk, "_supports_agents_sdk", lambda: True)
    monkeypatch.setattr(
        openai_client, "build_async_openai_client", lambda **kwargs: client
    )
    monkeypatch.setattr(
        agents_sdk, "_build_sdk_agent", raise_sensitive_registration_error
    )
    monkeypatch.setattr(
        basic_runtime, "answer_with_basic_runtime", fake_basic_runtime
    )
    caplog.set_level(logging.WARNING, logger=agents_sdk.__name__)

    answer = agents_sdk.answer_with_agents_sdk(
        _config("responses").with_overrides(
            llm_api_key=sensitive_values[0],
            llm_base_url=sensitive_values[2],
        ),
        sensitive_values[3],
    )

    assert answer is fallback_answer
    assert client.closed is True
    assert (
        fallback_call["runtime_reason"]
        == "runtime_fallback:agents_sdk_error:RuntimeError"
    )
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "exception_type=RuntimeError" in messages[0]
    assert (
        f"module={raise_sensitive_registration_error.__module__}" in messages[0]
    )
    assert "function=raise_sensitive_registration_error" in messages[0]
    assert messages[0].rsplit("line=", maxsplit=1)[1].isdecimal()
    for sensitive_value in sensitive_values:
        assert sensitive_value not in messages[0]


def test_search_documents_returns_pipeline_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = _retrieval_outcome()
    monkeypatch.setattr(rag, "retrieve", lambda config, query, top_k: outcome)

    result = tools.search_documents(_config("responses"), "attention", 1)

    assert result is outcome
    assert result.embedding_status.mode == "live"
    assert result.reranker_status.reason == "reranker_disabled"
