from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import DocumentChunk, ProviderStatus, RetrievalOutcome
from local_docs_rag_agent.rag.scoring import build_retrieval_hit
from local_docs_rag_agent.runtime import agents_sdk


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
    ) -> SimpleNamespace:
        wrapper = FakeRunContextWrapper(context)
        tool_output = agent.tools[1](wrapper, question, 1)
        cls.calls.append(
            {
                "question": question,
                "max_turns": max_turns,
                "tool_output": tool_output,
            }
        )
        return SimpleNamespace(final_output="Attention uses queries and keys. [S1]")


def _install_fake_agents(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    fake_module = ModuleType("agents")
    # Populated through the module namespace rather than by attribute assignment: a
    # synthetic module has no declared attributes for a type checker to accept.
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


def _config(api_style: str) -> AppConfig:
    return AppConfig.from_env().with_overrides(
        agent_runtime="agents_sdk",
        vector_backend="local",
        llm_api_key="fake-key",
        llm_base_url="https://provider.invalid/v1",
        llm_model="fake-model",
        llm_api_style=api_style,
    )


def _retrieval_outcome() -> RetrievalOutcome:
    chunk = DocumentChunk(
        chunk_id="attention::0",
        source_path="docs/attention.md",
        title="Attention",
        text="Attention uses queries, keys, and values.",
        chunk_index=0,
        start_char=0,
        end_char=41,
        embedding=[1.0],
    )
    return RetrievalOutcome(
        hits=[build_retrieval_hit(chunk, 0.95)],
        embedding_status=ProviderStatus(provider="fake-embedding", mode="live"),
        reranker_status=ProviderStatus(
            provider="none", mode="ready", reason="reranker_disabled"
        ),
    )


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
    monkeypatch.setattr(agents_sdk, "build_async_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(
        agents_sdk,
        "retrieve",
        lambda config, query, top_k: _retrieval_outcome(),
    )

    config = _config(api_style)
    answer = agents_sdk.answer_with_agents_sdk(config, "Explain attention")

    assert answer.answer == "Attention uses queries and keys. [S1]"
    assert answer.citations == ["docs/attention.md"]
    assert [span.chunk_id for span in answer.citation_spans] == ["attention::0"]
    assert [hit.chunk.chunk_id for hit in answer.retrieved_chunks] == ["attention::0"]
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
    monkeypatch.setattr(agents_sdk, "build_async_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(agents_sdk, "answer_with_basic_runtime", fake_basic_runtime)

    answer = agents_sdk.answer_with_agents_sdk(_config("responses"), "question")

    assert answer is fallback_answer
    assert client.closed is True
    assert fallback_call["requested_runtime"] == "agents_sdk"
    assert fallback_call["runtime_reason"] == "runtime_fallback:agents_sdk_error:RuntimeError"
