import json
from pathlib import Path

import pytest

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.evals import harness
from local_docs_rag_agent.evals.harness import (
    keyword_match_rate,
    load_eval_cases,
    source_match_rate,
)
from local_docs_rag_agent.exceptions import DataFormatError
from local_docs_rag_agent.models import (
    AgentAnswer,
    AnswerDiagnostics,
    ProviderStatus,
)
from local_docs_rag_agent.presenters import serialize_eval_summary


def test_empty_expectations_are_neutral_success() -> None:
    assert keyword_match_rate([], "anything") == 1.0
    assert source_match_rate([], []) == 1.0


def test_eval_loader_reports_malformed_line(tmp_path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        "\n".join(
            [
                json.dumps({"question": "valid"}),
                json.dumps({"question": "invalid", "expected_source_paths": "not-a-list"}),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataFormatError, match="line 2"):
        load_eval_cases(eval_path)


def test_eval_result_preserves_runtime_and_provider_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps({"question": "What is attention?"}), encoding="utf-8")
    diagnostics = AnswerDiagnostics(
        requested_runtime="basic",
        actual_runtime="basic",
        vector_backend="local",
        chat_provider=ProviderStatus(provider="fake-chat", mode="live"),
        embedding_provider=ProviderStatus(provider="fake-embedding", mode="live"),
    )

    class FakeAgent:
        def __init__(self, config: AppConfig) -> None:
            del config

        def answer(self, question: str) -> AgentAnswer:
            return AgentAnswer(
                question=question,
                answer="Attention uses queries and keys.",
                citations=[],
                citation_spans=[],
                retrieved_chunks=[],
                diagnostics=diagnostics,
            )

    monkeypatch.setattr(harness, "LocalDocsAgent", FakeAgent)
    config = AppConfig.from_env().with_overrides(
        eval_path=eval_path,
        vector_backend="local",
    )

    results = harness.run_eval(config)
    summary = serialize_eval_summary(results, runtime="basic", config=config)

    assert results[0].diagnostics == diagnostics
    assert summary["results"][0]["diagnostics"] == {
        "requested_runtime": "basic",
        "actual_runtime": "basic",
        "vector_backend": "local",
        "chat_provider": {
            "provider": "fake-chat",
            "mode": "live",
            "reason": None,
        },
        "embedding_provider": {
            "provider": "fake-embedding",
            "mode": "live",
            "reason": None,
        },
    }
