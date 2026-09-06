import json
import pathlib

import pytest

from local_docs_rag_agent import agent, exceptions, models, presenters
from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.evals import harness


def test_empty_expectations_are_neutral_success() -> None:
    assert harness.keyword_match_rate([], "anything") == 1.0
    assert harness.source_match_rate([], []) == 1.0


def test_eval_loader_reports_malformed_line(tmp_path: pathlib.Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        "\n".join(
            [
                json.dumps({"question": "valid"}),
                json.dumps(
                    {
                        "question": "invalid",
                        "expected_source_paths": "not-a-list",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(exceptions.DataFormatError, match="line 2"):
        harness.load_eval_cases(eval_path)


def test_eval_result_preserves_runtime_and_provider_diagnostics(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"question": "What is attention?"}), encoding="utf-8"
    )
    diagnostics = models.AnswerDiagnostics(
        requested_runtime="basic",
        actual_runtime="basic",
        vector_backend="local",
        chat_provider=models.ProviderStatus(provider="fake-chat", mode="live"),
        embedding_provider=models.ProviderStatus(
            provider="fake-embedding", mode="live"
        ),
        reranker=models.ProviderStatus(
            provider="none", mode="ready", reason="reranker_disabled"
        ),
    )

    class FakeAgent:
        def __init__(self, config: app_config.AppConfig) -> None:
            del config

        def answer(self, question: str) -> models.AgentAnswer:
            return models.AgentAnswer(
                question=question,
                answer="Attention uses queries and keys.",
                citations=[],
                citation_spans=[],
                retrieved_chunks=[],
                diagnostics=diagnostics,
            )

    monkeypatch.setattr(agent, "LocalDocsAgent", FakeAgent)
    config = app_config.AppConfig.from_env().with_overrides(
        eval_path=eval_path,
        vector_backend="local",
    )

    results = harness.run_eval(config)
    summary = presenters.serialize_eval_summary(
        results, runtime="basic", config=config
    )

    assert results[0].diagnostics == diagnostics
    # `serialize_eval_summary` returns an untyped payload, so narrow before
    # indexing.
    results_payload = summary["results"]
    assert isinstance(results_payload, list)
    assert results_payload[0]["diagnostics"] == {
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
        "reranker": {
            "provider": "none",
            "mode": "ready",
            "reason": "reranker_disabled",
        },
    }
