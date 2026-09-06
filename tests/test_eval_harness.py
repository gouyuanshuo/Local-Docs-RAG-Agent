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


# --- Rank-aware retrieval metrics --------------------------------------------

RELEVANT = "the manifest fingerprint detects a changed document"
NOISE = ["unrelated padding one", "unrelated padding two", "padding three"]
KEYWORDS = ["manifest fingerprint"]


def test_hit_rate_cannot_see_rank_which_is_why_reciprocal_rank_exists() -> None:
    # `keyword_match_rate` joins the results before matching, so reordering the
    # same chunks is invisible to it. This is not a bug in that metric, it is
    # its shape, and it is the reason a reranker needs a different one.
    best = [RELEVANT, *NOISE]
    worst = [*NOISE, RELEVANT]

    assert harness.keyword_match_rate(
        KEYWORDS, " ".join(best)
    ) == harness.keyword_match_rate(KEYWORDS, " ".join(worst))

    assert harness.reciprocal_rank(KEYWORDS, best) == 1.0
    assert harness.reciprocal_rank(KEYWORDS, worst) == 0.25


def test_reciprocal_rank_scores_each_position() -> None:
    assert harness.reciprocal_rank(KEYWORDS, [RELEVANT, *NOISE]) == 1.0
    assert harness.reciprocal_rank(KEYWORDS, [NOISE[0], RELEVANT]) == 0.5
    assert harness.reciprocal_rank(KEYWORDS, NOISE) == 0.0


def test_reciprocal_rank_averages_over_the_expected_keywords() -> None:
    ranked = ["alpha term", "beta term"]

    # First keyword at rank 1, second at rank 2, third missing entirely.
    assert harness.reciprocal_rank(
        ["alpha", "beta", "gamma"], ranked
    ) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_empty_expectations_stay_neutral_for_the_new_metrics() -> None:
    assert harness.reciprocal_rank([], NOISE) == 1.0
    assert harness.retrieval_precision([], NOISE) == 1.0


def test_precision_falls_as_the_window_widens() -> None:
    # A wider `top_k` raises the hit rate for free; precision is what makes it
    # cost something, which is what the leaderboard needs to rank honestly.
    narrow = [RELEVANT, NOISE[0]]
    wide = [RELEVANT, *NOISE, "more padding"]

    assert harness.retrieval_precision(KEYWORDS, narrow) == 0.5
    assert harness.retrieval_precision(KEYWORDS, wide) == 0.2


def test_precision_is_zero_when_nothing_was_retrieved() -> None:
    assert harness.retrieval_precision(KEYWORDS, []) == 0.0


def test_a_low_ranked_hit_is_reported_separately_from_a_miss() -> None:
    found_but_late = models.EvalResult(
        question="q",
        answer="a",
        citations=[],
        retrieved_sources=[],
        answer_keyword_hit_rate=1.0,
        retrieval_source_hit_rate=1.0,
        retrieval_span_hit_rate=1.0,
        retrieval_reciprocal_rank=0.25,
        retrieval_precision=0.25,
        citation_source_hit_rate=1.0,
        citation_span_hit_rate=1.0,
        response_time_ms=1.0,
        expected_retrieval_keywords=["manifest fingerprint"],
    )

    reasons = harness.failure_reasons(found_but_late)

    assert reasons == ["retrieval_ranked_expected_span_below_first"]
