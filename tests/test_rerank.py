from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import OpenAI

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.exceptions import ConfigurationError
from local_docs_rag_agent.models import DocumentChunk, ProviderStatus, RetrievalHit
from local_docs_rag_agent.rag import pipeline
from local_docs_rag_agent.rag.llm_rerank import LlmReranker
from local_docs_rag_agent.rag.pipeline import build_reranker, retrieve
from local_docs_rag_agent.rag.rerank import IdentityReranker
from local_docs_rag_agent.rag.scoring import build_retrieval_hit


class FakeEndpoint:
    """Stands in for `client.responses` or `client.chat.completions`."""

    def __init__(self, reply: str | None = None, error: Exception | None = None) -> None:
        self._reply = reply
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            output_text=self._reply,
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._reply))],
        )


class FakeStore:
    """Returns a fixed corpus slice and records the depth it was asked for."""

    def __init__(self, hits: list[RetrievalHit], mode: str = "live") -> None:
        self._hits = hits
        self._mode = mode
        self.requested_top_k: int | None = None

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        del query
        self.requested_top_k = top_k
        return self._hits[:top_k]

    @property
    def embedding_status(self) -> ProviderStatus:
        return ProviderStatus(provider="fake-embedding", mode="live")


def _hit(chunk_id: str, score: float, text: str = "Attention explained.") -> RetrievalHit:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        source_path=f"{chunk_id}.md",
        title=chunk_id,
        text=text,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
    )
    return build_retrieval_hit(chunk, score)


def _hits() -> list[RetrievalHit]:
    return [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)]


def _ids(hits: list[RetrievalHit]) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


def _reranker(
    reply: str | None = None,
    error: Exception | None = None,
    api_style: str = "responses",
    candidate_k: int = 20,
) -> tuple[LlmReranker, FakeEndpoint]:
    reranker = LlmReranker(
        api_key="fake-key",
        model="fake-model",
        candidate_k=candidate_k,
        api_style=api_style,
    )
    endpoint = FakeEndpoint(reply=reply, error=error)
    client = SimpleNamespace(
        responses=endpoint,
        chat=SimpleNamespace(completions=endpoint),
    )
    # The provider tests use the same shape: a fake endpoint installed in place of the
    # real client, so the ranking logic is exercised without a network call.
    reranker._client = cast(OpenAI, client)
    return reranker, endpoint


# --- The disabled reranker -------------------------------------------------------


def test_identity_reranker_asks_for_no_extra_candidates() -> None:
    # Turning reranking off must cost nothing: the first-stage query is exactly the
    # one the project issued before a second stage existed.
    assert IdentityReranker().candidate_depth(4) == 4


def test_identity_reranker_keeps_the_first_stage_order() -> None:
    reranker = IdentityReranker()

    reranked = reranker.rerank(query="attention", hits=_hits(), top_k=2)

    assert _ids(reranked) == ["a", "b"]
    assert reranker.status.mode == "ready"
    assert reranker.status.reason == "reranker_disabled"


# --- Candidate depth -------------------------------------------------------------


def test_llm_reranker_reads_a_wider_window_than_it_returns() -> None:
    reranker, _ = _reranker(candidate_k=20)

    assert reranker.candidate_depth(4) == 20


def test_candidate_depth_never_narrows_below_the_requested_top_k() -> None:
    # A window smaller than the answer would drop results before the reranker saw them.
    reranker, _ = _reranker(candidate_k=5)

    assert reranker.candidate_depth(12) == 12


# --- Ordering ---------------------------------------------------------------------


def test_llm_rerank_applies_the_model_order_and_scores_by_rank() -> None:
    reranker, endpoint = _reranker(reply="[2, 3]")

    reranked = reranker.rerank(query="attention", hits=_hits(), top_k=2)

    assert _ids(reranked) == ["b", "c"]
    # The score is `1 / position`, not a similarity: the model returned an ordering and
    # no calibrated relevance, so a sorted-by-score reader still sees the rerank order.
    assert [hit.score for hit in reranked] == [1.0, 0.5]
    assert reranker.status.mode == "live"
    assert len(endpoint.calls) == 1


def test_a_partial_ranking_keeps_the_rest_in_first_stage_order() -> None:
    # A model that names only what it considers relevant must not shrink the window.
    reranker, _ = _reranker(reply="[3]")

    reranked = reranker.rerank(query="attention", hits=_hits(), top_k=3)

    assert _ids(reranked) == ["c", "a", "b"]


def test_unusable_candidate_numbers_are_dropped_rather_than_trusted() -> None:
    # Out of range, repeated, boolean, and zero all name no candidate; a quoted number
    # does, and prose around the array does not stop it being read.
    reranker, _ = _reranker(reply='Best first: [5, 2, 2, "1", true, 0]')

    reranked = reranker.rerank(query="attention", hits=_hits(), top_k=3)

    assert _ids(reranked) == ["b", "a", "c"]
    assert reranker.status.mode == "live"


def test_rerank_honours_the_configured_chat_completions_api_style() -> None:
    reranker, endpoint = _reranker(reply="[2]", api_style="chat_completions")

    reranked = reranker.rerank(query="attention", hits=_hits(), top_k=1)

    assert _ids(reranked) == ["b"]
    assert "messages" in endpoint.calls[0]


# --- Degradation ------------------------------------------------------------------


def test_rerank_without_a_key_returns_the_candidates_and_reports_fallback() -> None:
    reranker = LlmReranker(api_key=None, model="fake-model", candidate_k=20)

    reranked = reranker.rerank(query="attention", hits=_hits(), top_k=2)

    assert _ids(reranked) == ["a", "b"]
    assert reranker.status.mode == "fallback"
    assert reranker.status.reason == "missing_api_key"


def test_unusable_model_output_degrades_instead_of_inventing_an_order() -> None:
    reranker, _ = _reranker(reply="I could not decide.")

    reranked = reranker.rerank(query="attention", hits=_hits(), top_k=2)

    assert _ids(reranked) == ["a", "b"]
    # Untouched first-stage scores prove the list came back rather than being rebuilt.
    assert [hit.score for hit in reranked] == [0.9, 0.8]
    assert reranker.status.mode == "fallback"
    assert reranker.status.reason == "unusable_rerank_output"


def test_a_provider_failure_degrades_to_the_first_stage_order() -> None:
    reranker, _ = _reranker(error=RuntimeError("rerank endpoint exploded"))

    reranked = reranker.rerank(query="attention", hits=_hits(), top_k=3)

    assert _ids(reranked) == ["a", "b", "c"]
    assert reranker.status.mode == "fallback"
    assert reranker.status.reason is not None
    assert reranker.status.reason.startswith("provider_error:RuntimeError")


def test_a_single_candidate_is_not_worth_a_model_call() -> None:
    reranker, endpoint = _reranker(reply="[1]")

    reranked = reranker.rerank(query="attention", hits=[_hit("a", 0.9)], top_k=4)

    assert _ids(reranked) == ["a"]
    assert endpoint.calls == []
    assert reranker.status.mode == "ready"
    assert reranker.status.reason == "rerank_not_needed"


def test_rerank_returns_nothing_for_an_empty_window_or_zero_top_k() -> None:
    reranker, _ = _reranker(reply="[1]")

    assert reranker.rerank(query="attention", hits=[], top_k=3) == []
    assert reranker.rerank(query="attention", hits=_hits(), top_k=0) == []


# --- Pipeline composition ---------------------------------------------------------


def test_pipeline_retrieves_the_reranker_window_and_returns_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore(_hits())
    reranker, _ = _reranker(reply="[3, 1]", candidate_k=20)
    monkeypatch.setattr(pipeline, "build_store", lambda config: store)
    monkeypatch.setattr(pipeline, "build_reranker", lambda config: reranker)

    outcome = retrieve(AppConfig.from_env().with_overrides(top_k=2), "attention")

    # The store is asked for the reranker's window, not for `top_k`.
    assert store.requested_top_k == 20
    assert _ids(outcome.hits) == ["c", "a"]


def test_pipeline_reports_both_retrieval_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    # Retrieval degrades in two independent places, so an answer's diagnostics have to
    # carry both statuses or a degraded rerank would pass unnoticed.
    store = FakeStore(_hits())
    reranker, _ = _reranker(reply="nonsense")
    monkeypatch.setattr(pipeline, "build_store", lambda config: store)
    monkeypatch.setattr(pipeline, "build_reranker", lambda config: reranker)

    outcome = retrieve(AppConfig.from_env(), "attention")

    assert outcome.embedding_status.mode == "live"
    assert outcome.reranker_status.mode == "fallback"


def test_pipeline_holds_the_reranker_off_the_query_when_it_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore(_hits())
    monkeypatch.setattr(pipeline, "build_store", lambda config: store)

    outcome = retrieve(AppConfig.from_env().with_overrides(top_k=2), "attention")

    assert store.requested_top_k == 2
    assert _ids(outcome.hits) == ["a", "b"]
    assert outcome.reranker_status.reason == "reranker_disabled"


# --- Configuration ----------------------------------------------------------------


def test_reranking_is_disabled_by_default() -> None:
    config = AppConfig.from_env()

    assert config.reranker == "none"
    assert isinstance(build_reranker(config), IdentityReranker)


def test_the_llm_reranker_reuses_the_chat_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER", "llm")
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_MODEL", "answering-model")

    reranker = build_reranker(AppConfig.from_env())

    assert isinstance(reranker, LlmReranker)
    assert reranker._model == "answering-model"


def test_rerank_model_overrides_only_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # A small ranking model alongside a larger answering one is the point of the split.
    monkeypatch.setenv("RERANKER", "llm")
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_MODEL", "answering-model")
    monkeypatch.setenv("RERANK_MODEL", "ranking-model")

    reranker = build_reranker(AppConfig.from_env())

    assert isinstance(reranker, LlmReranker)
    assert reranker._model == "ranking-model"


def test_unknown_reranker_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER", "magic")

    with pytest.raises(ConfigurationError, match="RERANKER"):
        AppConfig.from_env()


def test_non_positive_rerank_candidate_k_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="RERANK_CANDIDATE_K"):
        AppConfig.from_env().with_overrides(rerank_candidate_k=0)
