from __future__ import annotations

import pytest

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.exceptions import ConfigurationError
from local_docs_rag_agent.models import DocumentChunk, RetrievalHit
from local_docs_rag_agent.rag.bm25 import Bm25Index
from local_docs_rag_agent.rag.fusion import reciprocal_rank_fusion
from local_docs_rag_agent.rag.retrieval import (
    RetrievalSettings,
    needs_candidate_window,
    rank_chunks,
    rerank_dense_hits,
)
from local_docs_rag_agent.rag.scoring import build_retrieval_hit
from local_docs_rag_agent.rag.store_factory import retrieval_settings


def _chunk(
    chunk_id: str,
    text: str,
    embedding: list[float] | None = None,
    title: str = "Doc",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_path=f"{chunk_id}.md",
        title=title,
        text=text,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        embedding=embedding,
    )


def _ranked_ids(hits: list[RetrievalHit]) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


# --- BM25 ----------------------------------------------------------------------


def test_bm25_weights_a_rare_term_above_a_ubiquitous_one() -> None:
    # "model" is in every document and so carries almost no information; "hyperparameter"
    # appears once. The bare set-intersection score the project used before rated both
    # documents identically, because each matched exactly one query term.
    corpus = [
        "model model model model model",
        "model hyperparameter",
        "model training",
        "model inference",
        "model evaluation",
    ]
    index = Bm25Index.build(corpus)

    ranking = index.rank("model hyperparameter")

    assert ranking[0][0] == 1


def test_bm25_saturates_repeated_terms() -> None:
    index = Bm25Index.build(["alpha", "alpha alpha alpha alpha alpha alpha alpha alpha"])

    single = index.score(["alpha"], 0)
    repeated = index.score(["alpha"], 1)

    # More occurrences still score higher, but nowhere near eight times higher.
    assert single < repeated < single * 8


def test_bm25_ignores_documents_without_any_query_term() -> None:
    index = Bm25Index.build(["attention layers", "unrelated text"])

    assert [entry[0] for entry in index.rank("attention")] == [0]


def test_bm25_rank_is_empty_for_a_query_with_no_terms() -> None:
    assert Bm25Index.build(["anything"]).rank("!!! ???") == []


# --- Reciprocal rank fusion -----------------------------------------------------


def test_rrf_prefers_agreement_over_a_single_first_place() -> None:
    fused = reciprocal_rank_fusion([["a", "c"], ["b", "c"]], rrf_k=60)

    # "c" is second in both rankings and beats the two chunks each ranked first once.
    assert fused[0][0] == "c"


def test_rrf_breaks_ties_on_the_key_so_runs_are_reproducible() -> None:
    fused = reciprocal_rank_fusion([["b", "a"], ["b", "a"]], rrf_k=60)

    assert [key for key, _ in fused] == ["b", "a"]


# --- Strategy selection ---------------------------------------------------------


def test_blended_strategy_is_the_default_and_still_uses_embeddings() -> None:
    settings = RetrievalSettings()
    assert settings.strategy == "blended"

    hits = rank_chunks(
        query="attention",
        query_embedding=[1.0, 0.0],
        chunks=[
            _chunk("dense-match", "unrelated prose", embedding=[1.0, 0.0]),
            _chunk("no-signal", "unrelated prose", embedding=[0.0, 1.0]),
        ],
        top_k=1,
        settings=settings,
    )

    assert _ranked_ids(hits) == ["dense-match"]


def test_lexical_strategy_ignores_embeddings_entirely() -> None:
    hits = rank_chunks(
        query="attention",
        query_embedding=[1.0, 0.0],
        chunks=[
            _chunk("vector-only", "unrelated prose", embedding=[1.0, 0.0]),
            _chunk("text-only", "attention explained", embedding=[0.0, 1.0]),
        ],
        top_k=1,
        settings=RetrievalSettings(strategy="lexical"),
    )

    assert _ranked_ids(hits) == ["text-only"]


def test_dense_strategy_ignores_text_entirely() -> None:
    hits = rank_chunks(
        query="attention",
        query_embedding=[1.0, 0.0],
        chunks=[
            _chunk("vector-only", "unrelated prose", embedding=[1.0, 0.0]),
            _chunk("text-only", "attention explained", embedding=[0.0, 1.0]),
        ],
        top_k=1,
        settings=RetrievalSettings(strategy="dense"),
    )

    assert _ranked_ids(hits) == ["vector-only"]


def test_hybrid_rrf_surfaces_the_chunk_both_signals_agree_on() -> None:
    # `dense_only` wins on vectors, `lexical_only` wins on words, and `agreed` is second
    # on both. Neither single-signal strategy would return it first; fusion does.
    chunks = [
        _chunk("dense_only", "entirely unrelated prose", embedding=[1.0, 0.0]),
        _chunk("agreed", "attention mechanism explained", embedding=[0.92, 0.39]),
        _chunk("lexical_only", "attention attention attention", embedding=[0.0, 1.0]),
    ]

    fused = rank_chunks(
        query="attention mechanism",
        query_embedding=[1.0, 0.0],
        chunks=chunks,
        top_k=1,
        settings=RetrievalSettings(strategy="hybrid_rrf"),
    )
    dense_only = rank_chunks(
        query="attention mechanism",
        query_embedding=[1.0, 0.0],
        chunks=chunks,
        top_k=1,
        settings=RetrievalSettings(strategy="dense"),
    )

    assert _ranked_ids(fused) == ["agreed"]
    assert _ranked_ids(dense_only) == ["dense_only"]


def test_every_strategy_returns_no_more_than_top_k() -> None:
    chunks = [_chunk(f"c{index}", "attention", embedding=[1.0]) for index in range(5)]

    for strategy in ("blended", "dense", "lexical", "hybrid_rrf"):
        hits = rank_chunks(
            query="attention",
            query_embedding=[1.0],
            chunks=chunks,
            top_k=2,
            settings=RetrievalSettings(strategy=strategy),
        )
        assert len(hits) == 2, strategy


def test_ranking_an_empty_corpus_or_zero_top_k_returns_nothing() -> None:
    settings = RetrievalSettings(strategy="hybrid_rrf")

    assert rank_chunks(
        query="q", query_embedding=[1.0], chunks=[], top_k=3, settings=settings
    ) == []
    assert rank_chunks(
        query="q",
        query_embedding=[1.0],
        chunks=[_chunk("a", "attention")],
        top_k=0,
        settings=settings,
    ) == []


# --- Candidate window -----------------------------------------------------------


def test_candidate_window_widens_to_hold_a_larger_top_k() -> None:
    # The eval matrix sweeps top_k independently of the window, so a sweep must not be
    # silently truncated by a window narrower than the result set it asked for.
    settings = RetrievalSettings(candidate_k=20)

    assert settings.window(4) == 20
    assert settings.window(32) == 32


def test_only_the_fused_strategies_need_a_candidate_window() -> None:
    assert needs_candidate_window("hybrid_rrf") is True
    assert needs_candidate_window("lexical") is True
    assert needs_candidate_window("dense") is False
    assert needs_candidate_window("blended") is False


# --- Remote re-ranking ----------------------------------------------------------


def test_rerank_dense_hits_fuses_server_order_with_local_bm25() -> None:
    # Qdrant returns payloads without vectors, so the server's ordering *is* the dense
    # ranking. Fusion still has to change the outcome, or the re-rank does nothing.
    dense_hits = [
        build_retrieval_hit(_chunk("first", "entirely unrelated prose"), 0.90),
        build_retrieval_hit(_chunk("second", "attention mechanism explained"), 0.88),
        build_retrieval_hit(_chunk("third", "attention"), 0.10),
    ]

    reranked = rerank_dense_hits(
        query="attention mechanism",
        dense_hits=dense_hits,
        top_k=1,
        settings=RetrievalSettings(strategy="hybrid_rrf"),
    )

    assert _ranked_ids(reranked) == ["second"]


def test_rerank_dense_hits_with_no_candidates_returns_nothing() -> None:
    assert (
        rerank_dense_hits(
            query="attention",
            dense_hits=[],
            top_k=3,
            settings=RetrievalSettings(strategy="hybrid_rrf"),
        )
        == []
    )


# --- Configuration --------------------------------------------------------------


def test_retrieval_settings_are_projected_from_the_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVAL_STRATEGY", "hybrid_rrf")
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", "35")
    monkeypatch.setenv("RRF_K", "12")

    settings = retrieval_settings(AppConfig.from_env())

    assert settings == RetrievalSettings(strategy="hybrid_rrf", candidate_k=35, rrf_k=12)


def test_default_configuration_keeps_the_previous_ranking_behaviour() -> None:
    config = AppConfig.from_env()

    assert config.retrieval_strategy == "blended"
    assert retrieval_settings(config) == RetrievalSettings()


def test_unknown_retrieval_strategy_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_STRATEGY", "magic")

    with pytest.raises(ConfigurationError, match="RETRIEVAL_STRATEGY"):
        AppConfig.from_env()


def test_non_positive_fusion_constants_are_rejected() -> None:
    config = AppConfig.from_env()

    with pytest.raises(ConfigurationError, match="RRF_K"):
        config.with_overrides(rrf_k=0)
    with pytest.raises(ConfigurationError, match="RETRIEVAL_CANDIDATE_K"):
        config.with_overrides(retrieval_candidate_k=0)
