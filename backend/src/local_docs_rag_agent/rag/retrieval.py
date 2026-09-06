"""Ranks a chunk corpus against a query under a named retrieval strategy.

Ranking used to be one hard-coded expression inside the local store, which made
it impossible to change without changing the store and impossible to compare
against anything. Pulling it out here gives the project a retrieval *strategy*
it can select by configuration and — the point of the exercise — sweep with the
eval matrix, so a claim that ranking improved is something the repository can
measure rather than assert.

Four strategies share one entry point:

* ``blended`` — the original ``max(dense, lexical, weighted mix)``. Kept as the
  default and as the baseline every other strategy is measured against, so
  existing numbers stay reproducible.
* ``dense`` — embedding cosine similarity alone.
* ``lexical`` — BM25 alone, with the inverse document frequency the old lexical
  branch lacked.
* ``hybrid_rrf`` — dense and BM25 ranked independently, then fused by reciprocal
  rank.

Scores are only comparable *within* a strategy. An RRF score is a sum of
reciprocal ranks and lives around 0.03; a cosine similarity lives near 1.
Nothing downstream thresholds on the absolute value, and nothing should start
to.

``candidate_k`` is how deep each branch ranks before fusion, and it is
deliberately larger than ``top_k``: a chunk that one signal puts 12th and the
other puts 3rd deserves to surface, and it cannot if each branch only ever
offers its own best four.
"""

from __future__ import annotations

import dataclasses

from local_docs_rag_agent import constants, models
from local_docs_rag_agent.rag import bm25, fusion, scoring


@dataclasses.dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """The ranking knobs a store needs, without the rest of `AppConfig`.

    Passing this rather than the whole configuration keeps the stores unaware of
    providers, paths, and runtimes they have no business reading.
    """

    strategy: constants.RetrievalStrategyName = (
        constants.DEFAULT_RETRIEVAL_STRATEGY
    )
    candidate_k: int = constants.DEFAULT_RETRIEVAL_CANDIDATE_K
    rrf_k: int = constants.DEFAULT_RRF_K

    def window(self, top_k: int) -> int:
        """Return the candidate depth to rank to before selecting `top_k`.

        The eval matrix sweeps `top_k` independently of the window, so a sweep
        can ask for more results than the configured window holds. Widening to
        fit is the only sensible reading of that: a window narrower than the
        result set would silently truncate the sweep it was meant to measure.
        """

        return max(self.candidate_k, top_k)


def rank_chunks(
    *,
    query: str,
    query_embedding: list[float],
    chunks: list[models.DocumentChunk],
    top_k: int,
    settings: RetrievalSettings,
) -> list[models.RetrievalHit]:
    """Return at most `top_k` hits for `query`, best first, under
    `settings.strategy`.
    """

    if top_k <= 0 or not chunks:
        return []

    if settings.strategy == "blended":
        scored = [
            (
                index,
                scoring.score_local_chunk(
                    query_embedding=query_embedding,
                    query_terms=scoring.tokenize(query),
                    chunk=chunks[index],
                ),
            )
            for index in range(len(chunks))
        ]
        return _hits_from_scores(chunks, scored, top_k)

    if settings.strategy == "dense":
        return _hits_from_scores(
            chunks, _dense_ranking(query_embedding, chunks), top_k
        )

    if settings.strategy == "lexical":
        return _hits_from_scores(
            chunks, bm25.Bm25Index.build(_texts(chunks)).rank(query), top_k
        )

    return _fuse_rankings(
        chunks=chunks,
        dense=_dense_ranking(query_embedding, chunks)[: settings.window(top_k)],
        lexical=bm25.Bm25Index.build(_texts(chunks)).rank(query)[
            : settings.window(top_k)
        ],
        top_k=top_k,
        rrf_k=settings.rrf_k,
    )


def rerank_dense_hits(
    *,
    query: str,
    dense_hits: list[models.RetrievalHit],
    top_k: int,
    settings: RetrievalSettings,
) -> list[models.RetrievalHit]:
    """Re-rank hits a remote backend already ordered by vector similarity.

    Qdrant returns payloads without vectors, so the strategies that need an
    embedding cannot be recomputed here. What is available is the server's own
    ordering, which is exactly the dense ranking RRF wants, plus the chunk text
    BM25 needs. The lexical statistics are therefore drawn from the candidate
    window rather than the whole collection — narrower than the local store's
    corpus-wide frequencies, and the reason `candidate_k` should stay
    comfortably wider than `top_k`.
    """

    if top_k <= 0 or not dense_hits:
        return []

    chunks = [hit.chunk for hit in dense_hits]
    lexical = bm25.Bm25Index.build(_texts(chunks)).rank(query)
    if settings.strategy == "lexical":
        return _hits_from_scores(chunks, lexical, top_k)

    return _fuse_rankings(
        chunks=chunks,
        dense=[(index, hit.score) for index, hit in enumerate(dense_hits)],
        lexical=lexical[: settings.window(top_k)],
        top_k=top_k,
        rrf_k=settings.rrf_k,
    )


def needs_candidate_window(strategy: constants.RetrievalStrategyName) -> bool:
    """Report whether `strategy` ranks beyond `top_k` before choosing its results.

    A remote backend uses this to decide how many candidates to ask the server
    for: the dense-only paths can request exactly `top_k`, while the fused paths
    need a wider window to fuse over.
    """

    return strategy in ("lexical", "hybrid_rrf")


def _dense_ranking(
    query_embedding: list[float], chunks: list[models.DocumentChunk]
) -> list[tuple[int, float]]:
    scored = [
        (
            index,
            scoring.cosine_similarity(query_embedding, chunk.embedding or []),
        )
        for index, chunk in enumerate(chunks)
    ]
    return sorted(
        (entry for entry in scored if entry[1] > 0),
        key=lambda entry: (-entry[1], entry[0]),
    )


def _fuse_rankings(
    *,
    chunks: list[models.DocumentChunk],
    dense: list[tuple[int, float]],
    lexical: list[tuple[int, float]],
    top_k: int,
    rrf_k: int,
) -> list[models.RetrievalHit]:
    # Fusion is keyed by chunk id rather than list position so the ranking
    # survives being read back from a report, where the original list order is
    # gone.
    by_chunk_id = {chunk.chunk_id: chunk for chunk in chunks}
    fused = fusion.reciprocal_rank_fusion(
        [
            [chunks[index].chunk_id for index, _ in dense],
            [chunks[index].chunk_id for index, _ in lexical],
        ],
        rrf_k=rrf_k,
    )
    return [
        scoring.build_retrieval_hit(by_chunk_id[chunk_id], score)
        for chunk_id, score in fused[:top_k]
    ]


def _hits_from_scores(
    chunks: list[models.DocumentChunk],
    scored: list[tuple[int, float]],
    top_k: int,
) -> list[models.RetrievalHit]:
    ranked = sorted(
        (entry for entry in scored if entry[1] > 0),
        key=lambda entry: (-entry[1], entry[0]),
    )
    return [
        scoring.build_retrieval_hit(chunks[index], score)
        for index, score in ranked[:top_k]
    ]


def _texts(chunks: list[models.DocumentChunk]) -> list[str]:
    # Title and section metadata join the body so a heading-only match still
    # ranks, which is what the old lexical branch used its separate metadata
    # score for.
    return [
        " ".join(
            part
            for part in (
                chunk.title,
                str(chunk.metadata.get("section_title") or ""),
                chunk.text,
            )
            if part
        )
        for chunk in chunks
    ]
