"""Eval gold must cite text that actually appears in the named sources."""

from __future__ import annotations

import pathlib

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import constants
from local_docs_rag_agent.evals import harness
from local_docs_rag_agent.providers import factory as provider_factory
from local_docs_rag_agent.rag import chunker, discovery, retrieval

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PATH = REPO_ROOT / "data" / "evals" / "sample_eval.jsonl"
ENGINEERING_DOC_NAMES = (
    "architecture.md",
    "development-roadmap.md",
    "code-review-2026-08-29.md",
)


def test_retrieval_keywords_are_contiguous_source_substrings() -> None:
    cases = harness.load_eval_cases(EVAL_PATH)
    assert 8 <= len(cases) <= 15
    missing: list[str] = []
    for case in cases:
        assert case.expected_source_paths, case.question
        for source_path in case.expected_source_paths:
            source_file = REPO_ROOT / source_path
            assert source_file.is_file(), source_path
            text = source_file.read_text(encoding="utf-8").lower()
            for keyword in case.expected_retrieval_keywords:
                if keyword.lower() not in text:
                    missing.append(
                        f"{case.question!r} keyword {keyword!r} "
                        f"not in {source_path}"
                    )
    assert missing == []


def test_default_docs_dir_is_the_sample_corpus_not_engineering_docs() -> None:
    config = app_config.AppConfig.from_env()
    assert config.docs_dir == pathlib.Path("docs/sample")
    paths = discovery.collect_document_paths(
        REPO_ROOT / "docs" / "sample", config.docs_exclude_patterns
    )
    names = {path.name for path in paths}
    for banned in ENGINEERING_DOC_NAMES:
        assert banned not in names
    assert "lecture5_attention.md" in names
    assert len(paths) >= 2


def test_lexical_and_dense_disagree_on_eval_set_reciprocal_rank() -> None:
    # Hash embeddings are allowed here: this is retrieval scoring, not a
    # compare leaderboard winner. Fallback cells must not enter the board.
    config = app_config.AppConfig.from_env().with_overrides(
        docs_dir=REPO_ROOT / "docs" / "sample",
        eval_path=EVAL_PATH,
        vector_backend="local",
        embedding_api_key=None,
        top_k=4,
        chunk_strategy="markdown",
        chunk_size=800,
        chunk_overlap=120,
    )
    source_texts = discovery.read_source_texts(
        config.docs_dir, config.docs_exclude_patterns
    )
    chunks = []
    for source_path, text in source_texts.items():
        chunks.extend(
            chunker.chunk_text(
                source_path=pathlib.Path(source_path),
                text=text,
                chunk_strategy=config.chunk_strategy,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        )
    provider = provider_factory.build_embedding_provider(config)
    vectors = provider.embed_texts([chunk.text for chunk in chunks])
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.embedding = vector

    cases = harness.load_eval_cases(EVAL_PATH)

    def mean_rr(strategy: constants.RetrievalStrategyName) -> float:
        total = 0.0
        for case in cases:
            query_vector = provider.embed_texts([case.question])[0]
            hits = retrieval.rank_chunks(
                query=case.question,
                query_embedding=query_vector,
                chunks=chunks,
                top_k=config.top_k,
                settings=retrieval.RetrievalSettings(strategy=strategy),
            )
            ranked = [hit.chunk.text.lower() for hit in hits]
            total += harness.reciprocal_rank(
                case.expected_retrieval_keywords, ranked
            )
        return total / len(cases)

    lexical_rr = mean_rr("lexical")
    dense_rr = mean_rr("dense")
    assert lexical_rr != dense_rr, (
        f"eval set cannot separate strategies: "
        f"lexical={lexical_rr} dense={dense_rr}"
    )
