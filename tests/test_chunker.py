import pathlib

import pytest

from local_docs_rag_agent import exceptions
from local_docs_rag_agent.rag import chunker


def test_fixed_chunk_offsets_refer_to_original_text() -> None:
    source_text = "   leading text followed by more content   "

    chunks = chunker.chunk_text(
        source_path=pathlib.Path("sample.txt"),
        text=source_text,
        chunk_size=20,
        chunk_overlap=5,
        chunk_strategy="fixed",
    )

    assert chunks[0].start_char == 3
    assert all(
        chunk.text == source_text[chunk.start_char : chunk.end_char]
        for chunk in chunks
    )


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_chunk_parameters_are_rejected(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(exceptions.ConfigurationError):
        chunker.chunk_text(
            source_path=pathlib.Path("sample.txt"),
            text="content",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_unknown_chunk_strategy_is_rejected() -> None:
    with pytest.raises(exceptions.ConfigurationError, match="strategy"):
        chunker.chunk_text(
            source_path=pathlib.Path("sample.txt"),
            text="content",
            chunk_size=10,
            chunk_overlap=2,
            chunk_strategy="typo",
        )


def test_long_paragraph_is_split_to_configured_size() -> None:
    chunks = chunker.chunk_text(
        source_path=pathlib.Path("sample.txt"),
        text="a" * 95,
        chunk_size=30,
        chunk_overlap=5,
        chunk_strategy="paragraph",
    )

    assert len(chunks) == 4
    assert all(len(chunk.text) <= 30 for chunk in chunks)
    assert chunks[1].start_char == 25
