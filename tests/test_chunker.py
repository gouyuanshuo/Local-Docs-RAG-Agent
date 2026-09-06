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


def _block(
    text: str, section: str = "Intro", level: int | None = 2, start: int = 0
) -> chunker.ParagraphBlock:
    return chunker.ParagraphBlock(
        text=text,
        start_char=start,
        end_char=start + len(text),
        section_title=section,
        heading_level=level,
    )


# --- Grouping: which blocks belong together ----------------------------------


def test_group_stops_once_the_blocks_exceed_the_chunk_size() -> None:
    blocks = [_block("a" * 40), _block("b" * 40), _block("c" * 40)]

    stop = chunker._group_stop(
        blocks, start=0, chunk_size=50, markdown_aware=False
    )

    assert stop == 1


def test_group_takes_one_block_even_when_it_alone_is_oversized() -> None:
    # Oversized blocks are already split upstream, but a group of nothing
    # would never advance, so the first block is taken unconditionally.
    blocks = [_block("a" * 500), _block("b" * 10)]

    stop = chunker._group_stop(
        blocks, start=0, chunk_size=50, markdown_aware=False
    )

    assert stop == 1


def test_markdown_grouping_refuses_to_cross_a_heading() -> None:
    blocks = [
        _block("alpha", section="First"),
        _block("beta", section="Second"),
    ]

    assert (
        chunker._group_stop(
            blocks, start=0, chunk_size=1000, markdown_aware=True
        )
        == 1
    )


def test_paragraph_grouping_merges_across_headings() -> None:
    # Without markdown awareness a heading is just more text, so blocks merge
    # until the size limit. This is what separates the two strategies.
    blocks = [
        _block("alpha", section="First"),
        _block("beta", section="Second"),
    ]

    assert (
        chunker._group_stop(
            blocks, start=0, chunk_size=1000, markdown_aware=False
        )
        == 2
    )


# --- Advancing: where the next chunk starts ----------------------------------


def test_next_group_always_advances_by_at_least_one_block() -> None:
    # A block longer than the overlap must still move the cursor forward, or
    # the same group would be emitted forever.
    blocks = [_block("a" * 500), _block("b" * 500)]

    assert (
        chunker._next_group_start(blocks, start=0, stop=2, chunk_overlap=400)
        == 1
    )


def test_next_group_repeats_enough_blocks_to_cover_the_overlap() -> None:
    blocks = [_block("a" * 10) for _ in range(5)]

    # Two trailing blocks of ten characters cover an overlap of twenty.
    assert (
        chunker._next_group_start(blocks, start=0, stop=5, chunk_overlap=20)
        == 3
    )


def test_next_group_never_rewinds_before_the_current_group() -> None:
    blocks = [_block("a" * 5) for _ in range(4)]

    assert (
        chunker._next_group_start(blocks, start=2, stop=4, chunk_overlap=1000)
        == 3
    )


# --- End to end ---------------------------------------------------------------


def test_markdown_chunks_never_mix_two_sections() -> None:
    source_text = (
        "# Title\n\n## Alpha\n\nAlpha body text.\n\n"
        "## Beta\n\nBeta body text.\n"
    )

    chunks = chunker.chunk_text(
        source_path=pathlib.Path("doc.md"),
        text=source_text,
        chunk_size=1000,
        chunk_overlap=0,
        chunk_strategy="markdown",
    )

    assert all(
        not ("Alpha body" in chunk.text and "Beta body" in chunk.text)
        for chunk in chunks
    )


def test_paragraph_chunk_offsets_refer_to_original_text() -> None:
    source_text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n"

    chunks = chunker.chunk_text(
        source_path=pathlib.Path("doc.md"),
        text=source_text,
        chunk_size=25,
        chunk_overlap=5,
        chunk_strategy="paragraph",
    )

    assert chunks
    for chunk in chunks:
        span = source_text[chunk.start_char : chunk.end_char]
        assert chunk.text in span
