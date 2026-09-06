"""Split documents into retrievable chunks that keep their spans.

Three strategies share one entry point, :func:`chunk_text`: `fixed` slices by
size, `paragraph` groups blank-line-separated blocks, and `markdown`
additionally refuses to merge blocks across a heading so a chunk never mixes two
sections. Every chunk records the character span it came from, which is what
makes the citation spans in an answer point back at real source text.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Iterator

from local_docs_rag_agent import constants, exceptions, models


@dataclasses.dataclass(slots=True)
class Section:
    """One heading-delimited region of a document, with its offsets."""

    title: str
    heading_level: int | None
    start_char: int
    end_char: int
    text: str


@dataclasses.dataclass(slots=True)
class ParagraphBlock:
    """One paragraph, with its offsets and the heading it sits under."""

    text: str
    start_char: int
    end_char: int
    section_title: str
    heading_level: int | None


def chunk_text(
    source_path: pathlib.Path,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_strategy: str = constants.DEFAULT_CHUNK_STRATEGY,
) -> list[models.DocumentChunk]:
    """Split one document into chunks under the named strategy.

    Args:
      source_path: Path of the document, used for chunk ids and title.
      text: The document's full text.
      chunk_size: Maximum characters per chunk.
      chunk_overlap: Characters each chunk repeats from the previous
        one, so a fact spanning a boundary is still retrievable.
      chunk_strategy: `fixed`, `paragraph`, or `markdown`.

    Returns:
      The chunks, each carrying the character span it came from so a
      citation can point back at the source exactly. Empty when the
      document holds no non-whitespace text.

    Raises:
      ConfigurationError: If the chunk size or overlap is invalid, or
        the strategy is not one of the three supported names.
    """
    _validate_chunk_parameters(chunk_size, chunk_overlap)
    if not text.strip():
        return []

    strategy = _normalize_strategy(chunk_strategy)
    source_title = (
        source_path.stem.replace("_", " ").strip() or source_path.name
    )
    if strategy == "fixed":
        return _chunk_fixed(
            source_path, text, source_title, chunk_size, chunk_overlap
        )
    if strategy == "paragraph":
        return _chunk_paragraphs(
            source_path,
            text,
            source_title,
            chunk_size,
            chunk_overlap,
            markdown_aware=False,
        )
    return _chunk_paragraphs(
        source_path,
        text,
        source_title,
        chunk_size,
        chunk_overlap,
        markdown_aware=True,
    )


def _normalize_strategy(
    chunk_strategy: str | None,
) -> constants.ChunkStrategyName:
    if not chunk_strategy:
        return constants.DEFAULT_CHUNK_STRATEGY
    normalized = chunk_strategy.strip().lower()
    if normalized in constants.CHUNK_STRATEGIES:
        return normalized
    allowed = ", ".join(constants.CHUNK_STRATEGIES)
    raise exceptions.ConfigurationError(
        f"Chunk strategy must be one of {allowed}; got {chunk_strategy!r}"
    )


def _validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise exceptions.ConfigurationError(
            f"Chunk size must be greater than 0, got {chunk_size}"
        )
    if chunk_overlap < 0:
        raise exceptions.ConfigurationError(
            f"Chunk overlap must be at least 0, got {chunk_overlap}"
        )
    if chunk_overlap >= chunk_size:
        raise exceptions.ConfigurationError(
            "Chunk overlap must be smaller than chunk size "
            f"(overlap={chunk_overlap}, size={chunk_size})"
        )


def _chunk_fixed(
    source_path: pathlib.Path,
    text: str,
    source_title: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[models.DocumentChunk]:
    sections = _extract_sections(text, source_title)
    chunks: list[models.DocumentChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_body, content_start, content_end = _strip_span(text, start, end)
        if chunk_body:
            section = _section_for_offset(sections, content_start)
            title = section.title if section else source_title
            metadata = _build_metadata(
                strategy="fixed",
                source_title=source_title,
                section=section,
                start_char=content_start,
                end_char=content_end,
            )
            chunks.append(
                _make_chunk(
                    source_path=source_path,
                    title=title,
                    text=chunk_body,
                    chunk_index=chunk_index,
                    start_char=content_start,
                    end_char=content_end,
                    metadata=metadata,
                )
            )
            chunk_index += 1

        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)

    return chunks


def _chunk_paragraphs(
    source_path: pathlib.Path,
    text: str,
    source_title: str,
    chunk_size: int,
    chunk_overlap: int,
    markdown_aware: bool,
) -> list[models.DocumentChunk]:
    """Group paragraph blocks into chunks, overlapping at the boundaries.

    The work is three separate decisions, and they are kept separate:
    :func:`_group_stop` says which blocks belong together,
    :func:`_chunk_from_blocks` turns that run into a chunk, and
    :func:`_next_group_start` says where the following chunk begins.

    Args:
      source_path: Path of the document, used for chunk ids and titles.
      text: The document's full text.
      source_title: Fallback title for a chunk under no heading.
      chunk_size: Maximum characters per chunk.
      chunk_overlap: Characters each chunk repeats from the previous one.
      markdown_aware: Whether a heading ends a chunk.

    Returns:
      The chunks, in document order. A document with no paragraph blocks
      falls back to fixed-size slicing rather than returning nothing.
    """
    blocks = _extract_paragraph_blocks(
        text, source_title, markdown_aware=markdown_aware
    )
    if not blocks:
        return _chunk_fixed(
            source_path, text, source_title, chunk_size, chunk_overlap
        )
    blocks = _split_oversized_blocks(
        blocks,
        source_text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[models.DocumentChunk] = []
    strategy = "markdown" if markdown_aware else "paragraph"
    start = 0

    while start < len(blocks):
        stop = _group_stop(
            blocks,
            start=start,
            chunk_size=chunk_size,
            markdown_aware=markdown_aware,
        )
        chunk = _chunk_from_blocks(
            blocks[start:stop],
            source_path=source_path,
            source_title=source_title,
            strategy=strategy,
            chunk_index=len(chunks),
        )
        if chunk is None:
            # A run of blank blocks produces no chunk; step over its first
            # block rather than grouping the same run again.
            start += 1
            continue

        chunks.append(chunk)
        if stop >= len(blocks):
            break
        start = _next_group_start(
            blocks, start=start, stop=stop, chunk_overlap=chunk_overlap
        )

    return chunks


def _group_stop(
    blocks: list[ParagraphBlock],
    *,
    start: int,
    chunk_size: int,
    markdown_aware: bool,
) -> int:
    """Return one past the last block belonging with `blocks[start]`.

    Args:
      blocks: Every block of the document, in order.
      start: Index of the block the group begins at.
      chunk_size: Maximum characters a chunk should hold.
      markdown_aware: Whether a heading ends the group, so a chunk never
        mixes two sections.

    Returns:
      The exclusive end of the group, always greater than `start`: the first
      block is taken unconditionally, because a block too large to fit alone
      has already been split by `_split_oversized_blocks` and a group of
      nothing would not advance.
    """
    parts: list[str] = []
    section = blocks[start].section_title
    stop = start

    for index in range(start, len(blocks)):
        block = blocks[index]
        candidate = "\n\n".join([*parts, block.text]).strip()
        leaves_section = (
            markdown_aware and index > start and block.section_title != section
        )
        if parts and candidate and len(candidate) > chunk_size:
            break
        if parts and leaves_section:
            break

        parts.append(block.text)
        section = block.section_title
        stop = index + 1

        if len(candidate) >= chunk_size:
            break

    return stop


def _chunk_from_blocks(
    group: list[ParagraphBlock],
    *,
    source_path: pathlib.Path,
    source_title: str,
    strategy: str,
    chunk_index: int,
) -> models.DocumentChunk | None:
    """Build one chunk from a run of consecutive blocks.

    Args:
      group: The blocks to join, in order.
      source_path: Path of the document, used for the chunk id.
      source_title: Fallback title when the group sits under no heading.
      strategy: Chunk strategy recorded in the metadata.
      chunk_index: Position of this chunk within the document.

    Returns:
      The chunk, spanning the first block's start to the last block's end, or
      None when the group holds no text at all. The section title and heading
      level come from the last block, which is the one a reader of the chunk's
      final lines is under.
    """
    text = "\n\n".join(block.text for block in group).strip()
    if not text:
        return None

    first, last = group[0], group[-1]
    section = Section(
        title=last.section_title,
        heading_level=last.heading_level,
        start_char=first.start_char,
        end_char=last.end_char,
        text=text,
    )
    return _make_chunk(
        source_path=source_path,
        title=last.section_title or source_title,
        text=text,
        chunk_index=chunk_index,
        start_char=first.start_char,
        end_char=last.end_char,
        metadata=_build_metadata(
            strategy=strategy,
            source_title=source_title,
            section=section,
            start_char=first.start_char,
            end_char=last.end_char,
        ),
    )


def _next_group_start(
    blocks: list[ParagraphBlock],
    *,
    start: int,
    stop: int,
    chunk_overlap: int,
) -> int:
    """Return the block index the next chunk begins at.

    Args:
      blocks: Every block of the document, in order.
      start: Where the chunk just emitted began.
      stop: One past where it ended.
      chunk_overlap: Characters the next chunk should repeat.

    Returns:
      The earliest trailing block whose text, together with the blocks after
      it, covers `chunk_overlap` characters, so a fact spanning a boundary
      stays retrievable from both sides. Never earlier than `start + 1`: a
      block longer than the overlap must still advance, or the same group
      would be emitted forever.
    """
    repeated = 0
    for index in range(stop - 1, start, -1):
        repeated += len(blocks[index].text)
        if repeated >= chunk_overlap:
            return max(start + 1, index)
    return start + 1


def _make_chunk(
    source_path: pathlib.Path,
    title: str,
    text: str,
    chunk_index: int,
    start_char: int,
    end_char: int,
    metadata: dict[str, object],
) -> models.DocumentChunk:
    return models.DocumentChunk(
        chunk_id=f"{source_path.as_posix()}::chunk-{chunk_index}",
        source_path=source_path.as_posix(),
        title=title,
        text=text,
        chunk_index=chunk_index,
        start_char=start_char,
        end_char=end_char,
        metadata=metadata,
    )


def _extract_sections(text: str, source_title: str) -> list[Section]:
    sections: list[Section] = []
    active_title = source_title
    active_level: int | None = None
    section_start = 0

    for line_start, line_end, line in _iter_lines_with_offsets(text):
        heading_level, heading_title = _parse_markdown_heading(line)
        if heading_level is None:
            continue
        if line_start > section_start:
            body = text[section_start:line_start].strip()
            if body:
                sections.append(
                    Section(
                        title=active_title,
                        heading_level=active_level,
                        start_char=section_start,
                        end_char=line_start,
                        text=body,
                    )
                )
        active_title = heading_title
        active_level = heading_level
        section_start = line_end

    tail = text[section_start:].strip()
    if tail:
        sections.append(
            Section(
                title=active_title,
                heading_level=active_level,
                start_char=section_start,
                end_char=len(text),
                text=tail,
            )
        )

    if sections:
        return sections
    return [
        Section(
            title=source_title,
            heading_level=None,
            start_char=0,
            end_char=len(text),
            text=text,
        )
    ]


def _extract_paragraph_blocks(
    text: str, source_title: str, markdown_aware: bool
) -> list[ParagraphBlock]:
    blocks: list[ParagraphBlock] = []
    section_title = source_title
    heading_level: int | None = None
    paragraph_lines: list[str] = []
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    offset = 0

    for line in text.splitlines(keepends=True):
        raw_line = line
        stripped = raw_line.strip()
        heading_match = (
            _parse_markdown_heading(raw_line)
            if markdown_aware
            else (None, None)
        )
        if heading_match[0] is not None:
            _flush_paragraph(
                blocks,
                paragraph_lines,
                paragraph_start,
                paragraph_end,
                section_title,
                heading_level,
            )
            paragraph_lines = []
            paragraph_start = None
            paragraph_end = None
            heading_level, section_title = heading_match
            offset += len(raw_line)
            continue

        if stripped:
            if paragraph_start is None:
                paragraph_start = offset
            paragraph_lines.append(raw_line.rstrip("\n"))
            paragraph_end = offset + len(raw_line.rstrip("\r\n"))
        else:
            _flush_paragraph(
                blocks,
                paragraph_lines,
                paragraph_start,
                paragraph_end,
                section_title,
                heading_level,
            )
            paragraph_lines = []
            paragraph_start = None
            paragraph_end = None

        offset += len(raw_line)

    _flush_paragraph(
        blocks,
        paragraph_lines,
        paragraph_start,
        paragraph_end,
        section_title,
        heading_level,
    )
    return blocks


def _flush_paragraph(
    blocks: list[ParagraphBlock],
    paragraph_lines: list[str],
    paragraph_start: int | None,
    paragraph_end: int | None,
    section_title: str,
    heading_level: int | None,
) -> None:
    if not paragraph_lines or paragraph_start is None or paragraph_end is None:
        return
    text = "\n".join(line.rstrip() for line in paragraph_lines).strip()
    if not text:
        return
    blocks.append(
        ParagraphBlock(
            text=text,
            start_char=paragraph_start,
            end_char=paragraph_end,
            section_title=section_title,
            heading_level=heading_level,
        )
    )


def _split_oversized_blocks(
    blocks: list[ParagraphBlock],
    *,
    source_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ParagraphBlock]:
    normalized: list[ParagraphBlock] = []
    for block in blocks:
        if len(block.text) <= chunk_size:
            normalized.append(block)
            continue

        start = block.start_char
        while start < block.end_char:
            end = min(block.end_char, start + chunk_size)
            body, content_start, content_end = _strip_span(
                source_text, start, end
            )
            if body:
                normalized.append(
                    ParagraphBlock(
                        text=body,
                        start_char=content_start,
                        end_char=content_end,
                        section_title=block.section_title,
                        heading_level=block.heading_level,
                    )
                )
            if end >= block.end_char:
                break
            start = end - chunk_overlap
    return normalized


def _iter_lines_with_offsets(text: str) -> Iterator[tuple[int, int, str]]:
    offset = 0
    for line in text.splitlines(keepends=True):
        start = offset
        offset += len(line)
        yield start, offset, line.rstrip("\n")


def _parse_markdown_heading(line: str) -> tuple[int | None, str]:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None, ""
    marker = stripped.split(" ", 1)[0]
    if not marker or any(char != "#" for char in marker):
        return None, ""
    heading = stripped[len(marker) :].strip()
    if not heading:
        return None, ""
    return len(marker), heading


def _section_for_offset(sections: list[Section], offset: int) -> Section | None:
    for section in sections:
        if section.start_char <= offset < section.end_char:
            return section
    return sections[-1] if sections else None


def _build_metadata(
    strategy: str,
    source_title: str,
    section: Section | None,
    start_char: int,
    end_char: int,
) -> dict[str, object]:
    return {
        "start": start_char,
        "end": end_char,
        "chunk_strategy": strategy,
        "source_title": source_title,
        "section_title": section.title if section else source_title,
        "heading_level": section.heading_level if section else None,
    }


def _strip_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    raw = text[start:end]
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw) - len(raw.rstrip())
    content_start = start + leading
    content_end = end - trailing
    return text[content_start:content_end], content_start, content_end
