"""Split source documents into retrievable chunks that keep their character spans.

Three strategies share one entry point, :func:`chunk_text`: `fixed` slices by size,
`paragraph` groups blank-line-separated blocks, and `markdown` additionally refuses to
merge blocks across a heading so a chunk never mixes two sections. Every chunk records
the character span it came from, which is what makes the citation spans in an answer
point back at real source text.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from local_docs_rag_agent.constants import (
    CHUNK_STRATEGIES,
    DEFAULT_CHUNK_STRATEGY,
    ChunkStrategyName,
)
from local_docs_rag_agent.exceptions import ConfigurationError
from local_docs_rag_agent.models import DocumentChunk


@dataclass(slots=True)
class Section:
    title: str
    heading_level: int | None
    start_char: int
    end_char: int
    text: str


@dataclass(slots=True)
class ParagraphBlock:
    text: str
    start_char: int
    end_char: int
    section_title: str
    heading_level: int | None


def chunk_text(
    source_path: Path,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
) -> list[DocumentChunk]:
    _validate_chunk_parameters(chunk_size, chunk_overlap)
    if not text.strip():
        return []

    strategy = _normalize_strategy(chunk_strategy)
    source_title = source_path.stem.replace("_", " ").strip() or source_path.name
    if strategy == "fixed":
        return _chunk_fixed(source_path, text, source_title, chunk_size, chunk_overlap)
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


def _normalize_strategy(chunk_strategy: str | None) -> ChunkStrategyName:
    if not chunk_strategy:
        return DEFAULT_CHUNK_STRATEGY
    normalized = chunk_strategy.strip().lower()
    if normalized in CHUNK_STRATEGIES:
        return normalized
    allowed = ", ".join(CHUNK_STRATEGIES)
    raise ConfigurationError(f"Chunk strategy must be one of {allowed}; got {chunk_strategy!r}")


def _validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ConfigurationError(f"Chunk size must be greater than 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ConfigurationError(f"Chunk overlap must be at least 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ConfigurationError(
            "Chunk overlap must be smaller than chunk size "
            f"(overlap={chunk_overlap}, size={chunk_size})"
        )


def _chunk_fixed(
    source_path: Path,
    text: str,
    source_title: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    sections = _extract_sections(text, source_title)
    chunks: list[DocumentChunk] = []
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
    source_path: Path,
    text: str,
    source_title: str,
    chunk_size: int,
    chunk_overlap: int,
    markdown_aware: bool,
) -> list[DocumentChunk]:
    blocks = _extract_paragraph_blocks(text, source_title, markdown_aware=markdown_aware)
    if not blocks:
        return _chunk_fixed(source_path, text, source_title, chunk_size, chunk_overlap)
    blocks = _split_oversized_blocks(
        blocks,
        source_text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[DocumentChunk] = []
    start_index = 0
    chunk_index = 0
    strategy = "markdown" if markdown_aware else "paragraph"

    while start_index < len(blocks):
        text_parts: list[str] = []
        current_section = blocks[start_index].section_title
        current_heading_level = blocks[start_index].heading_level
        start_char = blocks[start_index].start_char
        end_char = blocks[start_index].end_char
        cursor = start_index

        while cursor < len(blocks):
            block = blocks[cursor]
            candidate_parts = [*text_parts, block.text]
            candidate_text = "\n\n".join(candidate_parts).strip()
            section_changed = (
                markdown_aware and cursor > start_index and block.section_title != current_section
            )
            if candidate_text and len(candidate_text) > chunk_size and text_parts:
                break
            if section_changed and text_parts:
                break

            text_parts = candidate_parts
            end_char = block.end_char
            current_section = block.section_title
            current_heading_level = block.heading_level
            cursor += 1

            if len(candidate_text) >= chunk_size:
                break

        chunk_text_value = "\n\n".join(text_parts).strip()
        if not chunk_text_value:
            start_index += 1
            continue

        section = Section(
            title=current_section,
            heading_level=current_heading_level,
            start_char=start_char,
            end_char=end_char,
            text=chunk_text_value,
        )
        metadata = _build_metadata(
            strategy=strategy,
            source_title=source_title,
            section=section,
            start_char=start_char,
            end_char=end_char,
        )
        chunks.append(
            _make_chunk(
                source_path=source_path,
                title=current_section or source_title,
                text=chunk_text_value,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                metadata=metadata,
            )
        )
        chunk_index += 1

        if cursor >= len(blocks):
            break

        overlap_chars = 0
        next_start_index = max(start_index + 1, cursor - 1)
        probe = cursor - 1
        while probe > start_index:
            overlap_chars += len(blocks[probe].text)
            if overlap_chars >= chunk_overlap:
                next_start_index = probe
                break
            probe -= 1
            next_start_index = probe
        start_index = max(start_index + 1, next_start_index)

    return chunks


def _make_chunk(
    source_path: Path,
    title: str,
    text: str,
    chunk_index: int,
    start_char: int,
    end_char: int,
    metadata: dict[str, object],
) -> DocumentChunk:
    return DocumentChunk(
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
        Section(title=source_title, heading_level=None, start_char=0, end_char=len(text), text=text)
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
        heading_match = _parse_markdown_heading(raw_line) if markdown_aware else (None, None)
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
            body, content_start, content_end = _strip_span(source_text, start, end)
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
