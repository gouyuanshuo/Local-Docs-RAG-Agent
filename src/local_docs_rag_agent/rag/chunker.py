from __future__ import annotations

from pathlib import Path

from local_docs_rag_agent.models import DocumentChunk


def chunk_text(
    source_path: Path,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    clean_text = text.strip()
    if not clean_text:
        return []

    title = source_path.stem.replace("_", " ").strip() or source_path.name
    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0

    while start < len(clean_text):
        end = min(len(clean_text), start + chunk_size)
        chunk_body = clean_text[start:end].strip()
        if chunk_body:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{source_path.as_posix()}::chunk-{chunk_index}",
                    source_path=source_path.as_posix(),
                    title=title,
                    text=chunk_body,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata={"start": start, "end": end},
                )
            )
            chunk_index += 1

        if end >= len(clean_text):
            break
        start = max(0, end - chunk_overlap)

    return chunks
