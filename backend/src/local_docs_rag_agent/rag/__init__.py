"""Retrieval-augmented generation: document discovery, chunking, storage, and ingest.

This module is the package facade. Code outside `rag` should import from here rather
than reaching into individual modules, so the internal split between the protocol, the
two store implementations, and the ingest pipeline stays free to change.

The dependency direction inside the package runs one way:

    discovery -> chunker -> ingest -> store_factory -> base/local_store/qdrant_store
                                                    -> retrieval -> bm25/fusion/scoring

`base` and `models` sit at the bottom and import nothing from the layers above them.
"""

from local_docs_rag_agent.rag.base import ChunkStore
from local_docs_rag_agent.rag.chunker import chunk_text
from local_docs_rag_agent.rag.discovery import (
    SUPPORTED_EXTENSIONS,
    collect_document_paths,
    read_source_texts,
)
from local_docs_rag_agent.rag.ingest import ensure_index, ingest_documents
from local_docs_rag_agent.rag.local_store import LocalJsonlChunkStore
from local_docs_rag_agent.rag.qdrant_store import QdrantChunkStore
from local_docs_rag_agent.rag.retrieval import RetrievalSettings, rank_chunks
from local_docs_rag_agent.rag.store_factory import build_store, retrieval_settings

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "ChunkStore",
    "LocalJsonlChunkStore",
    "QdrantChunkStore",
    "RetrievalSettings",
    "build_store",
    "chunk_text",
    "collect_document_paths",
    "ensure_index",
    "ingest_documents",
    "rank_chunks",
    "read_source_texts",
    "retrieval_settings",
]
