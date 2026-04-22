from __future__ import annotations

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    runtime: str | None = None


class EvalRequest(BaseModel):
    runtime: str | None = None


class IngestResponse(BaseModel):
    num_chunks: int
    vector_backend: str


class HealthResponse(BaseModel):
    status: str
    backend_time_utc: str


class AppInfoResponse(BaseModel):
    name: str
    runtime: str
    vector_backend: str
    docs_dir: str
    docs_exclude_patterns: list[str]
    docs_count: int
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    top_k: int
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    qdrant_collection: str


class DocumentsResponse(BaseModel):
    count: int
    documents: list[str]
