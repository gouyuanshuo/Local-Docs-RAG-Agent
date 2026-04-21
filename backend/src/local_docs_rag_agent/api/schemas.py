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


class AppInfoResponse(BaseModel):
    name: str
    runtime: str
    vector_backend: str
    docs_dir: str
