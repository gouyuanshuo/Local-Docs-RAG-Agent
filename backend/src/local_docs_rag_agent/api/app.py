from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from local_docs_rag_agent.agent import LocalDocsAgent
from local_docs_rag_agent.api.schemas import (
    AppInfoResponse,
    AskRequest,
    EvalRequest,
    HealthResponse,
    IngestResponse,
)
from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.evals.harness import run_eval
from local_docs_rag_agent.presenters import serialize_answer, serialize_eval_summary
from local_docs_rag_agent.rag.ingest import ensure_index, ingest_documents



def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repository root from api/app.py")


FRONTEND_DIR = _repo_root() / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"


def create_app() -> FastAPI:
    app = FastAPI(title="Local Docs RAG Agent", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if FRONTEND_DIST_DIR.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST_DIR / "assets"), name="assets")

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/api/info", response_model=AppInfoResponse)
    def info() -> AppInfoResponse:
        config = AppConfig.from_env()
        return AppInfoResponse(
            name="Local Docs RAG Agent",
            runtime=config.agent_runtime,
            vector_backend=config.vector_backend,
            docs_dir=str(config.docs_dir),
        )

    @app.post("/api/ingest", response_model=IngestResponse)
    def ingest() -> IngestResponse:
        config = AppConfig.from_env()
        chunks = ingest_documents(config)
        return IngestResponse(num_chunks=len(chunks), vector_backend=config.vector_backend)

    @app.post("/api/ask")
    def ask(payload: AskRequest) -> dict[str, object]:
        config = AppConfig.from_env().with_runtime(payload.runtime)
        ensure_index(config)
        agent = LocalDocsAgent(config)
        answer = agent.answer(payload.question)
        return serialize_answer(answer, runtime=config.agent_runtime)

    @app.post("/api/eval")
    def evaluate(payload: EvalRequest) -> dict[str, object]:
        config = AppConfig.from_env().with_runtime(payload.runtime)
        ensure_index(config)
        results = run_eval(config)
        return serialize_eval_summary(results, runtime=config.agent_runtime)

    @app.get("/", response_model=None)
    def index() -> FileResponse | JSONResponse:
        if FRONTEND_DIST_DIR.exists():
            return FileResponse(FRONTEND_DIST_DIR / "index.html")
        return JSONResponse(
            {
                "name": "Local Docs RAG Agent API",
                "message": "Frontend dev server not running. Start it with `pnpm run dev`.",
            }
        )

    return app


app = create_app()


def run() -> None:
    uvicorn.run(
        "local_docs_rag_agent.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
