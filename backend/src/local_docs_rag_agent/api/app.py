"""Creates the FastAPI application: middleware, static hosting, and error handling.

The app serves the built frontend from `frontend/dist` when it exists, and falls back
to a JSON pointer toward the dev server when it does not, so the same entry point works
for a built deployment and for local development.

Domain errors are translated here rather than in each route: a `LocalDocsError` becomes
a response carrying its stable code and action hint, which is what lets the frontend
show an actionable message instead of a generic failure.
"""

from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from local_docs_rag_agent.api.routes import router
from local_docs_rag_agent.exceptions import (
    ConfigurationError,
    DataFormatError,
    LocalDocsError,
    ProviderUnavailableError,
    VectorStoreError,
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repository root from api/app.py")


FRONTEND_DIST_DIR = _repo_root() / "frontend" / "dist"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"
FRONTEND_INDEX_PATH = FRONTEND_DIST_DIR / "index.html"


def create_app() -> FastAPI:
    app = FastAPI(title="Local Docs RAG Agent", version="0.1.0")
    _configure_cors(app)
    _register_error_handlers(app)
    app.include_router(router)

    if FRONTEND_ASSETS_DIR.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=FRONTEND_ASSETS_DIR),
            name="assets",
        )

    @app.get("/", response_model=None)
    def index() -> FileResponse | JSONResponse:
        if FRONTEND_INDEX_PATH.is_file():
            return FileResponse(FRONTEND_INDEX_PATH)
        return JSONResponse(
            {
                "name": "Local Docs RAG Agent API",
                "message": "Frontend dev server not running. Start it with `pnpm run dev`.",
            }
        )

    return app


def _configure_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "http://localhost:5173",
            "http://localhost:5174",
        ],
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(LocalDocsError)
    async def handle_local_docs_error(
        request: Request,
        exc: LocalDocsError,
    ) -> JSONResponse:
        del request
        status_code = _status_code_for_error(exc)
        return JSONResponse(
            status_code=status_code,
            content={
                "code": exc.code,
                "detail": exc.message,
                "action_hint": exc.action_hint,
            },
        )


def _status_code_for_error(exc: LocalDocsError) -> int:
    if isinstance(exc, (ConfigurationError, DataFormatError)):
        return 400
    if isinstance(exc, (ProviderUnavailableError, VectorStoreError)):
        return 503
    return 500


app = create_app()


def run() -> None:
    uvicorn.run(
        "local_docs_rag_agent.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
