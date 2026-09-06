"""Creates the FastAPI app: middleware, static hosting, error handling.

The app serves the built frontend from `frontend/dist` when it exists, and falls
back to a JSON pointer toward the dev server when it does not, so the same entry
point works for a built deployment and for local development.

Domain errors are translated here rather than in each route: a `LocalDocsError`
becomes a response carrying its stable code and action hint, which is what lets
the frontend show an actionable message instead of a generic failure.
"""

from __future__ import annotations

import pathlib

import fastapi
import uvicorn
from fastapi import responses as fastapi_responses
from fastapi import staticfiles as fastapi_staticfiles
from fastapi.middleware import cors as fastapi_cors

from local_docs_rag_agent import exceptions
from local_docs_rag_agent.api import routes


def _repo_root() -> pathlib.Path:
    current = pathlib.Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repository root from api/app.py")


FRONTEND_DIST_DIR = _repo_root() / "frontend" / "dist"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"
FRONTEND_INDEX_PATH = FRONTEND_DIST_DIR / "index.html"


def create_app() -> fastapi.FastAPI:
    """Build the FastAPI application.

    Returns:
      An app with CORS, error handlers, and API routes registered, and
      the built frontend mounted when `frontend/dist` exists.
    """
    app = fastapi.FastAPI(title="Local Docs RAG Agent", version="0.1.0")
    _configure_cors(app)
    _register_error_handlers(app)
    app.include_router(routes.router)

    if FRONTEND_ASSETS_DIR.is_dir():
        app.mount(
            "/assets",
            fastapi_staticfiles.StaticFiles(directory=FRONTEND_ASSETS_DIR),
            name="assets",
        )

    @app.get("/", response_model=None)
    def index() -> (
        fastapi_responses.FileResponse | fastapi_responses.JSONResponse
    ):
        """Serve the built frontend, or say where to find it in dev.

        Returns:
          The built `index.html` when one exists, otherwise a JSON
          pointer to the dev server, so hitting the API root during
          development reads as a hint rather than a 404.
        """
        if FRONTEND_INDEX_PATH.is_file():
            return fastapi_responses.FileResponse(FRONTEND_INDEX_PATH)
        return fastapi_responses.JSONResponse(
            {
                "name": "Local Docs RAG Agent API",
                "message": (
                    "Frontend dev server not running. "
                    "Start it with `pnpm run dev`."
                ),
            }
        )

    return app


def _configure_cors(app: fastapi.FastAPI) -> None:
    app.add_middleware(
        fastapi_cors.CORSMiddleware,
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


def _register_error_handlers(app: fastapi.FastAPI) -> None:
    @app.exception_handler(exceptions.LocalDocsError)
    async def handle_local_docs_error(
        request: fastapi.Request,
        exc: exceptions.LocalDocsError,
    ) -> fastapi_responses.JSONResponse:
        """Render an expected failure as the documented error payload.

        Args:
          request: Unused. The status depends on the error, not the
            route that raised it.
          exc: The failure to render.

        Returns:
          The error payload, with the stable code and action hint the
          exception carries.
        """
        del request
        status_code = _status_code_for_error(exc)
        return fastapi_responses.JSONResponse(
            status_code=status_code,
            content={
                "code": exc.code,
                "detail": exc.message,
                "action_hint": exc.action_hint,
            },
        )


def _status_code_for_error(exc: exceptions.LocalDocsError) -> int:
    if isinstance(
        exc, (exceptions.ConfigurationError, exceptions.DataFormatError)
    ):
        return 400
    if isinstance(
        exc, (exceptions.ProviderUnavailableError, exceptions.VectorStoreError)
    ):
        return 503
    return 500


app = create_app()


def run() -> None:
    """Serve the application on localhost:8000, for the console script."""
    uvicorn.run(
        "local_docs_rag_agent.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
