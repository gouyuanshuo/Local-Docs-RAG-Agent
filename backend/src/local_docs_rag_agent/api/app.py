"""Creates the FastAPI app: middleware, static hosting, error handling.

The app serves a valid built frontend when one is supplied or discovered in a
source checkout. It falls back to a JSON pointer toward the dev server when no
build is available, so an installed wheel can run as an API-only deployment.

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

from local_docs_rag_agent.api import routes
from local_docs_rag_agent.core import exceptions

_INVALID_FRONTEND_DIST_MESSAGE = (
    "frontend_dist_dir must contain index.html and an assets directory"
)


def _is_frontend_dist_dir(path: pathlib.Path) -> bool:
    return (
        path.is_dir()
        and (path / "index.html").is_file()
        and (path / "assets").is_dir()
    )


def _discover_frontend_dist_dir() -> pathlib.Path | None:
    current = pathlib.Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            frontend_dist_dir = parent / "frontend" / "dist"
            if _is_frontend_dist_dir(frontend_dist_dir):
                return frontend_dist_dir
            return None
    return None


def _select_frontend_dist_dir(
    frontend_dist_dir: pathlib.Path | None,
) -> pathlib.Path | None:
    if frontend_dist_dir is None:
        return _discover_frontend_dist_dir()
    if not _is_frontend_dist_dir(frontend_dist_dir):
        raise ValueError(_INVALID_FRONTEND_DIST_MESSAGE)
    return frontend_dist_dir


def create_app(
    frontend_dist_dir: pathlib.Path | None = None,
) -> fastapi.FastAPI:
    """Build the FastAPI application.

    Args:
      frontend_dist_dir: An optional built frontend directory. When omitted,
        a valid `frontend/dist` is discovered from a source checkout when
        available. An explicit directory must contain `index.html` and an
        `assets` directory.

    Returns:
      An app with CORS, error handlers, and API routes registered, and
      a built frontend mounted when one is available.

    Raises:
      ValueError: The explicit frontend directory is not a valid build.
    """
    selected_frontend_dist_dir = _select_frontend_dist_dir(frontend_dist_dir)
    frontend_assets_dir = (
        selected_frontend_dist_dir / "assets"
        if selected_frontend_dist_dir is not None
        else None
    )
    frontend_index_path = (
        selected_frontend_dist_dir / "index.html"
        if selected_frontend_dist_dir is not None
        else None
    )
    app = fastapi.FastAPI(title="Local Docs RAG Agent", version="0.1.0")
    _configure_cors(app)
    _register_error_handlers(app)
    app.include_router(routes.router)

    if frontend_assets_dir is not None:
        app.mount(
            "/assets",
            fastapi_staticfiles.StaticFiles(directory=frontend_assets_dir),
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
        if frontend_index_path is not None:
            return fastapi_responses.FileResponse(frontend_index_path)
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
