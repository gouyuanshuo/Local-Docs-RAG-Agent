# Deployment and rollback

The supported first deployment is one process on one host, bound to loopback.
It may serve only the API or an explicitly supplied frontend build. The
application has no authentication or TLS termination, so do not expose it on a
LAN or the public internet until those controls and their operating policy have
been designed separately.

## Supported toolchain and artifacts

The backend supports Python 3.11 and 3.13. CI exercises both; use Python 3.13
for the packaging examples below. Frontend builds use Node.js 22 (verified with
22.23.2) and pnpm 10.18.0.

A release has two independent artifacts:

- the Python wheel, which contains the API and CLI but **does not contain
  frontend assets**; and
- the matching `frontend/dist` directory, when the deployment serves the UI.

Keep the wheel, static directory, `pyproject.toml`, `uv.lock`, exported runtime
requirements, and checksums together in the release record. A rollback needs
the prior wheel and its matching static build, not whichever `frontend/dist`
happens to be in a checkout.

## Build and verify a release

Restore the exact development resolution and run the offline gates before
building:

```bash
uv sync --locked --all-extras --python 3.13
uv run --locked --all-extras python -m ruff check backend/src tests scripts
uv run --locked --all-extras python -m ruff format --check \
  backend/src tests scripts
uv run --locked --all-extras python -m ruff check \
  --preview --select DOC201 backend/src scripts
uv run --locked --all-extras python -m mypy
uv run --locked --all-extras python -m pytest
uv run --locked --all-extras python -m compileall -q backend/src
uv build --wheel --out-dir dist

pnpm install --frozen-lockfile
pnpm --filter local-docs-rag-agent-web test
pnpm run build
```

The backend restore is locked by `uv.lock`. Isolated wheel builds do not read
that lock, so the exact build requirements are pinned in `[build-system]` in
`pyproject.toml`. The frontend restore is frozen by `pnpm-lock.yaml`.

Export the locked runtime dependencies beside the wheel. This base export is
enough for the local backend and basic runtime:

```bash
uv export --quiet --locked --no-dev --no-emit-project \
  --format requirements.txt \
  --output-file dist/runtime-requirements.txt
```

Add `--extra agents` and/or `--extra qdrant` to that export when the deployment
uses those optional integrations.

Install and test the actual wheel in a fresh environment whose run directory is
outside the checkout and outside any ancestor containing `pyproject.toml`:

```bash
release_root="$(pwd)/dist"
smoke_root="$(mktemp -d)"
wheel_path="$(find "$release_root" -maxdepth 1 -type f \
  -name 'local_docs_rag_agent-*.whl' -print -quit)"
test -n "$wheel_path"
mkdir -p "$smoke_root/run"
uv venv --python 3.13 "$smoke_root/venv"
uv pip install \
  --python "$smoke_root/venv/bin/python" \
  -r "$release_root/runtime-requirements.txt"
uv pip install \
  --python "$smoke_root/venv/bin/python" \
  --no-deps "$wheel_path"
cd "$smoke_root/run"
env -u PYTHONPATH \
  "$smoke_root/venv/bin/python" -c \
  'from local_docs_rag_agent.api import app; assert app.create_app()'
```

This smoke must not use an editable install: importing from the source tree
would hide a missing package file or checkout-only startup dependency. CI
performs the same kind of isolated install and checks `GET /api/health` plus
the API-only root.

## Configure the process

Every route that needs configuration calls `AppConfig.from_env`. Exported
process variables take precedence over values loaded from `.env`; dotenv
loading never overwrites an existing variable. Automatic `.env` discovery is
convenient in a source checkout. For an installed service, use the service
manager's process environment so the configuration location is explicit.

Restart the process after changing either source. Values loaded from `.env`
enter the process environment on first use and are not overwritten by later
loads, so editing the file under a running process is not a reliable reload
mechanism. Check the effective non-secret settings with `GET /api/info` after a
restart.

Use absolute paths in a packaged deployment. The relative defaults resolve
from the process working directory:

| Setting | Access required | Purpose |
| --- | --- | --- |
| `DOCS_DIR` | read directory and source files | corpus to ingest |
| `INDEX_PATH` | read/write file and create/rename in parent | local JSONL index |
| `INGEST_MANIFEST_PATH` | read/write file and create/rename in parent | index ownership and recovery state |
| `EVAL_PATH` | read file | eval cases |

The service account must also be able to create advisory lock files beside the
local index and manifest. A Qdrant deployment creates same-host lock artifacts
under `~/.cache/local-docs-rag-agent/locks`, so that cache directory must be
writable by the service account. Keep documents, generated index data, eval
data, logs, and release artifacts in separately permissioned locations. Do not
put provider or Qdrant credentials in the frontend build.

## Run API-only

The installed console entry point binds only to `127.0.0.1:8000`:

```bash
/opt/local-docs-rag/current/venv/bin/local-docs-rag-api
```

In another shell, check liveness and then configuration/corpus access:

```bash
curl --fail --silent http://127.0.0.1:8000/api/health
curl --fail --silent http://127.0.0.1:8000/api/info
curl --fail --silent http://127.0.0.1:8000/api/documents
```

`/api/health` proves only that the process can answer. `/api/info` and
`/api/documents` exercise configuration and document discovery. In API-only
mode, `GET /` returns a JSON development pointer; that is expected and is not a
failed static deployment.

## Serve an explicit frontend build

A production frontend uses same-origin `/api/*` requests by default. Vite
development instead uses `http://127.0.0.1:8000`. To build for a different API
origin, set `VITE_API_BASE_URL` for the build:

```bash
VITE_API_BASE_URL=https://api.example.invalid pnpm run build
```

The value is compiled into the JavaScript bundle. Changing it requires a new
build; it is not an `AppConfig` setting and changing the backend process
environment cannot update an existing bundle. A separate origin also needs a
deliberate CORS, authentication, and TLS design. Same-origin serving on
loopback is the supported production default.

Copy the complete `frontend/dist` directory with the release. It must contain
`index.html` and an `assets` directory. Construct the app with that explicit
path; the normal `local-docs-rag-api` entry point does not accept a static path:

```python
from pathlib import Path

import uvicorn

from local_docs_rag_agent.api import app as api_app


application = api_app.create_app(
    frontend_dist_dir=Path("/opt/local-docs-rag/releases/0.1.0/frontend-dist")
)

if __name__ == "__main__":
    uvicorn.run(application, host="127.0.0.1", port=8000)
```

Run that launcher with the release virtual environment. An explicit path is
authoritative and startup raises `ValueError` if it lacks either required
entry. Verify `GET /` returns HTML, an `/assets/...` URL from that HTML returns
success, and `GET /api/health` still returns `{"status":"ok", ...}`.

## Back up data consistently

### Local JSONL backend

`INDEX_PATH` and `INGEST_MANIFEST_PATH` are one logical state. Stop the API and
all CLI writers using those targets, confirm they have exited, and copy both
files as one backup set before upgrading or rolling back. Do not copy one while
the other can still change. Lock files are stable coordination artifacts, not
data, and do not need to be restored.

The corpus remains the source of truth. A manifest with `repair_required` or
`needs_reindex` makes the next ask, eval, or explicit ingest repair the affected
sources. If an index/manifest pair is missing, corrupt, belongs to another
target, or is incompatible with the selected release, do not hand-edit the
manifest. Move both data files aside as a recoverable pair and run:

```bash
local-docs-rag ingest
```

That rebuilds from `DOCS_DIR`. Restore a prior pair only when it was captured
with the same corpus, target, and release/configuration record.

### Qdrant backend

The repository does not implement a Qdrant backup or distributed transaction.
Its advisory lock coordinates only processes owned by the same user on one
host. Stop lifecycle writers on every host and use the Qdrant operator's
supported snapshot/restore procedure for the exact collection; capture the
matching local ingest manifest in the same maintenance window. Multi-host
write exclusion, snapshot retention, restore testing, and collection access
control are operator responsibilities.

## Roll back

1. Stop the running process and every writer using the same local files or
   Qdrant collection.
2. Preserve the current local index/manifest pair, or take the coordinated
   Qdrant snapshot and manifest backup, before replacing anything.
3. Select the prior release record: wheel, exported locked requirements,
   matching static build, `pyproject.toml`, `uv.lock`, and artifact checksums.
4. Install the prior wheel into its own fresh virtual environment. Do not
   overwrite the current environment in place.
5. Restore the prior release's matching index/manifest pair when persistence
   format or index-defining configuration changed. Otherwise preserve the
   current pair and let `ensure_index` rebuild when its fingerprint or recovery
   state requires it. For Qdrant, follow the operator restore runbook.
6. Point the launcher at the prior static directory, start on loopback, and
   check `/api/health`, `/api/info`, `/api/documents`, one representative ask,
   the HTML root, and one built asset before retiring the failed release.

### Disposable local rehearsal

Rehearse both upgrade and rollback without production credentials or data:

1. Install the candidate and prior wheels into separate temporary virtual
   environments outside the checkout, using each release's exported locked
   requirements. Keep each matching `frontend/dist` beside its wheel.
2. Create a temporary corpus, eval file, local index path, and manifest path.
   Set all four paths to absolute locations under that temporary directory.
3. Set `PYTHON_DOTENV_DISABLED=1`, `VECTOR_BACKEND=local`,
   `AGENT_RUNTIME=basic`, `RERANKER=none`,
   `EXTERNAL_HTTP_TRUST_ENV=false`, and blank values for `LLM_API_KEY`,
   `OPENAI_API_KEY`, `LLM_BASE_URL`, `EMBEDDING_API_KEY`,
   `EMBEDDING_BASE_URL`, `QDRANT_URL`, and `QDRANT_API_KEY`. This keeps the
   rehearsal offline and prevents a repository `.env` from supplying live
   credentials.
4. Start the candidate on an unused loopback port, ingest, perform the startup
   checks, and copy its stopped index/manifest pair as a backup.
5. Stop it, switch to the prior wheel and matching static directory, restore
   the backup pair appropriate to that release (or rebuild from the temporary
   corpus), and repeat the checks.
6. Delete the temporary directory only after recording whether artifact
   selection, data restoration, and health/static checks all behaved as the
   runbook says.

This rehearsal validates the mechanics, not provider or Qdrant availability.
Live provider and Qdrant verification remains a separate, explicitly approved
operation.
