# Getting started

From a fresh clone to a question answered with citations. Everything here works
without an API key; a key only turns degraded answers into live ones.

## Prerequisites

| Tool | Version | Why |
| --- | --- | --- |
| Python | 3.11 or 3.13 | CI runs both supported boundaries |
| uv | 0.12.16 | resolves and restores `uv.lock` |
| Node.js | 22 | CI runs 22 |
| pnpm | 10.18.0 | matches `packageManager` and CI |

On Windows, `py -3` can resolve to the free-threaded interpreter (`3.13t`),
which breaks the `pydantic-core` wheel. Ask for the regular build explicitly
with `py -3.13`.

## 1. Restore the backend

```text
uv python install 3.13
uv sync --locked --all-extras --python 3.13
```

This creates `.venv` and installs the exact universal resolution in `uv.lock`.
The lock covers Python 3.11+ and all three extras. Use `--python 3.11` to work
at the other CI-tested boundary. Do not replace `--locked` with an unlocked
install during an ordinary restore.

The first two extras remain optional to package users. They are installed in a
development restore so tests exercise the real integrations. Each is imported
lazily, inside the only code that needs it.

| Extra | Installs | Needed for |
| --- | --- | --- |
| `agents` | `openai-agents` | `AGENT_RUNTIME=agents_sdk` |
| `qdrant` | `qdrant-client` | `VECTOR_BACKEND=qdrant` |
| `dev` | `ruff`, `mypy`, `pytest`, `pytest-cov` | the quality gates |

## 2. Restore the frontend

From the repository root, not from `frontend/`:

```bash
pnpm install --frozen-lockfile
```

## 3. Configure

```powershell
copy .env.example .env
```

```bash
cp .env.example .env
```

The app loads a project-local `.env` on start, without overriding variables
already exported in your shell. `.env` is gitignored; only `.env.example` is
tracked.

Leave the keys empty for a first run. Chat and embeddings then report `fallback`
in every answer's diagnostics instead of failing, which is enough to see the
whole pipeline work. Two things are worth knowing before you change anything:

- `DOCS_DIR` defaults to `data/corpus/sample`, the corpus the eval set is
  written against. Point it at your own documents if you like, but never at
  `docs/`: that is the project's documentation, and indexing it plants
  near-duplicate distractors next to the eval questions.
- The Qdrant backend refuses fallback embeddings, so it needs a working
  embedding key. The local backend does not.

## 4. Run it

Backend, with reload:

```bash
pnpm run dev:backend
```

This runs uvicorn on `http://127.0.0.1:8000` using the Windows virtual
environment path. Elsewhere, run it through the locked environment:

```bash
uv run --locked --all-extras python -m uvicorn \
  local_docs_rag_agent.api.app:app --reload --host 127.0.0.1 --port 8000
```

`local-docs-rag-api` serves the same app on the same address without reload.

Frontend, in a second terminal:

```bash
pnpm run dev
```

Vite serves the UI on `http://127.0.0.1:5173`.

## 5. First question

```bash
uv run --locked --all-extras local-docs-rag ingest
uv run --locked --all-extras local-docs-rag ask \
  "How is attention explained in lecture 5?"
uv run --locked --all-extras local-docs-rag eval
```

Read the `diagnostics` block in the answer before the answer itself:

- `requested_runtime` and `actual_runtime` differ when a runtime handed off to
  another.
- `chat_provider`, `embedding_provider`, and `reranker` each carry a `mode`.
  `live` means it really ran. `fallback` means it degraded, and `reason` says
  why.

An answer produced under `fallback` is labelled as one on purpose. The project
never reports a degraded run as a successful one.

## 6. Check your setup

Run the quality gates from [AGENTS.md](../../AGENTS.md#quality-gates). They are
offline and deterministic, so they pass on a fresh clone with no keys at all.
