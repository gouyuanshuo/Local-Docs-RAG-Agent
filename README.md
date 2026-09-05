# Local Docs RAG Agent

Local Docs RAG Agent is a small but end-to-end AI application built around this workflow:

`Agent -> Tools -> RAG -> Citation -> Eval`

Instead of treating an LLM as a chatbot only, this project treats it like a working system: it can read local documents, retrieve relevant context, answer with citations, switch model backends, and run simple automated evaluation.

## Why this project exists

Large models do not know your private notes, lecture slides, assignment briefs, or project docs by default. This repo explores what it takes to build a practical docs assistant around that limitation.

The goal is not model training. The goal is to build a usable AI application with:

- document ingestion
- retrieval over external knowledge
- tool-using agent runtimes
- cited answers
- configurable providers
- evaluation harnesses for iteration

## What it does today

- Ingests local `.md` and `.txt` files into a retrieval index
- Supports local retrieval and Qdrant-backed retrieval
- Ranks with a selectable retrieval strategy, comparable side by side in the eval
  matrix: `blended`, `dense`, `lexical` (BM25), and `hybrid_rrf` (rank fusion)
- Answers questions with cited source chunks
- Exposes two runtimes:
  - `basic`
  - `agents_sdk`
- Supports OpenAI-compatible providers through `.env`
- Works with Qwen via DashScope-compatible endpoints
- Tracks simple eval metrics:
  - keyword hit rate
  - source hit rate
  - citation span hit rate
  - response time
- Ships with a split frontend/backend setup:
  - `frontend`: Vite + React + TypeScript
  - `backend`: FastAPI + Python package

## Demo flow

The current happy path is:

1. Put docs under `docs/`
2. Run ingestion
3. Ask a question
4. Retrieve relevant chunks
5. Generate an answer with citations
6. Run evals to measure quality

That means the repo already behaves like a small RAG product, not just a notebook experiment.

## Architecture

### Frontend

- Vite
- React
- TypeScript

The frontend talks to the backend over HTTP during development and can also be built into `frontend/dist` for backend-served deployment.

### Backend

- FastAPI
- Python package under `backend/src/local_docs_rag_agent`

The backend is responsible for:

- config loading
- provider setup
- ingestion
- retrieval
- agent runtime dispatch
- eval execution
- API responses for the frontend

### Retrieval layer

- chunking
- embedding generation
- selectable ranking strategies (`rag/retrieval.py`), including BM25 (`rag/bm25.py`)
  and reciprocal rank fusion (`rag/fusion.py`)
- local hybrid retrieval
- Qdrant vector retrieval
- citation span tracking

### Agent layer

- `basic` runtime for a lightweight orchestration path
- `agents_sdk` runtime for tool-enabled execution with OpenAI Agents SDK

## Repository layout

```text
.
|-- .codex/
|   `-- config.toml
|-- .github/
|   `-- workflows/ci.yml
|-- skills/
|   |-- backend-api/
|   `-- review-bugfix/
|-- backend/
|   `-- src/
|       `-- local_docs_rag_agent/
|           |-- api/               HTTP delivery: app factory, routes, schemas
|           |-- commands/          CLI command handlers
|           |-- evals/             eval harness and comparison matrix
|           |-- providers/         chat and embedding providers
|           |-- rag/               discovery, chunking, stores, ingest
|           |-- runtime/           answer runtimes and dispatch
|           |-- agent.py           one-question facade
|           |-- cli.py             argument parsing and command registry
|           |-- config.py          immutable validated configuration
|           |-- constants.py       shared closed option sets
|           |-- env.py             typed env readers and validators
|           |-- exceptions.py      expected-failure taxonomy
|           |-- models.py          framework-free domain records
|           |-- presenters.py      dataclass to JSON payload conversion
|           `-- tools.py           capabilities exposed to agent runtimes
|-- frontend/
|   |-- src/
|   |   |-- components/            focused panels and result rendering
|   |   |-- hooks/                 workspace state and actions
|   |   |-- lib/                   HTTP client and display formatting
|   |   `-- types/                 backend-facing contracts
|   |-- index.html
|   |-- package.json
|   `-- vite.config.ts
|-- docs/
|   |-- architecture.md
|   |-- development-roadmap.md
|   `-- sample/
|-- data/
|   `-- evals/
|-- scripts/                       manual, environment-dependent checks
|-- tests/
|-- AGENTS.md
|-- spec.md
|-- tasks.md
|-- pyproject.toml
|-- package.json
|-- pnpm-workspace.yaml
`-- README.md
```

Two conventions keep the backend extensible:

- `constants.py` is the single source of truth for every closed option set
  (runtimes, chunk strategies, vector backends, API styles). Configuration parsing,
  HTTP request validation, and CLI argument choices all derive from it, so adding an
  option is one edit plus its implementation.
- `rag/` is imported through its package facade. Code outside it imports from
  `local_docs_rag_agent.rag`, not from individual modules, so the internal split
  between the `ChunkStore` protocol, the two store implementations, and the ingest
  pipeline stays free to change.

## Project docs

These files define how the repo is organized and how work should move forward:

- `AGENTS.md`
  - project operating rules
- `spec.md`
  - system requirements and scope
- `tasks.md`
  - current task board and short-horizon execution list
- `docs/development-roadmap.md`
  - long-horizon roadmap and phase plan
- `docs/architecture.md`
  - module boundaries, dependency direction, state ownership, and extension rules
- `docs/code-review-2026-08-29.md`
  - deep review findings, resolved defects, and remaining risks
- `.codex/config.toml`
  - project-local workflow hints
- `skills/backend-api/SKILL.md`
  - reusable backend implementation workflow
- `skills/review-bugfix/SKILL.md`
  - reusable review and bugfix workflow

## Quick start

### 1. Create a virtual environment

```bash
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install backend dependencies

```bash
python -m pip install -e .[agents,qdrant]
```

### 3. Install frontend dependencies

```bash
pnpm install
```

### 4. Copy environment variables

```bash
copy .env.example .env
```

The app auto-loads a project-local `.env`, so you do not need to export variables manually before each run.

## Development workflow

### Frontend

```bash
pnpm run dev
```

Vite runs on `http://127.0.0.1:5173` by default.

### Backend

```bash
pnpm run dev:backend
```

FastAPI runs on `http://127.0.0.1:8000`.

### CLI

You can still use the CLI directly:

```bash
local-docs-rag ingest
local-docs-rag ask "How is attention explained in lecture 5?"
local-docs-rag eval
```

## API

The FastAPI app currently exposes:

- `GET /api/health`
- `GET /api/info`
- `GET /api/documents`
- `POST /api/ingest`
- `POST /api/ask`
- `POST /api/eval`
- `POST /api/eval/compare`

## Configuration

The most important settings live in `.env`.

### LLM

- `LLM_PROVIDER`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`
- `LLM_API_STYLE`

### Embeddings

- `EMBEDDING_PROVIDER`
- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSIONS`
- `EMBEDDING_BATCH_SIZE`
- `EMBEDDING_MAX_RETRIES`
- `EMBEDDING_RETRY_BACKOFF_MS`

### Retrieval ranking

- `RETRIEVAL_STRATEGY=blended` — the original `max(dense, lexical, weighted mix)`
  ranking. Kept as the default so earlier eval numbers stay reproducible.
- `RETRIEVAL_STRATEGY=dense` — embedding cosine similarity alone
- `RETRIEVAL_STRATEGY=lexical` — Okapi BM25 alone
- `RETRIEVAL_STRATEGY=hybrid_rrf` — dense and BM25 fused by reciprocal rank
- `RETRIEVAL_CANDIDATE_K` — how deep each signal ranks before fusion (default 20).
  Widened automatically when `TOP_K` exceeds it.
- `RRF_K` — fusion damping constant (default 60)

Scores are only comparable *within* a strategy: an RRF score is a sum of reciprocal
ranks and sits near 0.03, while a cosine similarity sits near 1. Compare strategies
with `eval-compare`, not by reading scores side by side.

```bash
python -m local_docs_rag_agent.cli eval-compare \n  --retrieval-strategy blended --retrieval-strategy hybrid_rrf
```

### Retrieval backend

- `VECTOR_BACKEND=local`
- `VECTOR_BACKEND=qdrant`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`
- `QDRANT_TIMEOUT_S`

### External HTTP proxy behavior

OpenAI-compatible chat, embeddings, Agents SDK, and Qdrant clients honor standard
`HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` variables by default. If a stale local
proxy causes connection-refused errors, set this in `.env` instead of changing the
machine-wide environment:

```bash
EXTERNAL_HTTP_TRUST_ENV=false
```

Keep the default value `true` when the provider is reachable only through a proxy.

`scripts/` holds manual, environment-dependent checks. They are named `check_*` and
`verify_*` rather than `test_*` precisely because they are not part of the pytest
suite: they talk to live services and must never run in CI.

Read-only checks, safe against any deployment:

```bash
python scripts/check_chat_api.py
python scripts/check_embedding_api.py
python scripts/check_qdrant.py
```

The write/read/eval live gate writes to the configured collection, so run it only
against a disposable or explicitly approved one. It fails if the runtime, chat, or
embeddings silently degrade to fallback:

```bash
python scripts/verify_live_qdrant.py
```

### Runtime

- `AGENT_RUNTIME=basic`
- `AGENT_RUNTIME=agents_sdk`

## Example provider setups

### OpenAI

```bash
LLM_PROVIDER=openai
LLM_API_KEY=your_openai_key
LLM_BASE_URL=
LLM_MODEL=gpt-4.1-mini
LLM_API_STYLE=responses
```

### Qwen via DashScope-compatible endpoint

```bash
LLM_PROVIDER=qwen
LLM_API_KEY=your_dashscope_key
LLM_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
LLM_API_STYLE=chat_completions
EMBEDDING_PROVIDER=qwen
EMBEDDING_API_KEY=your_dashscope_key
EMBEDDING_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024
EMBEDDING_BATCH_SIZE=10
```

## Tests

Install the development extra and run the focused backend suite:

```bash
python -m pip install -e .[agents,qdrant,dev]
python -m pytest
```

Run the complete deterministic quality gate with:

```bash
python -m ruff check backend/src tests scripts
python -m mypy backend/src
python -m pytest
python -m compileall -q backend/src
pnpm run build
```

The suite covers environment parsing and configuration validation, CLI command
wiring, provider configuration, embedding batching/retry behavior, Qdrant network
diagnostics, incremental-ingest lifecycle behavior, prompt-context formatting and the
extractive fallback, and deterministic Agents SDK tool/runner integration, all without
calling live model providers.

Pull requests and pushes run the same backend gates plus the strict TypeScript/Vite
build through `.github/workflows/ci.yml`.

## Project highlights

What gives this project portfolio value is not just “calling an API”, but combining several pieces into one coherent system:

- AI application design
- retrieval engineering
- provider abstraction
- tool-capable agent runtime
- citations and source tracking
- evaluation-aware iteration
- full-stack integration with a real frontend and backend split

## Current status

Completed or substantially completed:

- stage 1: minimal runnable agent
- stage 2: tools and runtime paths
- stage 3: RAG and citations
- stage 4: initial eval harness
- stage 5: provider abstraction
- stage 6: first round of engineering and frontend/backend separation

## Next steps

Planned improvements include:

- richer eval datasets
- more robust retry and failure handling
- stronger Agents SDK orchestration
- better frontend result presentation
- optional Ollama/local model support
- deployment polish

## Notes

- On this machine, `py -3` may resolve to the free-threaded interpreter (`3.13t`), which can cause dependency issues with `pydantic-core`.
- Prefer `py -3.13`.
- `.env` is intentionally excluded from git.

## License

No license file has been added yet. If you plan to publish this repo publicly, adding one is a good next step.
