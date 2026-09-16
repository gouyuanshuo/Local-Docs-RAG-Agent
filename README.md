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
- Optionally reranks a wider candidate window with a second stage before answering
- Answers questions with cited source chunks
- Exposes two runtimes:
  - `basic`
  - `agents_sdk`
- Supports OpenAI-compatible providers through `.env`
- Works with Qwen via DashScope-compatible endpoints
- Tracks eval metrics that can see *where* the evidence ranked, not only
  whether it was retrieved:
  - retrieval reciprocal rank (the leaderboard's primary key)
  - retrieval precision
  - keyword hit rate
  - source hit rate
  - citation span hit rate
  - response time
- Ships with a split frontend/backend setup:
  - `frontend`: Vite + React + TypeScript
  - `backend`: FastAPI + Python package

## Demo flow

The current happy path is:

1. Put docs under `data/corpus/sample/` (the default `DOCS_DIR`). `docs/`
   holds the project's own documentation and is never the corpus
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
- an optional second-stage reranker (`rag/rerank.py`, `rag/llm_rerank.py`) composed
  with the first stage in `rag/pipeline.py`
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
|-- docs/                          documentation; start at docs/README.md
|   |-- design/
|   |-- planning/
|   `-- reviews/
|-- data/
|   |-- corpus/sample/             default retrieval and eval corpus
|   `-- evals/                     gold eval set
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

Start with [`docs/README.md`](docs/README.md): it maps each document to the
question it answers and says where each kind of fact lives.

The files you will reach for most often:

- `AGENTS.md`
  - routing, engineering rules, code style, and the quality-gate commands
- `spec.md`
  - system requirements and scope
- `tasks.md`
  - current task board and short-horizon execution list
- `docs/design/architecture.md`
  - module boundaries, dependency direction, state ownership, and the failure contract
- `docs/development/`
  - getting started, workflow, testing, extension recipes, and troubleshooting
- `docs/planning/development-roadmap.md`
  - long-horizon roadmap and phase plan
- `docs/reviews/`
  - dated review snapshots, not maintained
- `.codex/config.toml`
  - project-local workflow hints
- `skills/backend-api/SKILL.md` and `skills/review-bugfix/SKILL.md`
  - reusable implementation and review workflows

## Quick start

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[agents,qdrant,dev]"
pnpm install
copy .env.example .env
pnpm run dev:backend   # API on http://127.0.0.1:8000
pnpm run dev           # UI on http://127.0.0.1:5173
```

No API key is needed for a first run: answers are marked `fallback` rather
than failing. The [getting-started guide](docs/development/getting-started.md)
walks through each step, including POSIX shells and a first question, and the
[CLI](docs/reference/cli.md) and [HTTP API](docs/reference/http-api.md)
references list every command and endpoint.

## Configuration

Settings live in `.env`, copied from `.env.example`. The
[configuration reference](docs/reference/configuration.md) lists every
variable with its default, its validation, and what it changes, including
retrieval strategies, reranking, proxy handling, and provider templates for
OpenAI and Qwen. The [metrics reference](docs/reference/metrics.md) says what
each eval number measures and what it cannot see.

## Tests

The suite is offline and deterministic. The quality-gate commands are in
[AGENTS.md](AGENTS.md#quality-gates), the
[testing guide](docs/development/testing.md) explains how tests are organized
and written, and CI runs the same gates.

## Project highlights

What gives this project portfolio value is not just “calling an API”, but combining several pieces into one coherent system:

- AI application design
- retrieval engineering
- provider abstraction
- tool-capable agent runtime
- citations and source tracking
- evaluation-aware iteration
- full-stack integration with a real frontend and backend split

## Status and plans

What is being worked on now is in [tasks.md](tasks.md); the phase plan is in
the [roadmap](docs/planning/development-roadmap.md).

## License

No license file has been added yet. If you plan to publish this repo publicly, adding one is a good next step.
