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
|-- skills/
|   |-- backend-api/
|   `-- review-bugfix/
|-- backend/
|   `-- src/
|       `-- local_docs_rag_agent/
|           |-- api/
|           |-- commands/
|           |-- evals/
|           |-- providers/
|           |-- rag/
|           |-- runtime/
|           |-- agent.py
|           |-- cli.py
|           |-- config.py
|           |-- models.py
|           |-- presenters.py
|           `-- tools.py
|-- frontend/
|   |-- src/
|   |-- index.html
|   |-- package.json
|   `-- vite.config.ts
|-- docs/
|   |-- development-roadmap.md
|   `-- sample/
|-- data/
|   `-- evals/
|-- scripts/
|-- AGENTS.md
|-- spec.md
|-- tasks.md
|-- pyproject.toml
|-- package.json
|-- pnpm-workspace.yaml
`-- README.md
```

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
- `POST /api/ingest`
- `POST /api/ask`
- `POST /api/eval`

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

### Retrieval backend

- `VECTOR_BACKEND=local`
- `VECTOR_BACKEND=qdrant`

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
```

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
