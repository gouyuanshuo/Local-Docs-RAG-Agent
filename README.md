# Local Docs RAG Agent

Local Docs RAG Agent is a small AI application scaffold focused on the workflow:

`Agent -> Tools -> RAG -> Citation -> Eval`

This repository is intentionally lightweight. It gives you a clean MVP structure you can run locally today, while keeping clear extension points for:

- OpenAI as the first model provider
- Qdrant as the retrieval backend
- future Ollama/local model support
- automatic evaluation and experiment comparison

## What is included

- A Python package with clear module boundaries
- A document ingestion pipeline for local `.md` / `.txt` files
- A retriever that works locally out of the box
- Optional Qdrant integration points
- A provider abstraction for model backends
- A simple docs QA agent that answers with citations
- A JSONL eval harness
- Sample docs and sample eval cases

## Repository layout

```text
.
|-- README.md
|-- pyproject.toml
|-- package.json
|-- pnpm-workspace.yaml
|-- .env.example
|-- docs/
|   `-- sample/
|       `-- lecture5_attention.md
|-- data/
|   `-- evals/
|       `-- sample_eval.jsonl
|-- backend/
|   `-- src/
|       `-- local_docs_rag_agent/
|           |-- agent.py
|           |-- cli.py
|           |-- config.py
|           |-- models.py
|           |-- tools.py
|           |-- api/
|           |-- providers/
|           |-- rag/
|           |-- runtime/
|           `-- evals/
`-- frontend/
    |-- index.html
    |-- package.json
    `-- src/
```

## Quick start

1. Create and activate a virtual environment:

```bash
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the package:

```bash
python -m pip install -e .[agents,qdrant]
```

3. Copy environment variables:

```bash
copy .env.example .env
```

The CLI automatically loads values from a project-local `.env`, so you do not need to manually export each variable before running commands.

4. Build a local index from the sample docs:

```bash
local-docs-rag ingest
```

5. Ask a question:

```bash
local-docs-rag ask "How is attention explained in lecture 5?"
```

6. Run the sample eval harness:

```bash
local-docs-rag eval
```

7. Start the FastAPI app:

```bash
local-docs-rag-api
```

8. Install frontend dependencies:

```bash
pnpm install
```

9. Start the separate frontend dev server:

```bash
pnpm run dev
```

10. Start the backend dev server in another terminal:

```bash
pnpm run dev:backend
```

Then open `http://127.0.0.1:5173` for the Vite frontend, or `http://127.0.0.1:8000` for the backend API.

## Configuration

The default setup uses:

- local docs under `docs/`
- a local JSONL chunk store under `data/index/chunks.jsonl`
- OpenAI as the answer-generation provider

If `LLM_API_KEY` is not set, the agent still retrieves context and returns an extractive fallback answer so the pipeline remains testable.

Provider settings now live in `.env`:

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`
- `LLM_API_STYLE`
- `EMBEDDING_MODEL`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_DIMENSIONS`

Defaults:

- OpenAI: leave `LLM_BASE_URL` empty and keep `LLM_API_STYLE=responses`
- Qwen/DashScope OpenAI-compatible mode: set `LLM_BASE_URL` and usually use `LLM_API_STYLE=chat_completions`

Example `.env` for OpenAI:

```bash
LLM_PROVIDER=openai
LLM_API_KEY=your_openai_key
LLM_BASE_URL=
LLM_MODEL=gpt-4.1-mini
LLM_API_STYLE=responses
```

Example `.env` for Qwen via DashScope compatible mode:

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

## Retrieval backends

- `VECTOR_BACKEND=local`: stores chunk records in JSONL and performs hybrid dense-plus-lexical retrieval locally
- `VECTOR_BACKEND=qdrant`: stores dense embeddings in Qdrant and retrieves by vector similarity

If you want to use Qdrant, install the extra dependency:

```bash
py -3.13 -m pip install -e .[agents,qdrant]
```

## Runtime modes

There are two answer runtimes:

- `basic`: the local fallback orchestrator already included in the repo
- `agents_sdk`: OpenAI Agents SDK function-tool runtime for stage 2

Use either environment variables or CLI flags:

```bash
set AGENT_RUNTIME=agents_sdk
local-docs-rag ask "How is attention explained in lecture 5?"
```

or

```bash
local-docs-rag ask --runtime agents_sdk "How is attention explained in lecture 5?"
```

## Python note

On this machine, `py -3` resolves to the free-threaded interpreter (`3.13t`), which may not have compatible wheels for `pydantic-core`.
Prefer the standard interpreter:

```bash
py -3.13
```

## Frontend and backend

The project now supports a real split dev workflow:

- `frontend/`: Vite + React + TypeScript
- `backend/`: FastAPI + Python package

The backend enables CORS for the Vite dev server at `http://127.0.0.1:5173`.

For production-style serving, `pnpm build` writes the frontend to `frontend/dist`, and the FastAPI app will serve that build from `/` when it exists.

## API

The FastAPI app exposes:

- `GET /api/health`
- `GET /api/info`
- `POST /api/ingest`
- `POST /api/ask`
- `POST /api/eval`

## Why this scaffold works for the MVP

This version keeps the main chain simple and visible:

- `ingest`: reads files, chunks them, writes retrievable records
- `ask`: retrieves relevant chunks, builds a cited context, gets an answer
- `eval`: runs a batch of questions and records simple quality signals

That means you can validate architecture now, then upgrade individual layers later:

- swap local store -> Qdrant
- swap OpenAI provider -> Ollama provider
- swap single agent orchestration -> Agents SDK tool calling runtime
- expand eval metrics and tracing
