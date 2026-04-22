# Local Docs RAG Agent Spec

## Purpose

`Local Docs RAG Agent` is a small AI application system focused on:

- local/private document question answering
- retrieval with citations
- agent-style tool use
- evaluation and comparison
- provider flexibility

It is not a model training project.

Its core workflow is:

`Agent -> Tools -> RAG -> Citation -> Eval`

## System goals

The system should be able to:

1. read local documents
2. index and retrieve relevant knowledge
3. answer with citations and citation spans
4. expose tool-using behavior through multiple runtimes
5. compare retrieval and runtime configurations
6. support both cloud-compatible providers and future local providers

## Main components

### Frontend

- stack: `Vite + React + TypeScript`
- role:
  - ask questions
  - trigger ingest/eval/compare
  - display answers, citations, diagnostics, and compare results

### Backend

- stack: `FastAPI + Python package`
- role:
  - config loading
  - provider wiring
  - ingestion
  - retrieval
  - runtime dispatch
  - eval and comparison execution

## Retrieval requirements

The system should support:

- multiple chunk strategies
  - `fixed`
  - `paragraph`
  - `markdown`
- multiple retrieval backends
  - `local`
  - `qdrant`
- metadata-aware retrieval behavior
- document filtering through exclude patterns
- retrieval parameter comparison:
  - `top_k`
  - `chunk_size`
  - `chunk_overlap`

## Runtime requirements

The system currently supports:

- `basic`
- `agents_sdk`

The system should expose:

- requested runtime
- actual runtime
- fallback/degraded status
- provider mode and reason

## Eval requirements

The system should support:

- per-case eval results
- retrieval/source/citation/answer separation
- comparison across runtime and retrieval settings
- leaderboard-style summaries for quick inspection

## Near-term engineering direction

Use `docs/development-roadmap.md` for phase order.

Use `tasks.md` for the current execution queue.
