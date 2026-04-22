---
name: backend-api
description: Build and update backend APIs, runtime wiring, retrieval flows, provider integrations, config handling, and eval plumbing in this repository. Use when the task touches FastAPI endpoints, Python backend modules, retrieval engineering, runtime dispatch, ingestion, provider setup, or backend-facing compare/eval features.
---

# Backend API Skill

Use this skill for backend and retrieval work in this repository.

## Required pre-check

Before changing code:

1. Read `AGENTS.md`
2. Read `spec.md`
3. Read `tasks.md`
4. Check `docs/development-roadmap.md`

## Default workflow

1. Identify the current phase and active task in `tasks.md`
2. Confirm the backend area being changed:
   - `api/`
   - `commands/`
   - `evals/`
   - `providers/`
   - `rag/`
   - `runtime/`
3. Prefer extending the existing module structure before adding new top-level abstractions
4. Keep request/response payloads explicit through schemas or presenters
5. Run the smallest meaningful verification path after changes
6. Update `tasks.md` before closing the task

## Repository-specific reminders

- Backend package root: `backend/src/local_docs_rag_agent/`
- Current active architectural priorities:
  - measurable
  - explainable
  - reliable
- For retrieval work, keep configuration visible in outputs whenever possible
- For runtime work, do not hide fallback or degraded behavior

## Verification patterns

Use one or more of these depending on the change:

- `local-docs-rag ingest`
- `local-docs-rag ask "..."`
- `local-docs-rag eval`
- `local-docs-rag eval-compare`
- targeted API verification through FastAPI test paths or route calls

## Close-out

After implementation:

1. Review the changed code for regressions
2. Confirm task completion in `tasks.md`
3. If the work advanced the phase meaningfully, update `docs/development-roadmap.md`
