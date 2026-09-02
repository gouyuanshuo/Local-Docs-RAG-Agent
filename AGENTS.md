# Project Rules

This file is the main operating guide for working in this repository.

Use it together with:

- `spec.md`
- `tasks.md`
- `docs/development-roadmap.md`
- `skills/backend-api/SKILL.md`
- `skills/review-bugfix/SKILL.md`
- `.codex/config.toml`

## Required startup sequence

Before starting a new coding task:

1. Read `AGENTS.md`
2. Read `spec.md`
3. Read `tasks.md`
4. Check `docs/development-roadmap.md`
5. Open the relevant local skill if the task matches:
   - backend/API/runtime/retrieval work -> `skills/backend-api/SKILL.md`
   - debugging/review/fix/regression work -> `skills/review-bugfix/SKILL.md`

## Required close-out sequence

After making code changes:

1. Run the smallest meaningful verification path
2. Review the changed code for regressions or edge cases
3. Update `tasks.md`
   - mark completed items
   - add follow-up items if new work is discovered
4. If the work changes project direction or phase status, update `docs/development-roadmap.md`

## Project structure

The main repo layout is:

- `spec.md`
  - product and system requirements
- `tasks.md`
  - current task board and execution checklist
- `AGENTS.md`
  - operating rules for contributors and agents
- `.codex/config.toml`
  - local workflow preferences and behavior hints
- `skills/`
  - reusable local workflow templates
- `backend/`
  - Python backend
- `frontend/`
  - Vite + React + TypeScript frontend
- `docs/`
  - roadmap and supporting docs

## Engineering rules

- Follow the roadmap priority order unless there is a strong reason not to.
- Prefer measurable and explainable improvements before adding more surface area.
- Keep changes small and composable when possible.
- Do not treat fallback behavior as success; keep runtime state visible.
- Prefer updating existing modules over creating new top-level abstractions too early.
- Declare closed option sets once, in `backend/src/local_docs_rag_agent/constants.py`.
  Configuration parsing, API request validation, and CLI choices all derive from it;
  do not re-declare the same literal list in a second module.
- Import `rag` through its package facade (`local_docs_rag_agent.rag`) from outside the
  package, so its internal module split stays free to change.
- See `docs/architecture.md` for the per-layer boundaries and the extension recipes for
  adding a provider, a vector store, a runtime, a CLI command, or a config setting.

## Commit convention

Commits follow Conventional Commits:

```text
type(scope): short imperative description

Optional body explaining the reason and motivation for the change.

Optional footer for breaking changes or closed tasks.
```

Types in use: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

The header states what changed; the body states why. Prefer a scope that names the
affected area, such as `feat(rag):`, `fix(runtime):`, or `refactor(config):`.

## Task discipline

- `tasks.md` is the current source of truth for active work.
- Do not start a new implementation branch of work without checking `tasks.md`.
- When a task is complete, mark it complete in `tasks.md` in the same work session.
- If a task turns out to be bigger than expected, split it into smaller follow-up items.

## Skill routing

Use local skills as follows:

- `skills/backend-api/SKILL.md`
  - for backend endpoints, runtime wiring, providers, retrieval, eval, and config work
- `skills/review-bugfix/SKILL.md`
  - for debugging, self-review, regression checks, bugfix loops, and post-change validation

## Notes

- `.codex/config.toml` is a project-local workflow reference. Actual sandbox and approval behavior is still controlled by the Codex runtime.
- `docs/development-roadmap.md` is the long-horizon planning file.
- `tasks.md` is the short-horizon execution file.
