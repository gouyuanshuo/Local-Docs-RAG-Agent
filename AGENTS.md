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

## Code style

Python follows the [Google Python Style Guide][google-style]. The parts that are
mechanically enforced by `ruff check` and `ruff format` (see `pyproject.toml`) are:

- 80-column lines (3.2). `ruff format` holds code to it; comments and docstrings
  are wrapped by hand and `E501` still checks them.
- Module-only imports (2.2): `from x import y` where `y` is a module, then
  `y.Symbol` at the use site. Never import a class or function directly.
- Google docstrings (3.8) on every public module, class, and function, with a
  one-line summary and `Args:`, `Returns:`, and `Raises:` sections where they
  apply.

Four deviations are deliberate, and adding a fifth needs a reason in the commit:

- `typing` and `collections.abc` names are imported directly. The style guide
  exempts them, and `list[DocumentChunk]` reads better than the alternative.
- `qdrant_client` and `agents` are imported inside the functions that use them.
  The style guide permits a local import to break a dependency, and these are
  what keep the package installable without the optional extras.
- The `__init__.py` package facades re-export symbols. Re-exporting is what a
  facade is for, and the rule below requires outside code to reach `rag`
  through it.
- Tests are exempt from the docstring rules only. A test's name is its
  documentation, and a required one-line summary on each would add noise
  without adding information. Every other rule, the line limit included,
  applies to them.

Two ruff rules are deliberately not enabled: `DOC501` and `DOC502` read `raise`
statements literally, so they demand a `Raises:` entry for an exception that is
raised and caught inside the same function, and reject one for an exception
raised by a helper. This codebase does both, so `Raises:` sections are written
and reviewed by hand.

When a module import would shadow a common local name, bind it to an
unambiguous one: `config` is imported as `app_config`, `rag.base` and
`providers.base` as `rag_base` and `provider_base`, and `commands.eval` as
`eval_command`. A local variable must never shadow an imported module: it
raises `UnboundLocalError` at runtime and mypy does not report it.

[google-style]: https://google.github.io/styleguide/pyguide.html

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
