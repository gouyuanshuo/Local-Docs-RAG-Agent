# Project Rules

This is the **router** for contributors and coding agents. Do not read the
whole repository at the start of a task. Read this file, open the **one**
module contract the task belongs to, then code.

Product requirements: `spec.md`. Execution queue: `tasks.md`. Documentation
index: `docs/README.md`. System diagram: `docs/design/architecture.md`.
Extension recipes: `docs/development/extending.md`. Long-horizon phases:
`docs/planning/development-roadmap.md`.

## Startup (do this, not a full-repo dump)

1. Read this file.
2. Skim `tasks.md` current focus only.
3. Open the module `AGENTS.md` for the work (table below).
4. Open at most one adjacent contract if you must change a shared seam.
5. Open `skills/backend-api/SKILL.md` or `skills/review-bugfix/SKILL.md` when
   the task matches.

Do **not** start by reading every module contract, `qdrant_store.py`, or the
roadmap end-to-end.

## Module map

| If the task is about… | Read |
| --- | --- |
| option sets, domain records, errors, atomic file write, env readers | `backend/src/local_docs_rag_agent/core/AGENTS.md` |
| `AppConfig`, `.env` mapping, `with_overrides` | `backend/src/local_docs_rag_agent/config/AGENTS.md` |
| chat/embedding clients, factory, fallback status | `backend/src/local_docs_rag_agent/providers/AGENTS.md` |
| ingest, retrieve, chunk, stores, rerank | `backend/src/local_docs_rag_agent/rag/AGENTS.md` |
| answer runtimes, tools, citation blocks | `backend/src/local_docs_rag_agent/runtime/AGENTS.md` |
| eval cases, metrics, compare matrix, leaderboard | `backend/src/local_docs_rag_agent/evals/AGENTS.md` |
| HTTP, CLI, presenters | `backend/src/local_docs_rag_agent/api/AGENTS.md` |
| React UI, `types/api.ts` | `frontend/AGENTS.md` |

Dependency direction (down only):

```text
delivery -> evals, runtime, rag facade, providers, config, core
runtime  -> rag facade, providers, config, core
evals    -> runtime, rag facade, presenters, config, core
rag      -> providers, config, core
providers-> config, core
config   -> core
core     -> (nothing else in this package)
```

Outside `rag/`, import retrieval through `local_docs_rag_agent.rag` (the
package facade). Do not `from local_docs_rag_agent.rag.<module> import …`
in production code. Tests may import internals.

## Parallel subagents

- One subagent owns **one module** plus that module's tests.
- `core` and `config` are frozen unless the task names them.
- Shared seams (`core` models/constants, `presenters.py`): one owner at a
  time; changing them is a cross-module change — update the contract first.
- Do not two-at-once edit `presenters.py`, `core/models.py`, or
  `core/constants.py`.

## Close-out

1. Run the tests named in the module contract, then the quality gates below
   if the change can leak.
2. Review regressions at the module seam, not private helpers.
3. Update `tasks.md`. Update the roadmap only if phase status changed.
4. If you changed a public symbol, update that module's `AGENTS.md` in the
   same session.

## Engineering rules

- Measurable > explainable > reliable > more agent-like > more product-like.
- Do not treat fallback as success; keep `ProviderStatus` visible.
- Prefer extending an existing module over a new top-level package.
- Closed option sets live only in `core/constants.py`.
- See `docs/development/extending.md` to add a provider, store, runtime, CLI command,
  or config field.

## Code style

Python follows the [Google Python Style Guide][google-style], enforced by
`ruff` / `ruff format` (`pyproject.toml`):

- 80-column lines. Wrap comments and docstrings by hand; `E501` still checks
  them.
- Module-only imports: `from x import y` where `y` is a module, then
  `y.Symbol`. Never import a class or function directly.
- Google docstrings on every public module, class, and function.

Deliberate exceptions (adding a fifth needs a reason in the commit):

- `typing` and `collections.abc` names import directly.
- `qdrant_client` and `agents` import inside the functions that use them.
- Package facades re-export symbols.
- Tests are exempt from docstring rules only.

`DOC501` / `DOC502` stay off; write `Raises:` by hand.

Import aliases: `config` as `app_config`, `rag.base` as `rag_base`,
`providers.base` as `provider_base`, `commands.eval` as `eval_command`. A
local variable must never shadow an imported module.

[google-style]: https://google.github.io/styleguide/pyguide.html

## Quality gates

```text
python -m ruff check backend/src tests scripts
python -m ruff format --check backend/src tests scripts
python -m mypy
python -m pytest
python -m compileall -q backend/src
```

Frontend changes: `pnpm run build`. CI matches this (plus `DOC201` preview on
`backend/src` and `scripts`).

## Commit convention

```text
type(scope): short imperative description
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`. Scope is
the module: `feat(rag):`, `fix(runtime):`, `refactor(core):`.

## Do not do in a random session

Do not split `rag/qdrant_store.py` or `AppConfig` into sub-configs unless
that is the named task. Do not add CacheManager, StoreService, or a second
constants list. Do not treat a no-key compare leaderboard as a quality
ranking.

## Notes

- `.codex/config.toml` is a local workflow hint; the runtime still owns
  sandbox and approval.
- `tasks.md` is the short-horizon board. Module contracts are how to
  implement without a prior chat.
