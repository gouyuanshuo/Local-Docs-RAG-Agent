# Module: delivery (api / cli / commands / presenters)

HTTP and CLI. Thin: read `AppConfig`, call one application function,
serialize with `presenters`. No retrieval policy here.

## Public interface

- `api/app.py` — FastAPI, CORS, error mapping, static `frontend/dist`
- `api/routes.py` — `/api/health|info|documents|ingest|ask|eval|eval/compare`
- `api/schemas.py` — request/response; option types from `core.constants`
- `cli.py` + `commands/` — subcommands via `set_defaults(handler=...)`
- `presenters.py` — **shared seam** with evals (JSON payloads)

Error mapping: `ConfigurationError`/`DataFormatError` → 400;
`ProviderUnavailableError`/`VectorStoreError` → 503.

Ask `question` max length 4000.

## Invariants

- Application modules must not import FastAPI, Pydantic, or argparse.
- Fallback answers still 200 **with** diagnostics; compare cells that
  fallback are `degraded` (evals), not a silent board win.
- Schema field names for compare axes match `comparison.AXES`.

## May change

- New route that calls an existing application function + presenter.
- New CLI command: handler in `commands/`, `_register_*` in `cli.py`,
  list it in `COMMAND_REGISTRARS`.

## Must not

- Re-declare option literals.
- Put ranking strategy into a prompt or a route-local retrieve call.
- Import `rag.qdrant_store` / `rag.pipeline` directly; use the rag facade
  or `LocalDocsAgent`.

## Depends on / depended by

- Depends on: evals, runtime, rag facade, config, core, presenters.
- Depended by: frontend (`/api/*` only).

## Tests

`tests/test_api.py`, `tests/test_cli.py`. Compare contract:
`tests/test_eval_comparison.py` (schema vs AXES).

## Parallel ownership

Own: `api/*`, `cli.py`, `commands/*`, `presenters.py`. Coordinate with
evals before changing presenter keys.

## New session

Root `AGENTS.md` + this file. Adjacent: `evals/AGENTS.md` for compare
payloads; `frontend/AGENTS.md` if the wire shape changes.
