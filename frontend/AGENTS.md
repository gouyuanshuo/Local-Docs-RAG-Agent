# Module: frontend

Vite + React + TypeScript. Delivery only: ask, ingest, eval, compare, show
citations and diagnostics. No retrieval policy in the browser.

## Public interface

- `src/App.tsx` — layout composition
- `src/hooks/useRagWorkspace.ts` — all client state
- `src/lib/api.ts` — `fetch` to `/api/*` (`VITE_API_BASE_URL`)
- `src/types/api.ts` — hand-synced wire types, including comparison axes,
  dataset/configuration identities, nullable Qdrant run metadata, retrieval
  snapshots, and complete eval summaries/results
- Panels under `src/components/`

## Invariants

- Talk only to `/api/*`. No API keys in the bundle.
- Ask diagnostics (requested/actual runtime, chat/embedding/rerank) render
  in the answer card, not only in raw JSON.
- Citation cards render the backend's answer-local `source_id`; a null ID is
  shown as `unlabelled`, never replaced with a positional label.
- Health, configuration, and document bootstrap results remain independent;
  only `/api/health` determines backend liveness.
- Operation and bootstrap failures render beside their owning controls with
  `role="alert"`; raw payloads remain available for deeper diagnostics.
- Compare `degraded` / `skipped` / `error` runs are visible; fallback cells
  must not look like leaderboard wins.
- Buttons disable while in flight (`isAsking` / `isActing`).

## May change

- Copy, layout, showing extra fields that the backend already returns.
- Keep `types/api.ts` in the same change as `api/schemas.py`.

## Must not

- Add history, upload, or a router unless that is the named Phase F task.
- Invent a second API client. Do not put ranking logic in the UI.
- Enable `vite.config.js` emit (`tsconfig.node.json` has `noEmit`).

## Depends on / depended by

- Depends on: backend HTTP contract only.
- Depended by: nothing else in-repo (static files served from `frontend/dist`).

## Tests

Vitest + React Testing Library behavior tests live beside the owning component
or hook and mock HTTP only at the `fetch` boundary. Gates:

```text
pnpm --filter local-docs-rag-agent-web test
pnpm run build
```

CI restores the frozen lock, runs behavior tests, then builds.

## Parallel ownership

Own: `frontend/src/**`, `frontend/tsconfig*.json`, `frontend/vite.config.ts`.
Coordinate with delivery if `types/api.ts` changes.

## New session

Root `AGENTS.md` + this file. Adjacent: `api/AGENTS.md` for payload fields.
