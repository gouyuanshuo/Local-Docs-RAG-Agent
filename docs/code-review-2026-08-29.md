# Deep Code Review - 2026-08-29

This review covered every Python file under `backend/src`, `tests`, and `scripts`, plus
the API/frontend contract and repository structure. Line anchors below refer to the
post-refactor tree on this date.

## Baseline evidence

- Git baseline: `3c9e342 feat: harden qdrant lifecycle and diagnostics`
- Tests before refactor: 17 passing
- Coverage before refactor: 52%
- Ruff audit: 142 findings under `E,F,I,UP,B,SIM,RUF`
- mypy strict baseline: 46 errors across 8 files

## Resolved high-risk findings

| Severity | Finding | Resolution |
| --- | --- | --- |
| Critical | `chunk_overlap >= chunk_size` could make fixed chunking loop forever | Configuration and public chunker validation at `config.py:99` and `rag/chunker.py:32` |
| Critical | Missing `DOCS_DIR` could silently replace the index/manifest with empty state | Directory validation before mutation at `rag/ingest.py:199` |
| Critical | A changed document becoming empty left old Qdrant chunks behind | Replacement plan uses changed source paths at `rag/ingest.py:61` |
| Critical | Qdrant search could send a 128D hash fallback vector to a live-model collection | Live-provider guard at `rag/qdrant_store.py:135` |
| High | An existing index remained "ready" after embedding/chunk configuration changed | Manifest index fingerprint and rebuild check in `rag/ingest.py` |
| High | Qdrant `load()` truncated collections at 10,000 records | Offset pagination at `rag/qdrant_store.py:113` |
| High | Missing/inconsistent embeddings were silently omitted during Qdrant save | Full vector validation at `rag/qdrant_store.py:51` |
| High | Local index and manifest writes were interruptible in place | Atomic sibling-file replacement in `rag/file_io.py` |
| High | Agents SDK mutated a process-global OpenAI client and ignored API style | Per-run client/model selection at `runtime/agents_sdk.py:34` and model builder |
| High | API endpoints exposed unbounded strings and inconsistent exception responses | Typed schemas at `api/schemas.py:15` and error mapping at `api/app.py:75` |
| High | Eval matrix input could create an unbounded Cartesian product of runs | Per-axis request limits and a 128-run application guard in `evals/comparison.py` |
| Medium | Malformed chunks, manifests, and eval JSON produced loose/raw errors | Typed parsing in `models.py`, `rag/manifest.py:36`, and `evals/harness.py:47` |
| Medium | Empty eval expectations counted as metric failures | Neutral metric semantics at `evals/harness.py:71` |
| Medium | Fallback answers discarded multiline chunk content | Multiline context extraction in `providers/chat.py` |
| Medium | Store, scoring, Qdrant, and citation concerns lived in one large module | Split into `store.py`, `qdrant_store.py`, and `scoring.py` |
| Medium | API construction, routes, and schemas were coupled in one module | Split application assembly and `api/routes.py:29` |
| Low | Four compatibility modules had no callers and duplicated import paths | Removed obsolete runtime/provider/embedding wrappers |

## Module-level review outcome

- `config.py`: immutable configuration, typed env parsing, invariant validation, safe overrides
- `models.py`: explicit persisted-payload validation instead of unchecked `**payload`
- `providers/`: shared client/error policy; truthful fallback reasons; request-owned async client
- `rag/chunker.py`: stable original-source offsets, parameter guards, oversized paragraph splitting
- `rag/ingest.py`: explicit ingest plan and mutation ordering
- `rag/store.py`: atomic local persistence and isolated local scoring dependency
- `rag/qdrant_store.py`: lifecycle validation, pagination, live-vector guard, normalized failures
- `evals/`: strict case loading, neutral optional expectations, flatter matrix orchestration
- `api/`: typed contracts, thin routes, framework assembly isolated from domain logic
- `commands/` and `cli.py`: domain errors now produce a controlled exit instead of traceback
- `tests/`: lifecycle, corruption, validation, pagination, offset, and API failure regressions added
- `frontend/`: contract checked; structural split deferred because this pass targeted Python

## Remaining risks and follow-ups

1. Qdrant source delete followed by upsert is not a cross-operation transaction; a remote
   failure between them can temporarily remove a changed source until the next ingest.
2. Live `qdrant ingest -> ask -> eval` remains network-dependent and should be repeated
   from a path that does not reset the remote connection.
3. Agents SDK tool behavior needs a deterministic integration test with a fake runner.
4. `frontend/src/App.tsx` is about 500 lines and should be split into API types/client,
   feature hooks, and presentation components.
5. The quality gates exist locally but are not yet enforced by CI.

## Post-refactor evidence

- Ruff: zero findings
- mypy strict: zero errors across 40 source files
- Tests: 40 passing
- Coverage: 68%
