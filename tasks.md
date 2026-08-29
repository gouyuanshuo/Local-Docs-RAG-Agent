# Current Tasks

This file is the short-horizon task board for the repository.

## Operating checklist

Before coding:

- [ ] Read `AGENTS.md`
- [ ] Read `spec.md`
- [ ] Read `tasks.md`
- [ ] Check `docs/development-roadmap.md`
- [ ] Open the relevant local skill if needed

After coding:

- [ ] Run a relevant verification path
- [ ] Review the changed code
- [ ] Mark completed tasks below
- [ ] Add follow-up tasks if new work appears

## Current focus

Current active phase:

- `Phase D: Qdrant engineering`

Current theme:

- make Qdrant indexing lifecycle stable and maintainable

## Active tasks

### Phase C

- [x] Add configurable chunk strategies
- [x] Add document filtering and retrieval config snapshots
- [x] Add eval comparison matrix in CLI
- [x] Add compare API and frontend leaderboard
- [x] Add retrieval parameter comparison for `top_k`, `chunk_size`, and `chunk_overlap`
- [x] Run and analyze a stronger `local vs qdrant` comparison with a stable reachable Qdrant backend
- [x] Resolve current Qdrant connectivity issue blocking real compare runs (`qdrant_unreachable`)
- [x] Decide whether Phase C is complete enough to move primary focus to Phase D
- [ ] Optional: run broader retrieval tuning sweeps (more questions and larger docs) if needed

### Phase D preparation

- [x] Design incremental Qdrant ingest behavior
- [x] Define checksum/manifest strategy for document-level updates
- [x] Define stale document deletion behavior
- [x] Improve Qdrant ingest error reporting when embedding provider is fallback/transient
- [x] Add optional backoff/retry policy before failing Qdrant ingest on embedding instability
- [x] Normalize Qdrant network-unreachable errors into actionable ingest diagnostics
- [x] Add `source_path` payload index bootstrap for Qdrant delete filters
- [x] Add configurable Qdrant client timeout (`QDRANT_TIMEOUT_S`)
- [x] Add configurable embedding request batching (`EMBEDDING_BATCH_SIZE`)
- [x] Normalize proxy-env handling for external providers (`HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`) in dev docs/bootstrap
- [x] Add focused automated tests for embedding batching/retry and incremental Qdrant lifecycle behavior
- [ ] Re-run live `qdrant ingest -> ask -> eval` from a network path that can reach the configured cloud endpoint (current direct check resets with `WinError 10054`)

## Review notes

Use `skills/review-bugfix/SKILL.md` after meaningful code changes, especially when:

- runtime behavior changed
- retrieval logic changed
- eval output shape changed
- API payloads changed
