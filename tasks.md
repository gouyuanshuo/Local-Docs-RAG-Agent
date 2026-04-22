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

- `Phase C: Retrieval engineering reinforcement`

Current theme:

- make retrieval quality easier to compare, understand, and tune

## Active tasks

### Phase C

- [x] Add configurable chunk strategies
- [x] Add document filtering and retrieval config snapshots
- [x] Add eval comparison matrix in CLI
- [x] Add compare API and frontend leaderboard
- [x] Add retrieval parameter comparison for `top_k`, `chunk_size`, and `chunk_overlap`
- [ ] Run and analyze a stronger `local vs qdrant` comparison with a stable reachable Qdrant backend
- [ ] Decide whether Phase C is complete enough to move primary focus to Phase D

### Phase D preparation

- [ ] Design incremental Qdrant ingest behavior
- [ ] Define checksum/manifest strategy for document-level updates
- [ ] Define stale document deletion behavior

## Review notes

Use `skills/review-bugfix/SKILL.md` after meaningful code changes, especially when:

- runtime behavior changed
- retrieval logic changed
- eval output shape changed
- API payloads changed
