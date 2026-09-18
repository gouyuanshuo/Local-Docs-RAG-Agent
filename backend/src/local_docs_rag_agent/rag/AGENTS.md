# Module: rag

Discovery, chunking, ingest, storage, ranking, rerank. Outside this
package, import **only** `local_docs_rag_agent.rag` (the facade).

## Public interface (`rag/__init__.py`)

- `retrieve` / `build_reranker` — the only answer-path retrieval entry
- `ingest_documents` / `ensure_index` — lifecycle; `ensure_index` honors
  `needs_reindex`
- `index_guard` — same-user, single-host lifecycle critical section
- `initialize_owned_qdrant_index` / `delete_owned_qdrant_index` — atomic
  first-ingest and fail-closed cleanup for disposable Qdrant collections
- `qdrant_orphaned_collection` — structured exact-name recovery signal on an
  exception escaping interrupted Qdrant initialization
- `validate_storage_target` — pure target-name/URL validation with no
  filesystem inspection; a missing Qdrant URL is invalid to this seam
- `build_store` / `retrieval_settings`
- `chunk_text`, `collect_document_paths`, `read_source_texts`
- `rank_chunks`, `RetrievalSettings`, `ChunkStore`, store types, `Reranker`

Internal layout (do not import from outside):

```text
discovery -> chunker -> ingest -> pipeline
pipeline -> store_factory -> local_store / qdrant_store / retrieval
retrieval -> bm25 / fusion / scoring
pipeline -> rerank -> llm_rerank
index_lock -> config, core (and lazy storage identity lookup)
```

## Invariants

- Store does not know about rerank; reranker does not know the backend.
- `ChunkStore.exists()` means the configured index target is ready to inspect;
  `supports_incremental_updates` selects changed/removed-source saves; and
  `requires_live_embeddings` rejects fallback vectors before mutation.
- Generic ingest and readiness decisions use those capabilities, never a
  backend name or concrete store type. Factory selection and Qdrant ownership
  remain explicitly backend-specific.
- Qdrant `blended` is the server's dense order (`blended ≡ dense`).
- Qdrant ingest/search refuse non-live embeddings.
- After delete-then-upsert failure, `needs_reindex` is set; `ensure_index`
  must restore even when checksums match.
- Every store save follows durable `repair_required` publication. A clean
  manifest is published only after the store mutation succeeds.
- Ingest and retrieval readiness/query lock both the canonical manifest and
  storage target in global order under one deadline. Sharing either resource
  serializes; configurations sharing neither remain concurrent. Reranking runs
  after both resources are released.
- After `fork`, child lock/ownership registries are reset and inherited lock
  descriptors are closed without unlocking the parent's guard. Contention on
  one target does not serialize an unrelated target.
- An interruption between thread/OS lock acquisition and bookkeeping rolls
  back the unpublished acquisition without replacing the active exception.
- Same-thread recursion increments only logical guard depth and does not
  reacquire the `RLock`, so a pre-publication nested interruption cannot
  release ownership held by the outer guard.
- Logical depth, thread locks, file locks, and registry retention are restored
  idempotently after release interruptions. Per-frame retention tokens prevent
  a cleanup retry from decrementing another guard's ownership.
- A resource frame retains physical locks and its token until exact logical
  depth restoration, then retains the token until file- and thread-lock
  cleanup obligations are conclusively discharged. Internal registry mutexes
  likewise retain per-acquisition cleanup obligations across a failed
  fallback. The complete depth/phase/physical cleanup transition retries once
  after an interruption; a protected-body exception remains authoritative and
  never triggers an unrelated extra mutex release.
- Nested guards may reenter the same resource set. Partial overlap is rejected
  before retention, and disjoint nested sets must preserve global resource
  order; callers should normally exit one configuration's guard before
  entering another.
- Manifest and local-store lock artifacts live beside their resolved files.
  Qdrant store artifacts live in the per-user cache and contain only a hashed,
  credential-free identity. Persistent lock files are harmless.
- Host-local guards do not coordinate Qdrant lifecycle work across hosts;
  multi-host deployments require external coordination.
- The versioned ingest manifest binds source ownership to a hashed storage
  identity. A known target mismatch must fail before store mutation.
- Dirty source paths remain deletion-owned until a successful repair removes
  or republishes them, even if they leave document scope before recovery.
- Qdrant ownership initialization holds the guard through absence check and
  first ingest. It returns a cleanup token only after collection creation.
- An initialization `BaseException` discards its claim and attempts exact
  cleanup. Cleanup failure never replaces the original; it adds an exact-name
  note and structured `qdrant_orphaned_collection` marker.
- Qdrant storage identity uses effective REST port `6333` when the URL omits a
  port. Explicit ports, including `80` and `443`, remain distinct targets.
- Qdrant endpoint parsing reports a fixed credential-free configuration error;
  IPv6 literals are compressed and port `0` is invalid.
- Qdrant operation errors omit raw URLs and exception messages and suppress
  unsafe chaining while retaining operation, collection, exception class,
  reason code, and recovery hints.
- Explicitly marked internal Qdrant validation diagnostics bypass raw-client
  normalization unchanged; arbitrary client/domain exceptions do not.
- The local store resolves its bound index path once before I/O, matching the
  identity path and preserving a final-component symlink during atomic writes.
- The manifest target is likewise resolved before locking and I/O, preserving
  existing or dangling final-component symlinks. A local index and manifest
  may never resolve to the same file.
- The index fingerprint includes the resolved document root and sorted unique
  exclusion patterns. Same-target scope changes rebuild while retaining the
  prior source list for stale-source deletion.
- Legacy local manifests rebuild and acquire an identity. Legacy Qdrant
  manifests require an explicit ownership decision before adoption.
- Default corpus is `data/corpus/sample`; nothing under `docs/` is indexed.

## May change

- Chunk / ranking internals behind the facade.
- New `Reranker` implementations (`none` / `llm` already exist).
- New `ChunkStore` implementations that declare all lifecycle capabilities,
  are constructed only by `store_factory`, and preserve manifest semantics.

## Must not

- Split `qdrant_store.py` unless that is the named task.
- Bypass `retrieve` for answers (tools, basic, eval all go through it).
- Let hash vectors into a live Qdrant collection.
- Import `runtime`, `evals`, or `api`.

## Depends on / depended by

- Depends on: `providers`, `config`, `core`.
- Depended by: runtime, evals, delivery, via the facade.

## Tests

`tests/test_chunker.py`, `test_index_lifecycle.py`, `test_ingest.py`,
`test_local_store.py`,
`test_qdrant_store.py`, `test_retrieval.py`, `test_rerank.py`,
`test_import_graph.py`.

## Parallel ownership

Own: `rag/*` except do not casually rewrite `qdrant_store.py` structure.
Do not edit `runtime/` or `evals/` from a rag-only task.

## New session

Root `AGENTS.md` + this file. Open `qdrant_store.py` only if the task is
lifecycle or search on that backend.
