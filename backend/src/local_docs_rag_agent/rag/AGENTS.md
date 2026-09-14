# Module: rag

Discovery, chunking, ingest, storage, ranking, rerank. Outside this
package, import **only** `local_docs_rag_agent.rag` (the facade).

## Public interface (`rag/__init__.py`)

- `retrieve` / `build_reranker` — the only answer-path retrieval entry
- `ingest_documents` / `ensure_index` — lifecycle; `ensure_index` honors
  `needs_reindex`
- `build_store` / `retrieval_settings`
- `chunk_text`, `collect_document_paths`, `read_source_texts`
- `rank_chunks`, `RetrievalSettings`, `ChunkStore`, store types, `Reranker`

Internal layout (do not import from outside):

```text
discovery -> chunker -> ingest -> pipeline
pipeline -> store_factory -> local_store / qdrant_store / retrieval
retrieval -> bm25 / fusion / scoring
pipeline -> rerank -> llm_rerank
```

## Invariants

- Store does not know about rerank; reranker does not know the backend.
- Qdrant `blended` is the server's dense order (`blended ≡ dense`).
- Qdrant ingest/search refuse non-live embeddings.
- After delete-then-upsert failure, `needs_reindex` is set; `ensure_index`
  must restore even when checksums match.
- Default corpus is `data/corpus/sample`; nothing under `docs/` is indexed.

## May change

- Chunk / ranking internals behind the facade.
- New `Reranker` implementations (`none` / `llm` already exist).

## Must not

- Split `qdrant_store.py` unless that is the named task.
- Bypass `retrieve` for answers (tools, basic, eval all go through it).
- Let hash vectors into a live Qdrant collection.
- Import `runtime`, `evals`, or `api`.

## Depends on / depended by

- Depends on: `providers`, `config`, `core`.
- Depended by: runtime, evals, delivery, via the facade.

## Tests

`tests/test_chunker.py`, `test_ingest.py`, `test_local_store.py`,
`test_qdrant_store.py`, `test_retrieval.py`, `test_rerank.py`,
`test_import_graph.py`.

## Parallel ownership

Own: `rag/*` except do not casually rewrite `qdrant_store.py` structure.
Do not edit `runtime/` or `evals/` from a rag-only task.

## New session

Root `AGENTS.md` + this file. Open `qdrant_store.py` only if the task is
lifecycle or search on that backend.
