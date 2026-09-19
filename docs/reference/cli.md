# CLI Reference

The command-line interface `local-docs-rag` provides commands for ingestion,
querying, evaluation, and parametric comparison.

```bash
local-docs-rag <command> [options]
```

---

## Commands

### `ingest`
Scans documents under `DOCS_DIR`, generates chunks and embeddings, and brings
the configured store and ingest manifest up to date. The local store rewrites
the complete corpus; Qdrant applies changed, removed, and replaced source
transitions.

```bash
local-docs-rag ingest
```

### `ask`
Queries the RAG pipeline with a user prompt and outputs answer and citations.

```bash
local-docs-rag ask "What is reciprocal rank fusion?"
```

Supported options:
- `--runtime`: Override the execution runtime (`basic` or `agents_sdk`).

### `eval`
Runs the evaluation suite against gold questions defined in `EVAL_PATH`.

```bash
local-docs-rag eval
```

Supported options:
- `--runtime`: Override the execution runtime (`basic` or `agents_sdk`).

### `eval-compare`
Sweeps a multi-dimensional matrix of retrieval, chunking, and runtime
configurations to generate comparative benchmarks.

An omitted axis uses its `AXES` default. Runtime, reranker, `top_k`, chunk size,
and chunk overlap hold the configured value; chunk strategy sweeps all three
choices; retrieval strategy sweeps all four; and vector backend uses local plus
Qdrant only when `QDRANT_URL` is configured. The bare command therefore plans
12 local cells, or 24 with a configured Qdrant URL.

Planning is atomic with respect to execution: it resolves the entire Cartesian
product, rejects empty axes or more than 128 cells, constructs every validated
`AppConfig` variant, and preflights every executable storage target before gold
is loaded or any cell can ingest. A malformed configured Qdrant target aborts
the request without a partial report. An explicitly requested Qdrant cell with
no URL remains a planned `missing_qdrant_url` skip.

This command is not necessarily offline. Configured credentials can multiply
paid embedding, chat, and reranker calls by corpus, cases, and cells. Local
cells use temporary index/manifest pairs. Qdrant cells create UUID-named owned
disposable collections and attempt exact cleanup; inspect `run_metadata` for
orphan recovery if cleanup fails. With no provider keys, no Qdrant URL, and the
local backend, all 12 default cells use fallback and the leaderboard is empty
by design.

```bash
local-docs-rag eval-compare \
  --vector-backend local \
  --vector-backend qdrant \
  --retrieval-strategy blended \
  --retrieval-strategy hybrid_rrf
```

---

## Comparison Matrix Flags

The `eval-compare` command accepts repeatable flags corresponding to matrix
dimensions:

| Flag | Argument Type | Allowed Choices / Format | Description |
| --- | --- | --- | --- |
| `--runtime` | string | `basic`, `agents_sdk` | Target agent runtime to benchmark. |
| `--vector-backend` | string | `local`, `qdrant` | Vector backend storage engine. |
| `--chunk-strategy` | string | `markdown`, `paragraph`, `fixed` | Strategy used to segment source texts. |
| `--retrieval-strategy` | string | `blended`, `dense`, `lexical`, `hybrid_rrf` | Ranking and fusion strategy. |
| `--reranker` | string | `none`, `llm` | Second-stage reranker choice. |
| `--top-k` | integer | positive integer | Number of context chunks delivered to prompt. |
| `--chunk-size` | integer | positive integer | Character size target for document chunks. |
| `--chunk-overlap` | integer | non-negative integer | Character overlap between consecutive chunks. |
| `--output` | path | output JSON file path | Destination path for comparison report JSON (defaults to `data/evals/compare_latest.json`). |

The report is replaced atomically only after a complete report exists. It is a
generated run artifact; do not hand-edit it. The HTTP comparison endpoint
returns the same report shape but does not write this CLI output path.
