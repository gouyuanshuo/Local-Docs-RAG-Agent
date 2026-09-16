# CLI Reference

The command-line interface `local-docs-rag` provides commands for ingestion,
querying, evaluation, and parametric comparison.

```bash
local-docs-rag <command> [options]
```

---

## Commands

### `ingest`
Scans documents under `DOCS_DIR`, generates chunks and embeddings, and updates
the index store.

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
