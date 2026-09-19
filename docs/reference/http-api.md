# HTTP API Reference

The backend exposes a RESTful HTTP service mounted at `/api`.

All endpoints accept and return JSON payloads. Handlers re-read configuration
on each invocation so one operation uses one validated snapshot. This is not a
hot-reload promise: restart the backend after changing its launching process
environment or `.env`. A running process cannot receive later shell exports,
and dotenv does not overwrite values it already loaded. Check effective
non-secret settings with `GET /api/info` after restart.

---

## Service Endpoints

### Health & Diagnostics

#### `GET /api/health`
Reports backend service availability and current UTC timestamp.

- **Response**: `200 OK`
  ```json
  {
    "status": "ok",
    "backend_time_utc": "2026-09-16T13:40:00.000000Z"
  }
  ```

#### `GET /api/info`
Returns runtime settings, provider specifications, vector backend status,
and current document count.

- **Response**: `200 OK`
  ```json
  {
    "name": "Local Docs RAG Agent",
    "runtime": "basic",
    "vector_backend": "local",
    "docs_dir": "data/corpus/sample",
    "docs_exclude_patterns": [],
    "docs_count": 10,
    "llm_provider": "openai",
    "llm_model": "gpt-4.1-mini",
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small",
    "top_k": 4,
    "retrieval_strategy": "blended",
    "reranker": "none",
    "chunk_strategy": "markdown",
    "chunk_size": 800,
    "chunk_overlap": 120,
    "qdrant_collection": "local-docs-rag",
    "external_http_trust_env": true
  }
  ```

---

## Documents & Ingestion

#### `GET /api/documents`
Lists discovered document paths available for indexing and searching.

Paths are the complete discovered paths returned by the configured `DOCS_DIR`,
not basenames. Relative defaults therefore include `data/corpus/sample/`;
absolute `DOCS_DIR` values produce absolute paths.

- **Response**: `200 OK`
  ```json
  {
    "count": 10,
    "documents": [
      "data/corpus/sample/cross_section.md",
      "data/corpus/sample/lecture5_attention.md",
      "data/corpus/sample/near_dup_canonical.md",
      "data/corpus/sample/near_dup_distractor.md",
      "data/corpus/sample/optimizer_adafactor.md",
      "data/corpus/sample/optimizer_noise.md",
      "data/corpus/sample/paraphrase_memory.md",
      "data/corpus/sample/qdrant_payload.md",
      "data/corpus/sample/rrf_dense_only.md",
      "data/corpus/sample/rrf_notes.md"
    ]
  }
  ```

#### `POST /api/ingest`
Brings the configured store and ingest manifest up to date from `DOCS_DIR`.
The local backend rewrites the full corpus; Qdrant applies incremental changed,
removed, and replaced-source transitions. `num_chunks` is the number written by
this run, which for an incremental backend may cover only changed documents.

- **Response**: `200 OK`
  ```json
  {
    "num_chunks": 42,
    "vector_backend": "local"
  }
  ```

---

## Retrieval & Question Answering

#### `POST /api/ask`
Submits a query to the RAG pipeline and returns generated answer, citations,
and retrieval diagnostics.

- **Request Body**:
  ```json
  {
    "question": "What is the attention mechanism?",
    "runtime": "basic"
  }
  ```
- **Response**: `200 OK`
  ```json
  {
    "question": "What is the attention mechanism?",
    "answer": "Provider fallback (missing_api_key); extractive answer: Attention computes a weighted combination of values.",
    "citations": ["data/corpus/sample/lecture5_attention.md"],
    "citation_spans": [
      {
        "source_path": "data/corpus/sample/lecture5_attention.md",
        "chunk_id": "data/corpus/sample/lecture5_attention.md::chunk-0",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 52,
        "text": "Attention computes a weighted combination of values.",
        "source_id": "S1"
      }
    ],
    "runtime": "basic",
    "diagnostics": {
      "requested_runtime": "basic",
      "actual_runtime": "basic",
      "vector_backend": "local",
      "chat_provider": {
        "provider": "openai",
        "mode": "fallback",
        "reason": "missing_api_key"
      },
      "embedding_provider": {
        "provider": "openai",
        "mode": "fallback",
        "reason": "missing_api_key"
      },
      "reranker": {
        "provider": "none",
        "mode": "ready",
        "reason": "reranker_disabled"
      }
    }
  }
  ```

`source_id` is the answer-local label shown to the model (`S1`, `S2`, ...),
not a stored chunk identifier. Legacy or otherwise unlabelled spans return
`null`. A fallback answer remains `200 OK`; inspect the runtime pair and every
provider `mode`/`reason` before treating it as live.

---

## Evaluation & Benchmarking

#### `POST /api/eval`
Evaluates the gold dataset at `EVAL_PATH` under the requested/configured
runtime. The response is one flat aggregate summary with its per-case `results`;
there is no nested `summary` object. This can call configured providers and
`ensure_index` can update stale configured storage, so it is offline only under
an explicit local no-key configuration.

- **Request Body**:
  ```json
  {
    "runtime": "basic"
  }
  ```
- **Response**: `200 OK`
  ```json
  {
    "num_cases": 1,
    "runtime": "basic",
    "retrieval_config": {
      "vector_backend": "local",
      "chunk_strategy": "markdown",
      "chunk_size": 800,
      "chunk_overlap": 120,
      "top_k": 4,
      "retrieval_strategy": "blended",
      "retrieval_candidate_k": 20,
      "rrf_k": 60,
      "reranker": "none",
      "rerank_candidate_k": 20,
      "docs_dir": "data/corpus/sample",
      "docs_exclude_patterns": []
    },
    "answer_keyword_hit_rate": 1.0,
    "retrieval_source_hit_rate": 1.0,
    "retrieval_span_hit_rate": 1.0,
    "retrieval_reciprocal_rank": 1.0,
    "retrieval_precision": 1.0,
    "citation_source_hit_rate": 1.0,
    "citation_span_hit_rate": 1.0,
    "avg_response_time_ms": 12.34,
    "avg_keyword_hit_rate": 1.0,
    "source_hit_rate": 1.0,
    "avg_citation_span_hit_rate": 1.0,
    "results": [
      {
        "question": "What is the attention mechanism?",
        "answer": "Provider fallback (missing_api_key); extractive answer: Attention computes a weighted combination of values.",
        "citations": ["data/corpus/sample/lecture5_attention.md"],
        "retrieved_sources": [
          "data/corpus/sample/lecture5_attention.md"
        ],
        "answer_keyword_hit_rate": 1.0,
        "retrieval_source_hit_rate": 1.0,
        "retrieval_span_hit_rate": 1.0,
        "retrieval_reciprocal_rank": 1.0,
        "retrieval_precision": 1.0,
        "citation_source_hit_rate": 1.0,
        "citation_span_hit_rate": 1.0,
        "response_time_ms": 12.34,
        "diagnostics": {
          "requested_runtime": "basic",
          "actual_runtime": "basic",
          "vector_backend": "local",
          "chat_provider": {
            "provider": "openai",
            "mode": "fallback",
            "reason": "missing_api_key"
          },
          "embedding_provider": {
            "provider": "openai",
            "mode": "fallback",
            "reason": "missing_api_key"
          },
          "reranker": {
            "provider": "none",
            "mode": "ready",
            "reason": "reranker_disabled"
          }
        },
        "expected_source_paths": [
          "data/corpus/sample/lecture5_attention.md"
        ],
        "expected_answer_keywords": ["weighted combination"],
        "expected_span_keywords": ["weighted combination of values"],
        "expected_retrieval_keywords": ["weighted combination of values"],
        "failure_reasons": []
      }
    ]
  }
  ```

#### `POST /api/eval/compare`
Executes an evaluation comparison matrix across multiple swept axes
(vector backends, chunk sizes, retrieval strategies, etc.) and returns
a leaderboard. An omitted axis uses its declared matrix default: runtime,
reranker, `top_k`, chunk size, and overlap hold configured values; all three
chunk strategies and all four retrieval strategies sweep; and Qdrant joins
local only when `QDRANT_URL` is configured. An empty body therefore plans 12
local cells, or 24 with a configured Qdrant URL.

The service resolves the entire Cartesian product, rejects empty axes or more
than 128 cells, constructs and validates every configuration, and preflights
every executable storage target before loading gold or allowing any cell to
ingest. Invalid planning returns a request-level domain error with no partial
run. Provider/runtime failures after planning remain per-cell outcomes.

Configured credentials can make this a paid operation: embedding, chat, and a
selected reranker can run across every applicable corpus/case/cell. Local cells
use temporary storage. Qdrant cells create owned UUID-named disposable
collections and attempt exact cleanup. The HTTP endpoint returns the report but
does not write the CLI's `data/evals/compare_latest.json` file.

- **Request Body**:
  ```json
  {
    "runtimes": ["basic"],
    "chunk_strategies": ["markdown"],
    "vector_backends": ["qdrant"],
    "top_ks": [4],
    "chunk_sizes": [800],
    "chunk_overlaps": [120],
    "retrieval_strategies": ["blended"],
    "rerankers": ["none"]
  }
  ```
- **Response**: `200 OK`
  ```json
  {
    "num_runs": 1,
    "dataset_identity": "<stable-corpus-and-gold-digest>",
    "runtimes": ["basic"],
    "chunk_strategies": ["markdown"],
    "vector_backends": ["qdrant"],
    "top_ks": [4],
    "chunk_sizes": [800],
    "chunk_overlaps": [120],
    "retrieval_strategies": ["blended"],
    "rerankers": ["none"],
    "runs": [
      {
        "label": "basic:qdrant:markdown:blended:rr-none:k4:s800:o120",
        "status": "error",
        "dataset_identity": "<stable-corpus-and-gold-digest>",
        "configuration_identity": "<semantic-configuration-digest>",
        "run_metadata": {
          "disposable_qdrant_collection": "local-docs-rag-eval-<uuid>",
          "orphan_recovery_required": true,
          "pre_cleanup_status": "error",
          "cleanup_error": "RuntimeError: cleanup failure",
          "prior_error": "RuntimeError: eval failure"
        },
        "reason": null,
        "error": "Qdrant comparison cleanup failed for disposable collection local-docs-rag-eval-<uuid>: RuntimeError: cleanup failure; original cell failure: RuntimeError: eval failure",
        "retrieval_config": {
          "vector_backend": "qdrant",
          "chunk_strategy": "markdown",
          "chunk_size": 800,
          "chunk_overlap": 120,
          "top_k": 4,
          "retrieval_strategy": "blended",
          "retrieval_candidate_k": 20,
          "rrf_k": 60,
          "reranker": "none",
          "rerank_candidate_k": 20,
          "docs_dir": "data/corpus/sample",
          "docs_exclude_patterns": []
        },
        "runtime": "basic",
        "summary": null
      }
    ],
    "leaderboard": []
  }
  ```

`dataset_identity` identifies the corpus and gold data without including
document bodies or temporary storage paths. Every run repeats that identity
and adds a `configuration_identity` for its semantic runtime, retrieval, and
provider settings. Qdrant runs also expose lifecycle `run_metadata`; when
cleanup fails, the exact disposable collection and
`orphan_recovery_required: true` make manual recovery possible without
guessing at a collection name. Local runs omit `run_metadata` in the raw
report and return it as `null` through the typed API model.

## Error Responses

Request-schema validation is FastAPI's standard `422 Unprocessable Entity`
shape, not the domain-error shape below. For example, a blank ask question is
trimmed and rejected:

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "question"],
      "msg": "String should have at least 1 character",
      "input": "   ",
      "ctx": {"min_length": 1}
    }
  ]
}
```

Expected domain failure responses always contain `code`, `detail`, and nullable
`action_hint`:

```json
{
  "code": "invalid_configuration",
  "detail": "DOCS_DIR does not exist: missing",
  "action_hint": "Create the directory or set DOCS_DIR to an existing directory."
}
```

| Status | Domain exception | Stable code |
| --- | --- | --- |
| `400 Bad Request` | `ConfigurationError` | `invalid_configuration` |
| `400 Bad Request` | `DataFormatError` | `invalid_data_format` |
| `503 Service Unavailable` | `ProviderUnavailableError` | `provider_unavailable` |
| `503 Service Unavailable` | `VectorStoreError` | `vector_store_<reason_code>` |
| `500 Internal Server Error` | another expected `LocalDocsError` | `local_docs_error` or the subclass code |

Unexpected programming exceptions are not converted into this payload.
Fallback answers are also not errors: `/api/ask` returns 200 with fallback
diagnostics. Comparison execution failures remain 200 report responses with
`degraded`, `skipped`, or `error` cells; only request/schema/planning failures
use HTTP errors.
