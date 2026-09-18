# HTTP API Reference

The backend exposes a RESTful HTTP service mounted at `/api`.

All endpoints accept and return JSON payloads. Handlers re-read configuration
on each invocation to ensure environment overrides and parameter flags are
observed without requiring server restarts.

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

- **Response**: `200 OK`
  ```json
  {
    "count": 10,
    "documents": ["cross_section.md", "lecture5_attention.md"]
  }
  ```

#### `POST /api/ingest`
Triggers full ingestion and vector indexing across documents in `DOCS_DIR`.

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
    "answer": "...",
    "citations": [],
    "retrieved_sources": [],
    "diagnostics": {}
  }
  ```

---

## Evaluation & Benchmarking

#### `POST /api/eval`
Executes offline evaluation over gold test dataset at `EVAL_PATH`.

- **Request Body**:
  ```json
  {
    "runtime": "basic"
  }
  ```
- **Response**: `200 OK`
  ```json
  {
    "summary": {},
    "results": []
  }
  ```

#### `POST /api/eval/compare`
Executes an evaluation comparison matrix across multiple swept axes
(vector backends, chunk sizes, retrieval strategies, etc.) and returns
a leaderboard. Default requests plan 12 local cells (three chunk strategies by
four retrieval strategies), or 24 when Qdrant is configured. The service
validates every planned configuration before any cell ingests; invalid input is
a request error, while provider and runtime failures remain per-cell outcomes.

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
