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
a leaderboard.

- **Request Body**:
  ```json
  {
    "vector_backends": ["local", "qdrant"],
    "retrieval_strategies": ["dense", "hybrid_rrf"]
  }
  ```
- **Response**: `200 OK`
  ```json
  {
    "num_runs": 4,
    "runs": [],
    "leaderboard": []
  }
  ```
