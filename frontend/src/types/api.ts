export type RuntimeName = "basic" | "agents_sdk";
export type RuntimeSelection = "" | RuntimeName;

export type ProviderMode = "ready" | "live" | "fallback" | "unknown";

export type ProviderStatus = {
  provider: string;
  mode: ProviderMode;
  reason: string | null;
};

export type CitationSpan = {
  source_path: string;
  chunk_id: string;
  chunk_index: number;
  start_char: number;
  end_char: number;
  text: string;
};

export type AnswerDiagnostics = {
  requested_runtime: RuntimeName;
  actual_runtime: RuntimeName;
  vector_backend: "local" | "qdrant";
  chat_provider: ProviderStatus;
  embedding_provider: ProviderStatus;
};

export type AskResponse = {
  question: string;
  answer: string;
  citations: string[];
  citation_spans: CitationSpan[];
  runtime: RuntimeName;
  diagnostics: AnswerDiagnostics;
};

export type AppInfo = {
  name: string;
  runtime: RuntimeName;
  vector_backend: "local" | "qdrant";
  docs_dir: string;
  docs_exclude_patterns: string[];
  docs_count: number;
  llm_provider: string;
  llm_model: string;
  embedding_provider: string;
  embedding_model: string;
  top_k: number;
  retrieval_strategy: "blended" | "dense" | "lexical" | "hybrid_rrf";
  chunk_strategy: "fixed" | "paragraph" | "markdown";
  chunk_size: number;
  chunk_overlap: number;
  qdrant_collection: string;
  external_http_trust_env: boolean;
};

export type Health = {
  status: "ok";
  backend_time_utc: string;
};

export type DocumentsResponse = {
  count: number;
  documents: string[];
};

export type CompareLeaderboardRow = {
  label: string;
  answer_keyword_hit_rate: number;
  retrieval_source_hit_rate: number;
  retrieval_span_hit_rate: number;
  citation_span_hit_rate: number;
  avg_response_time_ms: number;
};

export type CompareRun = {
  label: string;
  status: "ok" | "skipped" | "error";
  reason?: string;
  error?: string;
};

export type CompareResponse = {
  num_runs: number;
  runtimes: RuntimeName[];
  chunk_strategies: string[];
  vector_backends: string[];
  leaderboard: CompareLeaderboardRow[];
  runs: CompareRun[];
};

export type ApiErrorPayload = {
  code?: string;
  detail?: unknown;
  action_hint?: string | null;
};
