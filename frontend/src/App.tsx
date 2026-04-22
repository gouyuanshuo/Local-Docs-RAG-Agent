import { useEffect, useState } from "react";

type Runtime = "" | "basic" | "agents_sdk";

type AskResponse = {
  question: string;
  answer: string;
  citations: string[];
  citation_spans: CitationSpan[];
  runtime: string;
  diagnostics: AnswerDiagnostics;
};

type CitationSpan = {
  source_path: string;
  chunk_id: string;
  chunk_index: number;
  start_char: number;
  end_char: number;
  text: string;
};

type AppInfo = {
  name: string;
  runtime: string;
  vector_backend: string;
  docs_dir: string;
  docs_count: number;
  llm_provider: string;
  llm_model: string;
  embedding_provider: string;
  embedding_model: string;
  top_k: number;
  chunk_strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  qdrant_collection: string;
};

type Health = {
  status: string;
  backend_time_utc: string;
};

type DocumentsResponse = {
  count: number;
  documents: string[];
};

type ProviderStatus = {
  provider: string;
  mode: string;
  reason: string | null;
};

type AnswerDiagnostics = {
  requested_runtime: string;
  actual_runtime: string;
  vector_backend: string;
  chat_provider: ProviderStatus;
  embedding_provider: ProviderStatus;
};

const apiBaseUrl =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }

  return (await response.json()) as T;
}

function pretty(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function relativeTime(isoString: string): string {
  const timestamp = new Date(isoString).getTime();
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

function fileLabel(path: string): string {
  const parts = path.split("/");
  return parts[parts.length - 1] ?? path;
}

function excerpt(text: string, limit = 220): string {
  if (text.length <= limit) return text;
  return `${text.slice(0, limit).trimEnd()}...`;
}

function providerLabel(status: ProviderStatus): string {
  return `${status.provider} · ${status.mode}`;
}

export default function App() {
  const [runtime, setRuntime] = useState<Runtime>("");
  const [question, setQuestion] = useState("How is attention explained in lecture 5?");
  const [health, setHealth] = useState<Health | null>(null);
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [documents, setDocuments] = useState<string[]>([]);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const [askResult, setAskResult] = useState<AskResponse | null>(null);
  const [askRaw, setAskRaw] = useState("Waiting for a question...");
  const [actionRaw, setActionRaw] = useState("System actions will appear here...");
  const [isAsking, setIsAsking] = useState(false);
  const [isActing, setIsActing] = useState(false);

  useEffect(() => {
    void bootstrap();
  }, []);

  async function bootstrap() {
    try {
      setBootstrapError(null);
      const [healthResult, infoResult, documentsResult] = await Promise.all([
        requestJson<Health>("/api/health"),
        requestJson<AppInfo>("/api/info"),
        requestJson<DocumentsResponse>("/api/documents"),
      ]);
      setHealth(healthResult);
      setInfo(infoResult);
      setDocuments(documentsResult.documents);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setBootstrapError(message);
    }
  }

  async function handleAsk() {
    setIsAsking(true);
    setAskRaw("Thinking...");
    setAskResult(null);
    try {
      const payload = await requestJson<AskResponse>("/api/ask", {
        method: "POST",
        body: JSON.stringify({
          question,
          runtime: runtime || null,
        }),
      });
      setAskResult(payload);
      setAskRaw(pretty(payload));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setAskRaw(`Error:\n${message}`);
    } finally {
      setIsAsking(false);
    }
  }

  async function handleIngest() {
    setIsActing(true);
    setActionRaw("Running ingest...");
    try {
      const payload = await requestJson("/api/ingest", { method: "POST" });
      setActionRaw(pretty(payload));
      await bootstrap();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setActionRaw(`Error:\n${message}`);
    } finally {
      setIsActing(false);
    }
  }

  async function handleEval() {
    setIsActing(true);
    setActionRaw("Running eval...");
    try {
      const payload = await requestJson("/api/eval", {
        method: "POST",
        body: JSON.stringify({ runtime: runtime || null }),
      });
      setActionRaw(pretty(payload));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setActionRaw(`Error:\n${message}`);
    } finally {
      setIsActing(false);
    }
  }

  const statusLabel = bootstrapError
    ? "Backend unavailable"
    : health
      ? `Backend ${health.status}`
      : "Checking backend...";

  return (
    <main className="shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Agent + RAG + Eval</p>
          <h1>Local Docs RAG Agent</h1>
          <p className="lede">
            A local-docs workspace for retrieval, citations, and evaluation with a split frontend and
            backend development flow.
          </p>
        </div>
        <div className="hero-stats">
          <article className="stat-card accent">
            <span className="stat-label">Backend status</span>
            <strong>{statusLabel}</strong>
            <span className="stat-foot">
              {health ? `Updated ${relativeTime(health.backend_time_utc)}` : "Waiting for health check"}
            </span>
          </article>
          <article className="stat-card">
            <span className="stat-label">Documents</span>
            <strong>{info?.docs_count ?? documents.length}</strong>
            <span className="stat-foot">{info?.docs_dir ?? "docs/"}</span>
          </article>
          <article className="stat-card">
            <span className="stat-label">Retrieval</span>
            <strong>{info?.vector_backend ?? "loading..."}</strong>
            <span className="stat-foot">top-k {info?.top_k ?? "-"}</span>
          </article>
        </div>
      </section>

      <section className="workspace">
        <div className="main-column">
          <article className="panel panel-ask">
            <div className="panel-head">
              <div>
                <h2>Ask the docs</h2>
                <p className="panel-subtle">Run the current runtime against indexed knowledge with cited output.</p>
              </div>
              <label className="runtime">
                <span>Runtime</span>
                <select value={runtime} onChange={(event) => setRuntime(event.target.value as Runtime)}>
                  <option value="">Default</option>
                  <option value="basic">basic</option>
                  <option value="agents_sdk">agents_sdk</option>
                </select>
              </label>
            </div>

            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} />

            <div className="toolbar">
              <button className="action primary" onClick={handleAsk} disabled={isAsking}>
                {isAsking ? "Thinking..." : "Ask Docs"}
              </button>
              <span className="helper-text">API base: {apiBaseUrl}</span>
            </div>

            {askResult ? (
              <section className="answer-card">
                <div className="answer-head">
                  <div>
                    <h3>Answer</h3>
                    <p className="panel-subtle">Model output with source-aware retrieval context.</p>
                  </div>
                  <div className="meta-row">
                    <span className="meta-pill">Runtime: {askResult.diagnostics.actual_runtime}</span>
                    <span className="meta-pill">{askResult.citation_spans.length} citation spans</span>
                  </div>
                </div>

                <div className="diagnostics-strip">
                  <span className={`status-chip ${askResult.diagnostics.actual_runtime !== askResult.diagnostics.requested_runtime ? "warn" : ""}`}>
                    requested {askResult.diagnostics.requested_runtime} {"->"} actual {askResult.diagnostics.actual_runtime}
                  </span>
                  <span className={`status-chip ${askResult.diagnostics.chat_provider.mode !== "live" ? "warn" : "ok"}`}>
                    chat {providerLabel(askResult.diagnostics.chat_provider)}
                  </span>
                  <span className={`status-chip ${askResult.diagnostics.embedding_provider.mode !== "live" ? "warn" : "ok"}`}>
                    embedding {providerLabel(askResult.diagnostics.embedding_provider)}
                  </span>
                </div>

                {(askResult.diagnostics.chat_provider.reason || askResult.diagnostics.embedding_provider.reason) ? (
                  <div className="reason-list">
                    {askResult.diagnostics.chat_provider.reason ? (
                      <p>
                        <strong>Chat status:</strong> {askResult.diagnostics.chat_provider.reason}
                      </p>
                    ) : null}
                    {askResult.diagnostics.embedding_provider.reason ? (
                      <p>
                        <strong>Embedding status:</strong> {askResult.diagnostics.embedding_provider.reason}
                      </p>
                    ) : null}
                  </div>
                ) : null}

                <p className="answer-text">{askResult.answer}</p>

                <div className="source-strip">
                  {askResult.citations.map((citation) => (
                    <span key={citation} className="source-chip">
                      {fileLabel(citation)}
                    </span>
                  ))}
                </div>

                <div className="citation-list">
                  {askResult.citation_spans.map((span, index) => (
                    <article key={span.chunk_id} className="citation-card">
                      <header className="citation-head">
                        <div>
                          <span className="citation-index">S{index + 1}</span>
                          <strong>{fileLabel(span.source_path)}</strong>
                          <p>{span.source_path}</p>
                        </div>
                        <div className="citation-meta">
                          <span>chunk {span.chunk_index}</span>
                          <span>
                            chars {span.start_char}-{span.end_char}
                          </span>
                        </div>
                      </header>
                      <blockquote>{excerpt(span.text, 340)}</blockquote>
                    </article>
                  ))}
                </div>
              </section>
            ) : null}

            <details className="debug-panel">
              <summary>Raw answer payload</summary>
              <pre className="output">{askRaw}</pre>
            </details>
          </article>
        </div>

        <aside className="side-column">
          <article className="panel panel-side">
            <h2>Run controls</h2>
            <div className="stack">
              <div className="control-card">
                <h3>Index</h3>
                <p>Refresh the retrieval index from the configured docs directory.</p>
                <button className="action" onClick={handleIngest} disabled={isActing}>
                  Run Ingest
                </button>
              </div>
              <div className="control-card">
                <h3>Eval</h3>
                <p>Run the current sample eval set with the active runtime selection.</p>
                <button className="action" onClick={handleEval} disabled={isActing}>
                  Run Eval
                </button>
              </div>
            </div>
            <details className="debug-panel side-debug">
              <summary>Raw action payload</summary>
              <pre className="output">{actionRaw}</pre>
            </details>
          </article>

          <article className="panel panel-side">
            <h2>Current config</h2>
            {bootstrapError ? <p className="error-text">{bootstrapError}</p> : null}
            <dl className="config-list">
              <div>
                <dt>LLM</dt>
                <dd>{info ? `${info.llm_provider} / ${info.llm_model}` : "loading..."}</dd>
              </div>
              <div>
                <dt>Embeddings</dt>
                <dd>{info ? `${info.embedding_provider} / ${info.embedding_model}` : "loading..."}</dd>
              </div>
              <div>
                <dt>Vector backend</dt>
                <dd>{info?.vector_backend ?? "loading..."}</dd>
              </div>
              <div>
                <dt>Default runtime</dt>
                <dd>{info?.runtime ?? "loading..."}</dd>
              </div>
              <div>
                <dt>Chunking</dt>
                <dd>
                  {info
                    ? `${info.chunk_strategy} / ${info.chunk_size} size / ${info.chunk_overlap} overlap`
                    : "loading..."}
                </dd>
              </div>
              <div>
                <dt>Qdrant collection</dt>
                <dd>{info?.qdrant_collection ?? "loading..."}</dd>
              </div>
            </dl>
          </article>

          <article className="panel panel-side">
            <div className="panel-head compact">
              <div>
                <h2>Documents</h2>
                <p className="panel-subtle">Source files currently available to the retriever.</p>
              </div>
              <span className="meta-pill">{documents.length}</span>
            </div>
            <div className="document-list">
              {documents.length > 0 ? (
                documents.map((document) => (
                  <article key={document} className="document-card">
                    <strong>{fileLabel(document)}</strong>
                    <span>{document}</span>
                  </article>
                ))
              ) : (
                <p className="panel-subtle">No documents found yet.</p>
              )}
            </div>
          </article>
        </aside>
      </section>
    </main>
  );
}
