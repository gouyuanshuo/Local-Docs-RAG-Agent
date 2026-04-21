import { useEffect, useState } from "react";

type Runtime = "" | "basic" | "agents_sdk";

type AskResponse = {
  question: string;
  answer: string;
  citations: string[];
  citation_spans: CitationSpan[];
  runtime: string;
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
};

type Health = {
  status: string;
};

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

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

export default function App() {
  const [runtime, setRuntime] = useState<Runtime>("");
  const [question, setQuestion] = useState("How is attention explained in lecture 5?");
  const [health, setHealth] = useState("Checking backend...");
  const [info, setInfo] = useState("Loading config...");
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
      const [healthResult, infoResult] = await Promise.all([
        requestJson<Health>("/api/health"),
        requestJson<AppInfo>("/api/info"),
      ]);
      setHealth(`Backend: ${healthResult.status}`);
      setInfo(`Runtime ${infoResult.runtime} · Backend ${infoResult.vector_backend}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setHealth("Backend unavailable");
      setInfo(message);
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

  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">Agent + RAG + Eval</p>
        <h1>Local Docs RAG Agent</h1>
        <p className="lede">
          Separate frontend and backend, with a live docs QA workspace on top of your local retrieval pipeline.
        </p>
        <div className="statusbar">
          <span className="pill">{health}</span>
          <span className="pill muted">{info}</span>
        </div>
      </section>

      <section className="grid">
        <article className="panel panel-ask">
          <div className="panel-head">
            <h2>Ask</h2>
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
          <button className="action primary" onClick={handleAsk} disabled={isAsking}>
            {isAsking ? "Thinking..." : "Ask Docs"}
          </button>

          {askResult ? (
            <section className="answer-card">
              <h3>Answer</h3>
              <p>{askResult.answer}</p>
              <div className="meta-row">
                <span className="meta-pill">Runtime: {askResult.runtime}</span>
                {askResult.citations.map((citation) => (
                  <span key={citation} className="meta-pill">
                    {citation}
                  </span>
                ))}
              </div>
              <div className="citation-list">
                {askResult.citation_spans.map((span) => (
                  <article key={span.chunk_id} className="citation-card">
                    <header>
                      <strong>{span.source_path}</strong>
                      <span>
                        chars {span.start_char}-{span.end_char}
                      </span>
                    </header>
                    <pre>{span.text}</pre>
                  </article>
                ))}
              </div>
            </section>
          ) : null}

          <pre className="output">{askRaw}</pre>
        </article>

        <article className="panel panel-actions">
          <div className="stack">
            <div>
              <h2>Index</h2>
              <p>Refresh the retrieval index from the configured docs directory.</p>
              <button className="action" onClick={handleIngest} disabled={isActing}>
                Run Ingest
              </button>
            </div>
            <div>
              <h2>Eval</h2>
              <p>Run the current batch eval set against the configured backend.</p>
              <button className="action" onClick={handleEval} disabled={isActing}>
                Run Eval
              </button>
            </div>
          </div>
          <pre className="output">{actionRaw}</pre>
        </article>
      </section>
    </main>
  );
}
