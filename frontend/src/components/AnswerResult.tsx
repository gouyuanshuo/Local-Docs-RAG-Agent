import { excerpt, fileLabel, providerLabel } from "../lib/presentation";
import type { AskResponse } from "../types/api";

type AnswerResultProps = {
  result: AskResponse;
};

export function AnswerResult({ result }: AnswerResultProps) {
  const { diagnostics } = result;

  return (
    <section className="answer-card">
      <div className="answer-head">
        <div>
          <h3>Answer</h3>
          <p className="panel-subtle">Model output with source-aware retrieval context.</p>
        </div>
        <div className="meta-row">
          <span className="meta-pill">Runtime: {diagnostics.actual_runtime}</span>
          <span className="meta-pill">{result.citation_spans.length} citation spans</span>
        </div>
      </div>

      <div className="diagnostics-strip">
        <span
          className={`status-chip ${
            diagnostics.actual_runtime !== diagnostics.requested_runtime ? "warn" : ""
          }`}
        >
          requested {diagnostics.requested_runtime} {"->"} actual {diagnostics.actual_runtime}
        </span>
        <span className={`status-chip ${diagnostics.chat_provider.mode !== "live" ? "warn" : "ok"}`}>
          chat {providerLabel(diagnostics.chat_provider)}
        </span>
        <span
          className={`status-chip ${
            diagnostics.embedding_provider.mode !== "live" ? "warn" : "ok"
          }`}
        >
          embedding {providerLabel(diagnostics.embedding_provider)}
        </span>
      </div>

      {diagnostics.chat_provider.reason || diagnostics.embedding_provider.reason ? (
        <div className="reason-list">
          {diagnostics.chat_provider.reason ? (
            <p>
              <strong>Chat status:</strong> {diagnostics.chat_provider.reason}
            </p>
          ) : null}
          {diagnostics.embedding_provider.reason ? (
            <p>
              <strong>Embedding status:</strong> {diagnostics.embedding_provider.reason}
            </p>
          ) : null}
        </div>
      ) : null}

      <p className="answer-text">{result.answer}</p>

      <div className="source-strip">
        {result.citations.map((citation) => (
          <span key={citation} className="source-chip">
            {fileLabel(citation)}
          </span>
        ))}
      </div>

      <div className="citation-list">
        {result.citation_spans.map((span, index) => (
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
  );
}
