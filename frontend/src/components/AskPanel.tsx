import { apiBaseUrl } from "../lib/api";
import type { AskResponse, RuntimeSelection } from "../types/api";
import { AnswerResult } from "./AnswerResult";

type AskPanelProps = {
  runtime: RuntimeSelection;
  question: string;
  result: AskResponse | null;
  rawPayload: string;
  isAsking: boolean;
  error: string | null;
  onRuntimeChange: (runtime: RuntimeSelection) => void;
  onQuestionChange: (question: string) => void;
  onAsk: () => Promise<void>;
};

export function AskPanel({
  runtime,
  question,
  result,
  rawPayload,
  isAsking,
  error,
  onRuntimeChange,
  onQuestionChange,
  onAsk,
}: AskPanelProps) {
  return (
    <article className="panel panel-ask">
      <div className="panel-head">
        <div>
          <h2>Ask the docs</h2>
          <p className="panel-subtle">
            Run the current runtime against indexed knowledge with cited output.
          </p>
        </div>
        <label className="runtime">
          <span>Runtime</span>
          <select
            value={runtime}
            onChange={(event) => onRuntimeChange(event.target.value as RuntimeSelection)}
          >
            <option value="">Default</option>
            <option value="basic">basic</option>
            <option value="agents_sdk">agents_sdk</option>
          </select>
        </label>
      </div>

      <label className="question-field">
        <span>Question</span>
        <textarea
          value={question}
          aria-describedby={error ? "ask-error" : undefined}
          onChange={(event) => onQuestionChange(event.target.value)}
        />
      </label>

      {error ? (
        <p id="ask-error" className="error-text" role="alert">
          {error}
        </p>
      ) : null}

      <div className="toolbar">
        <button
          className="action primary"
          onClick={() => void onAsk()}
          disabled={isAsking}
        >
          {isAsking ? "Thinking..." : "Ask Docs"}
        </button>
        <span className="helper-text">API base: {apiBaseUrl}</span>
      </div>

      {result ? <AnswerResult result={result} /> : null}

      <details className="debug-panel">
        <summary>Raw answer payload</summary>
        <pre className="output">{rawPayload}</pre>
      </details>
    </article>
  );
}
