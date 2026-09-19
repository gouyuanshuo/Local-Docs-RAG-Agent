import { fileLabel } from "../lib/presentation";

type DocumentsPanelProps = {
  documents: string[];
  error: string | null;
};

export function DocumentsPanel({ documents, error }: DocumentsPanelProps) {
  return (
    <article className="panel panel-side">
      <div className="panel-head compact">
        <div>
          <h2>Documents</h2>
          <p className="panel-subtle">Source files currently available to the retriever.</p>
        </div>
        <span className="meta-pill">{documents.length}</span>
      </div>
      <div className="document-list">
        {error ? (
          <p className="error-text" role="alert">
            Documents unavailable: {error}
          </p>
        ) : documents.length > 0 ? (
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
  );
}
