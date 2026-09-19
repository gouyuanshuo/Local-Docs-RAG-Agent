import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";

import App from "../App";
import type { AppInfo, AskResponse, Health } from "../types/api";

const HEALTH: Health = {
  status: "ok",
  backend_time_utc: "2026-09-19T00:00:00Z",
};

const INFO: AppInfo = {
  name: "Local Docs RAG Agent",
  runtime: "basic",
  vector_backend: "local",
  docs_dir: "docs",
  docs_exclude_patterns: [],
  docs_count: 1,
  llm_provider: "test-chat",
  llm_model: "test-model",
  embedding_provider: "test-embedding",
  embedding_model: "test-embedding-model",
  top_k: 4,
  retrieval_strategy: "blended",
  reranker: "none",
  chunk_strategy: "fixed",
  chunk_size: 800,
  chunk_overlap: 120,
  qdrant_collection: "test-collection",
  external_http_trust_env: false,
};

const ASK_RESPONSE: AskResponse = {
  question: "What changed?",
  answer: "The retry recovered.",
  citations: ["docs/example.md"],
  citation_spans: [
    {
      source_path: "docs/example.md",
      chunk_id: "example-0",
      chunk_index: 0,
      start_char: 0,
      end_char: 20,
      text: "Supporting evidence.",
      source_id: "S1",
    },
  ],
  runtime: "basic",
  diagnostics: {
    requested_runtime: "basic",
    actual_runtime: "basic",
    vector_backend: "local",
    chat_provider: { provider: "test-chat", mode: "live", reason: null },
    embedding_provider: {
      provider: "test-embedding",
      mode: "live",
      reason: null,
    },
    reranker: { provider: "none", mode: "ready", reason: null },
  },
};

type RouteHandler = (init: RequestInit | undefined) => Response | Promise<Response>;
type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
};

let fetchMock: ReturnType<typeof vi.fn<typeof fetch>>;

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function deferred<T>(): Deferred<T> {
  let resolve: (value: T) => void = () => undefined;
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve;
  });
  return { promise, resolve };
}

function mockApi(overrides: Record<string, RouteHandler> = {}) {
  const routes: Record<string, RouteHandler> = {
    "/api/health": () => jsonResponse(HEALTH),
    "/api/info": () => jsonResponse(INFO),
    "/api/documents": () =>
      jsonResponse({ count: 1, documents: ["docs/example.md"] }),
    ...overrides,
  };
  fetchMock.mockImplementation(async (input, init) => {
    const path = new URL(String(input)).pathname;
    const handler = routes[path];
    if (!handler) {
      throw new Error(`Unexpected request: ${path}`);
    }
    return handler(init);
  });
}

function callsFor(path: string) {
  return fetchMock.mock.calls.filter(([input]) => new URL(String(input)).pathname === path);
}

beforeEach(() => {
  fetchMock = vi.fn<typeof fetch>();
  vi.stubGlobal("fetch", fetchMock);
});

it("rejects a blank question visibly without calling the ask endpoint", async () => {
  mockApi();
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText("Backend ok");

  await user.clear(screen.getByRole("textbox"));
  await user.click(screen.getByRole("button", { name: "Ask Docs" }));

  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Question must not be blank",
  );
  expect(callsFor("/api/ask")).toHaveLength(0);
});

it("shows an ask provider failure outside the raw payload", async () => {
  mockApi({
    "/api/ask": () =>
      jsonResponse(
        {
          detail: "Embedding provider unavailable",
          action_hint: "Configure a live embedding provider and retry.",
        },
        503,
      ),
  });
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText("Backend ok");

  await user.click(screen.getByRole("button", { name: "Ask Docs" }));

  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("Embedding provider unavailable");
  expect(alert).toHaveTextContent("Configure a live embedding provider and retry.");
  expect(screen.getByText(/Error:\s*Embedding provider unavailable/)).toBeInTheDocument();
});

it("retains health and config when only document bootstrap fails", async () => {
  mockApi({
    "/api/documents": () =>
      jsonResponse(
        {
          detail: "Document inventory unavailable",
          action_hint: "Check the configured docs directory.",
        },
        503,
      ),
  });
  render(<App />);

  expect(await screen.findByText("Backend ok")).toBeVisible();
  expect(screen.getByText("test-chat / test-model")).toBeVisible();
  const alert = screen.getByRole("alert");
  expect(alert).toHaveTextContent("Document inventory unavailable");
  expect(alert).toHaveTextContent("Check the configured docs directory.");
  expect(screen.queryByText("Backend unavailable")).not.toBeInTheDocument();
});

it("renders completed bootstrap resources while a sibling is still pending", async () => {
  const pendingDocuments = deferred<Response>();
  mockApi({
    "/api/documents": () => pendingDocuments.promise,
  });
  render(<App />);

  expect(await screen.findByText("Backend ok")).toBeVisible();
  expect(screen.getByText("test-chat / test-model")).toBeVisible();
  expect(callsFor("/api/documents")).toHaveLength(1);
});

it("finishes ingest while a document refresh is still pending", async () => {
  const pendingDocuments = deferred<Response>();
  let documentCalls = 0;
  mockApi({
    "/api/documents": () => {
      documentCalls += 1;
      return documentCalls === 1
        ? jsonResponse({ count: 1, documents: ["docs/example.md"] })
        : pendingDocuments.promise;
    },
    "/api/ingest": () => jsonResponse({ status: "ok" }),
  });
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText("Backend ok");

  const ingestButton = screen.getByRole("button", { name: "Run Ingest" });
  await user.click(ingestButton);
  await waitFor(() => {
    expect(callsFor("/api/documents")).toHaveLength(2);
  });

  await waitFor(() => {
    expect(ingestButton).toBeEnabled();
  });
  expect(screen.getByText(/"status": "ok"/)).toBeInTheDocument();
});

it("ignores an older bootstrap that settles after a post-ingest refresh", async () => {
  const initialHealth = deferred<Response>();
  const initialInfo = deferred<Response>();
  const initialDocuments = deferred<Response>();
  let healthCalls = 0;
  let infoCalls = 0;
  let documentCalls = 0;
  const freshInfo: AppInfo = {
    ...INFO,
    llm_provider: "fresh-chat",
    llm_model: "fresh-model",
    docs_count: 1,
  };
  const staleInfo: AppInfo = {
    ...INFO,
    llm_provider: "stale-chat",
    llm_model: "stale-model",
    docs_count: 1,
  };
  mockApi({
    "/api/health": () => {
      healthCalls += 1;
      return healthCalls === 1 ? initialHealth.promise : jsonResponse(HEALTH);
    },
    "/api/info": () => {
      infoCalls += 1;
      return infoCalls === 1 ? initialInfo.promise : jsonResponse(freshInfo);
    },
    "/api/documents": () => {
      documentCalls += 1;
      return documentCalls === 1
        ? initialDocuments.promise
        : jsonResponse({ count: 1, documents: ["docs/fresh.md"] });
    },
    "/api/ingest": () => jsonResponse({ status: "ok" }),
  });
  const user = userEvent.setup();
  render(<App />);
  await waitFor(() => {
    expect(callsFor("/api/health")).toHaveLength(1);
  });

  await user.click(screen.getByRole("button", { name: "Run Ingest" }));
  expect(await screen.findByText("fresh-chat / fresh-model")).toBeVisible();
  expect(await screen.findByText("fresh.md")).toBeVisible();

  await act(async () => {
    initialHealth.resolve(jsonResponse(HEALTH));
    initialInfo.resolve(jsonResponse(staleInfo));
    initialDocuments.resolve(
      jsonResponse({ count: 1, documents: ["docs/stale.md"] }),
    );
    await Promise.all([
      initialHealth.promise,
      initialInfo.promise,
      initialDocuments.promise,
    ]);
  });

  expect(screen.getByText("fresh-chat / fresh-model")).toBeVisible();
  expect(screen.getByText("fresh.md")).toBeVisible();
  expect(screen.queryByText("stale-chat / stale-model")).not.toBeInTheDocument();
  expect(screen.queryByText("stale.md")).not.toBeInTheDocument();
});

it("clears an ask failure after a successful retry", async () => {
  let attempts = 0;
  mockApi({
    "/api/ask": () => {
      attempts += 1;
      return attempts === 1
        ? jsonResponse({ detail: "Provider temporarily unavailable" }, 503)
        : jsonResponse(ASK_RESPONSE);
    },
  });
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText("Backend ok");

  await user.click(screen.getByRole("button", { name: "Ask Docs" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Provider temporarily unavailable",
  );

  await user.click(screen.getByRole("button", { name: "Ask Docs" }));
  expect(await screen.findByText("The retry recovered.")).toBeVisible();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(callsFor("/api/ask")).toHaveLength(2);
});

it("renders backend citation ids and preserves an unlabelled span", async () => {
  mockApi({
    "/api/ask": () =>
      jsonResponse({
        ...ASK_RESPONSE,
        citation_spans: [
          ASK_RESPONSE.citation_spans[0],
          {
            ...ASK_RESPONSE.citation_spans[0],
            chunk_id: "example-1",
            chunk_index: 1,
            source_id: null,
          },
        ],
      }),
  });
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText("Backend ok");

  await user.click(screen.getByRole("button", { name: "Ask Docs" }));

  expect(await screen.findByText("S1")).toBeVisible();
  expect(screen.getByText("unlabelled")).toBeVisible();
  expect(screen.queryByText("S2")).not.toBeInTheDocument();
});

it("shows action failures and clears them after a successful retry", async () => {
  let attempts = 0;
  mockApi({
    "/api/eval": () => {
      attempts += 1;
      return attempts === 1
        ? jsonResponse({ detail: "Evaluation service unavailable" }, 503)
        : jsonResponse({ num_cases: 0 });
    },
  });
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText("Backend ok");

  await user.click(screen.getByRole("button", { name: "Run Eval" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Evaluation service unavailable",
  );

  await user.click(screen.getByRole("button", { name: "Run Eval" }));
  expect(await screen.findByText(/"num_cases": 0/)).toBeInTheDocument();
  await waitFor(() => {
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
  expect(callsFor("/api/eval")).toHaveLength(2);
});
