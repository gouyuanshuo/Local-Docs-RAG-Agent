import { useCallback, useEffect, useState } from "react";

import { errorMessage, pretty, requestJson } from "../lib/api";
import type {
  AppInfo,
  AskResponse,
  CompareResponse,
  DocumentsResponse,
  Health,
  RuntimeSelection,
} from "../types/api";

const DEFAULT_QUESTION = "How is attention explained in lecture 5?";

export function useRagWorkspace() {
  const [runtime, setRuntime] = useState<RuntimeSelection>("");
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [health, setHealth] = useState<Health | null>(null);
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [documents, setDocuments] = useState<string[]>([]);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const [askResult, setAskResult] = useState<AskResponse | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [askRaw, setAskRaw] = useState("Waiting for a question...");
  const [actionRaw, setActionRaw] = useState("System actions will appear here...");
  const [isAsking, setIsAsking] = useState(false);
  const [isActing, setIsActing] = useState(false);

  const bootstrap = useCallback(async () => {
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
      setBootstrapError(errorMessage(error));
    }
  }, []);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  const ask = useCallback(async () => {
    setIsAsking(true);
    setAskRaw("Thinking...");
    setAskResult(null);
    try {
      const payload = await requestJson<AskResponse>("/api/ask", {
        method: "POST",
        body: JSON.stringify({ question, runtime: runtime || null }),
      });
      setAskResult(payload);
      setAskRaw(pretty(payload));
    } catch (error) {
      setAskRaw(`Error:\n${errorMessage(error)}`);
    } finally {
      setIsAsking(false);
    }
  }, [question, runtime]);

  const ingest = useCallback(async () => {
    setIsActing(true);
    setActionRaw("Running ingest...");
    try {
      const payload = await requestJson<Record<string, unknown>>("/api/ingest", {
        method: "POST",
      });
      setActionRaw(pretty(payload));
      await bootstrap();
    } catch (error) {
      setActionRaw(`Error:\n${errorMessage(error)}`);
    } finally {
      setIsActing(false);
    }
  }, [bootstrap]);

  const evaluate = useCallback(async () => {
    setIsActing(true);
    setActionRaw("Running eval...");
    try {
      const payload = await requestJson<Record<string, unknown>>("/api/eval", {
        method: "POST",
        body: JSON.stringify({ runtime: runtime || null }),
      });
      setActionRaw(pretty(payload));
    } catch (error) {
      setActionRaw(`Error:\n${errorMessage(error)}`);
    } finally {
      setIsActing(false);
    }
  }, [runtime]);

  const compare = useCallback(async () => {
    setIsActing(true);
    setActionRaw("Running compare eval...");
    setCompareResult(null);
    try {
      const payload = await requestJson<CompareResponse>("/api/eval/compare", {
        method: "POST",
        body: JSON.stringify({ runtimes: runtime ? [runtime] : undefined }),
      });
      setCompareResult(payload);
      setActionRaw(pretty(payload));
    } catch (error) {
      setActionRaw(`Error:\n${errorMessage(error)}`);
    } finally {
      setIsActing(false);
    }
  }, [runtime]);

  return {
    runtime,
    setRuntime,
    question,
    setQuestion,
    health,
    info,
    documents,
    bootstrapError,
    askResult,
    compareResult,
    askRaw,
    actionRaw,
    isAsking,
    isActing,
    ask,
    ingest,
    evaluate,
    compare,
  };
}
