import { useCallback, useEffect, useRef, useState } from "react";

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
  const bootstrapGeneration = useRef(0);
  const [runtime, setRuntime] = useState<RuntimeSelection>("");
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [health, setHealth] = useState<Health | null>(null);
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [documents, setDocuments] = useState<string[]>([]);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [askResult, setAskResult] = useState<AskResponse | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [askRaw, setAskRaw] = useState("Waiting for a question...");
  const [actionRaw, setActionRaw] = useState("System actions will appear here...");
  const [askError, setAskError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [isAsking, setIsAsking] = useState(false);
  const [isActing, setIsActing] = useState(false);

  const bootstrap = useCallback(async () => {
    bootstrapGeneration.current += 1;
    const generation = bootstrapGeneration.current;
    setHealthError(null);
    setInfoError(null);
    setDocumentsError(null);
    const [healthResult, infoResult, documentsResult] = await Promise.allSettled([
      requestJson<Health>("/api/health"),
      requestJson<AppInfo>("/api/info"),
      requestJson<DocumentsResponse>("/api/documents"),
    ]);
    if (generation !== bootstrapGeneration.current) {
      return;
    }
    if (healthResult.status === "fulfilled") {
      setHealth(healthResult.value);
    } else {
      setHealth(null);
      setHealthError(errorMessage(healthResult.reason));
    }
    if (infoResult.status === "fulfilled") {
      setInfo(infoResult.value);
    } else {
      setInfo(null);
      setInfoError(errorMessage(infoResult.reason));
    }
    if (documentsResult.status === "fulfilled") {
      setDocuments(documentsResult.value.documents);
    } else {
      setDocuments([]);
      setDocumentsError(errorMessage(documentsResult.reason));
    }
  }, []);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  const ask = useCallback(async () => {
    setAskError(null);
    setAskResult(null);
    if (!question.trim()) {
      const message = "Question must not be blank";
      setAskError(message);
      setAskRaw(`Error:\n${message}`);
      return;
    }
    setIsAsking(true);
    setAskRaw("Thinking...");
    try {
      const payload = await requestJson<AskResponse>("/api/ask", {
        method: "POST",
        body: JSON.stringify({ question, runtime: runtime || null }),
      });
      setAskResult(payload);
      setAskRaw(pretty(payload));
    } catch (error) {
      const message = errorMessage(error);
      setAskError(message);
      setAskRaw(`Error:\n${message}`);
    } finally {
      setIsAsking(false);
    }
  }, [question, runtime]);

  const ingest = useCallback(async () => {
    setActionError(null);
    setIsActing(true);
    setActionRaw("Running ingest...");
    try {
      const payload = await requestJson<Record<string, unknown>>("/api/ingest", {
        method: "POST",
      });
      setActionRaw(pretty(payload));
      await bootstrap();
    } catch (error) {
      const message = errorMessage(error);
      setActionError(message);
      setActionRaw(`Error:\n${message}`);
    } finally {
      setIsActing(false);
    }
  }, [bootstrap]);

  const evaluate = useCallback(async () => {
    setActionError(null);
    setIsActing(true);
    setActionRaw("Running eval...");
    try {
      const payload = await requestJson<Record<string, unknown>>("/api/eval", {
        method: "POST",
        body: JSON.stringify({ runtime: runtime || null }),
      });
      setActionRaw(pretty(payload));
    } catch (error) {
      const message = errorMessage(error);
      setActionError(message);
      setActionRaw(`Error:\n${message}`);
    } finally {
      setIsActing(false);
    }
  }, [runtime]);

  const compare = useCallback(async () => {
    setActionError(null);
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
      const message = errorMessage(error);
      setActionError(message);
      setActionRaw(`Error:\n${message}`);
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
    healthError,
    infoError,
    documentsError,
    askResult,
    compareResult,
    askRaw,
    actionRaw,
    askError,
    actionError,
    isAsking,
    isActing,
    ask,
    ingest,
    evaluate,
    compare,
  };
}
