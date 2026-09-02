import type { ApiErrorPayload } from "../types/api";

export const apiBaseUrl =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string | undefined;
  readonly actionHint: string | null;

  constructor(status: number, payload: ApiErrorPayload | null, fallback: string) {
    const message =
      typeof payload?.detail === "string"
        ? payload.detail
        : payload?.detail
          ? JSON.stringify(payload.detail, null, 2)
          : fallback;
    super(payload?.action_hint ? `${message}\n${payload.action_hint}` : message);
    this.name = "ApiError";
    this.status = status;
    this.code = payload?.code;
    this.actionHint = payload?.action_hint ?? null;
  }
}

export async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (typeof init?.body === "string" && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers,
  });

  if (!response.ok) {
    const body = await response.text();
    let payload: ApiErrorPayload | null = null;
    try {
      payload = body ? (JSON.parse(body) as ApiErrorPayload) : null;
    } catch {
      // Non-JSON proxy/server errors retain their original response body.
    }
    throw new ApiError(response.status, payload, body || `Request failed: ${response.status}`);
  }

  return (await response.json()) as T;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function pretty(value: unknown): string {
  return JSON.stringify(value, null, 2);
}
