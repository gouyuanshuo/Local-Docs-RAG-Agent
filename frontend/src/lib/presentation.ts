import type { ProviderStatus } from "../types/api";

export function relativeTime(isoString: string): string {
  const timestamp = new Date(isoString).getTime();
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  return `${Math.floor(minutes / 60)}h ago`;
}

export function fileLabel(path: string): string {
  const parts = path.split("/");
  return parts[parts.length - 1] ?? path;
}

export function excerpt(text: string, limit = 220): string {
  if (text.length <= limit) return text;
  return `${text.slice(0, limit).trimEnd()}...`;
}

export function providerLabel(status: ProviderStatus): string {
  return `${status.provider} / ${status.mode}`;
}
