const isDev = process.env.NODE_ENV !== "production";

/**
 * Base URL for the Forge REST API.
 * Set NEXT_PUBLIC_API_URL in .env.local to override.
 * In production the env var is required.
 */
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? (isDev ? "http://localhost:8000" : "");

/** Convert an http(s) API_BASE to a ws(s) URL for WebSocket connections. */
export function wsBase(): string {
  return API_BASE.replace(/^http/, "ws");
}
