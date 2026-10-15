// Where the sandbox proxy sends a request. FORGE_SANDBOX_HOST names the machine
// sandboxes run on, the same setting the backend reads.

const DEFAULT_SANDBOX_HOST = "127.0.0.1";

export function sandboxHost(env: Record<string, string | undefined>): string {
  return env.FORGE_SANDBOX_HOST?.trim() || DEFAULT_SANDBOX_HOST;
}

export function sandboxUpstreamUrl(
  host: string,
  port: number,
  segments: string[] | undefined,
  search: string,
): string {
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`sandbox port must be 1-65535, got ${port}`);
  }
  const parts = segments ?? [];
  if (parts.some((segment) => segment === ".." || segment === ".")) {
    throw new Error("sandbox path segment may not be . or ..");
  }
  const path = parts.map(encodeURIComponent).join("/");
  return `http://${host}:${port}/${path}${search}`;
}

// The API as the Next.js server reaches it. FORGE_API_URL is a server-only,
// direct address (http://127.0.0.1:8000 in the Compose stack), so the lookup
// skips the public TLS host the browser uses.
export function serverApiBase(env: Record<string, string | undefined>): string {
  return env.FORGE_API_URL?.trim() || env.NEXT_PUBLIC_API_URL?.trim() || "http://localhost:8000";
}
