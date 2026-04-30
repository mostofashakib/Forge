import { NextRequest, NextResponse } from "next/server";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function getContainerPort(envName: string): Promise<number | null> {
  try {
    const res = await fetch(`${API_BASE}/api/sandbox/${envName}`, { cache: "no-store" });
    if (!res.ok) return null;
    const data = await res.json();
    if (data.status !== "running") return null;
    return data.container_port ?? null;
  } catch {
    return null;
  }
}

async function proxyRequest(
  req: NextRequest,
  params: { env_name: string; path?: string[] },
  method: string
): Promise<NextResponse> {
  const { env_name, path } = params;
  const port = await getContainerPort(env_name);
  if (!port) {
    return new NextResponse(
      `<!doctype html><html><body style="font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#f9fafb">
        <div style="text-align:center;color:#6b7280">
          <div style="font-size:2rem;margin-bottom:8px">⏸</div>
          <div style="font-weight:600;color:#374151">Container not running</div>
          <div style="font-size:0.875rem;margin-top:4px">Use the Start button to restart this environment.</div>
        </div>
      </body></html>`,
      { status: 503, headers: { "Content-Type": "text/html" } }
    );
  }

  const pathStr = path && path.length > 0 ? path.join("/") : "";
  const qs = req.nextUrl.search;
  const url = `http://localhost:${port}/${pathStr}${qs}`;
  const headers = new Headers(req.headers);
  headers.set("X-Forge-Actor", "user");
  headers.delete("host");

  const body = method !== "GET" && method !== "HEAD" ? await req.arrayBuffer() : undefined;

  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method,
      headers,
      body: body ? Buffer.from(body) : undefined,
      redirect: "manual",
    });
  } catch {
    return new NextResponse(
      `<!doctype html><html><body style="font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#f9fafb">
        <div style="text-align:center;color:#6b7280">
          <div style="font-size:2rem;margin-bottom:8px">⚠️</div>
          <div style="font-weight:600;color:#374151">Container unreachable</div>
          <div style="font-size:0.875rem;margin-top:4px">The container may have stopped. Use the Start button to restart.</div>
        </div>
      </body></html>`,
      { status: 503, headers: { "Content-Type": "text/html" } }
    );
  }

  const responseHeaders = new Headers(upstream.headers);
  responseHeaders.delete("access-control-allow-origin");

  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ env_name: string; path?: string[] }> }
) {
  return proxyRequest(req, await params, "GET");
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ env_name: string; path?: string[] }> }
) {
  return proxyRequest(req, await params, "POST");
}

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ env_name: string; path?: string[] }> }
) {
  return proxyRequest(req, await params, "PUT");
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ env_name: string; path?: string[] }> }
) {
  return proxyRequest(req, await params, "DELETE");
}
