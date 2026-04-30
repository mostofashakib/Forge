import { NextRequest, NextResponse } from "next/server";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function getContainerPort(envName: string): Promise<number | null> {
  try {
    const res = await fetch(`${API_BASE}/api/sandbox/${envName}`, { cache: "no-store" });
    if (!res.ok) return null;
    const data = await res.json();
    return data.container_port ?? null;
  } catch {
    return null;
  }
}

async function proxyRequest(
  req: NextRequest,
  params: { env_name: string; path: string[] },
  method: string
): Promise<NextResponse> {
  const { env_name, path } = params;
  const port = await getContainerPort(env_name);
  if (!port) {
    return NextResponse.json(
      { error: "Container not running or port unavailable" },
      { status: 503 }
    );
  }

  const url = `http://localhost:${port}/${path.join("/")}`;
  const headers = new Headers(req.headers);
  headers.set("X-Forge-Actor", "user");
  headers.delete("host");

  const body = method !== "GET" && method !== "HEAD" ? await req.arrayBuffer() : undefined;

  const upstream = await fetch(url, {
    method,
    headers,
    body: body ? Buffer.from(body) : undefined,
    redirect: "manual",
  });

  const responseHeaders = new Headers(upstream.headers);
  responseHeaders.delete("access-control-allow-origin");

  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

export async function GET(
  req: NextRequest,
  { params }: { params: { env_name: string; path: string[] } }
) {
  return proxyRequest(req, params, "GET");
}

export async function POST(
  req: NextRequest,
  { params }: { params: { env_name: string; path: string[] } }
) {
  return proxyRequest(req, params, "POST");
}

export async function PUT(
  req: NextRequest,
  { params }: { params: { env_name: string; path: string[] } }
) {
  return proxyRequest(req, params, "PUT");
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: { env_name: string; path: string[] } }
) {
  return proxyRequest(req, params, "DELETE");
}
