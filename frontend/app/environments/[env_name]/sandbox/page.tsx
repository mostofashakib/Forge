"use client";
import { use, useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";
import { SandboxEventFeed } from "@/components/SandboxEventFeed";
import { SandboxTerminal } from "@/components/SandboxTerminal";

interface SandboxInfo {
  id: string;
  status: string;
  env_type: string;
  container_port: number | null;
  ttl_days: number;
  expires_at: string;
}

interface Props {
  params: Promise<{ env_name: string }>;
}

export default function SandboxPage({ params }: Props) {
  const { env_name } = use(params);
  const [info, setInfo] = useState<SandboxInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/sandbox/${env_name}`)
      .then((r) => r.json())
      .then(setInfo)
      .catch(() => setError("Failed to load sandbox info"));
  }, [env_name]);

  async function reset() {
    await fetch(`/api/proxy/${env_name}/forge/reset`, { method: "POST" });
  }

  async function start() {
    const res = await fetch(`${API_BASE}/api/sandbox/${env_name}/start`, { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      setInfo((prev) => prev ? { ...prev, status: "running", container_port: data.container_port } : prev);
    }
  }

  async function stop() {
    await fetch(`${API_BASE}/api/sandbox/${env_name}/stop`, { method: "POST" });
    setInfo((prev) => prev ? { ...prev, status: "stopped" } : prev);
  }

  async function deleteSandbox() {
    if (!confirm(`Delete environment "${env_name}"? This cannot be undone.`)) return;
    await fetch(`${API_BASE}/api/sandbox/${env_name}`, { method: "DELETE" });
    window.location.href = "/environments";
  }

  const daysLeft = info?.expires_at
    ? Math.max(0, Math.ceil((new Date(info.expires_at).getTime() - Date.now()) / 86400000))
    : null;

  const envType = info?.env_type ?? "general";

  return (
    <div className="h-screen flex flex-col">
      <header className="border-b px-4 py-2 flex items-center gap-3 shrink-0 bg-white">
        <h1 className="font-bold text-lg mr-2">{env_name}</h1>
        <span className={`text-xs px-2 py-0.5 rounded-full ${
          info?.status === "running" ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-500"
        }`}>
          {info?.status ?? "loading…"}
        </span>
        {envType !== "general" && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700">
            {envType}
          </span>
        )}
        {daysLeft !== null && (
          <span className="text-xs text-gray-400 ml-1">expires in {daysLeft}d</span>
        )}
        <div className="flex gap-2 ml-auto">
          {info?.status === "running" && envType === "general" && (
            <button onClick={reset} className="px-3 py-1 border rounded text-sm hover:bg-gray-50">
              Reset
            </button>
          )}
          {info?.status !== "running" && (
            <button onClick={start} className="px-3 py-1 border border-green-300 text-green-700 rounded text-sm hover:bg-green-50">
              Start
            </button>
          )}
          {info?.status === "running" && (
            <button onClick={stop} className="px-3 py-1 border rounded text-sm hover:bg-gray-50">
              Stop
            </button>
          )}
          <button
            onClick={deleteSandbox}
            className="px-3 py-1 border border-red-300 text-red-600 rounded text-sm hover:bg-red-50"
          >
            Delete
          </button>
        </div>
        {error && <span className="text-red-500 text-sm">{error}</span>}
      </header>

      {/* CLI: full-screen terminal */}
      {envType === "cli" && (
        <div className="flex-1 min-h-0">
          <SandboxTerminal envName={env_name} />
        </div>
      )}

      {/* Browser: full-screen VNC iframe */}
      {envType === "browser" && (
        <div className="flex-1 min-h-0">
          {info?.status === "running" && info.container_port ? (
            <iframe
              src={`http://localhost:${info.container_port}/`}
              className="w-full h-full border-0"
              title={`${env_name} browser`}
            />
          ) : (
            <div className="flex-1 h-full flex items-center justify-center text-gray-400 text-sm">
              {info?.status === "building" ? "Container starting…" : "Container not running — use Start to launch"}
            </div>
          )}
        </div>
      )}

      {/* General: 3-panel layout */}
      {envType === "general" && (
        <div className="flex-1 grid grid-cols-3 divide-x min-h-0">
          <div className="flex flex-col min-h-0">
            <div className="text-xs text-gray-400 px-2 py-1 border-b bg-gray-50 shrink-0">
              Live App
            </div>
            {info?.container_port ? (
              <iframe
                src={`/api/proxy/${env_name}/ui`}
                className="flex-1 w-full border-0"
                title={`${env_name} live UI`}
              />
            ) : (
              <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
                {info?.status === "building" ? "Container starting…" : "Container not running"}
              </div>
            )}
          </div>

          <div className="flex flex-col min-h-0">
            <div className="text-xs text-gray-400 px-2 py-1 border-b bg-gray-50 shrink-0">
              Terminal
            </div>
            <div className="flex-1 min-h-0">
              <SandboxTerminal envName={env_name} />
            </div>
          </div>

          <div className="flex flex-col min-h-0">
            <div className="text-xs text-gray-400 px-2 py-1 border-b bg-gray-50 shrink-0">
              Observability Feed
            </div>
            <div className="flex-1 min-h-0">
              <SandboxEventFeed envName={env_name} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
