"use client";
import { use, useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";
import { SandboxEventFeed } from "@/components/SandboxEventFeed";
import { SandboxTerminal } from "@/components/SandboxTerminal";

interface SandboxInfo {
  id: string;
  status: string;
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

  async function stop() {
    await fetch(`${API_BASE}/api/sandbox/${env_name}/stop`, { method: "POST" });
    setInfo((prev) => prev ? { ...prev, status: "stopped" } : prev);
  }

  async function deleteSandbox() {
    if (!confirm(`Delete environment "${env_name}"? This cannot be undone.`)) return;
    await fetch(`${API_BASE}/api/sandbox/${env_name}`, { method: "DELETE" });
    window.location.href = "/environments";
  }

  async function extendTTL() {
    alert("TTL extension coming in a future update.");
  }

  const daysLeft = info?.expires_at
    ? Math.max(0, Math.ceil((new Date(info.expires_at).getTime() - Date.now()) / 86400000))
    : null;

  return (
    <div className="h-screen flex flex-col">
      <header className="border-b px-4 py-2 flex items-center gap-3 flex-shrink-0 bg-white">
        <h1 className="font-bold text-lg mr-2">{env_name}</h1>
        <span
          className={`text-xs px-2 py-0.5 rounded-full ${
            info?.status === "running"
              ? "bg-green-100 text-green-700"
              : "bg-gray-100 text-gray-500"
          }`}
        >
          {info?.status ?? "loading…"}
        </span>
        {daysLeft !== null && (
          <span className="text-xs text-gray-400 ml-1">expires in {daysLeft}d</span>
        )}
        <div className="flex gap-2 ml-auto">
          <button onClick={reset} className="px-3 py-1 border rounded text-sm hover:bg-gray-50">
            Reset
          </button>
          <button onClick={extendTTL} className="px-3 py-1 border rounded text-sm hover:bg-gray-50">
            Extend TTL
          </button>
          <button onClick={stop} className="px-3 py-1 border rounded text-sm hover:bg-gray-50">
            Stop
          </button>
          <button
            onClick={deleteSandbox}
            className="px-3 py-1 border border-red-300 text-red-600 rounded text-sm hover:bg-red-50"
          >
            Delete
          </button>
        </div>
        {error && <span className="text-red-500 text-sm">{error}</span>}
      </header>

      <div className="flex-1 grid grid-cols-3 divide-x min-h-0">
        {/* Left: live app iframe */}
        <div className="flex flex-col min-h-0">
          <div className="text-xs text-gray-400 px-2 py-1 border-b bg-gray-50 flex-shrink-0">
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

        {/* Center: terminal */}
        <div className="flex flex-col min-h-0">
          <div className="text-xs text-gray-400 px-2 py-1 border-b bg-gray-50 flex-shrink-0">
            Terminal
          </div>
          <div className="flex-1 min-h-0">
            <SandboxTerminal envName={env_name} />
          </div>
        </div>

        {/* Right: event feed */}
        <div className="flex flex-col min-h-0">
          <div className="text-xs text-gray-400 px-2 py-1 border-b bg-gray-50 flex-shrink-0">
            Observability Feed
          </div>
          <div className="flex-1 min-h-0">
            <SandboxEventFeed envName={env_name} />
          </div>
        </div>
      </div>
    </div>
  );
}
