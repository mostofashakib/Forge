"use client";
import { use, useEffect, useRef, useState } from "react";
import { API_BASE } from "@/lib/api";
import { ActivityLog } from "@/components/ActivityLog";
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

type Tab = "app" | "terminal" | "observability";

const TABS: { id: Tab; label: string }[] = [
  { id: "app",           label: "App" },
  { id: "terminal",      label: "Terminal" },
  { id: "observability", label: "Observability" },
];

export default function SandboxPage({ params }: Props) {
  const { env_name } = use(params);
  const [info, setInfo]       = useState<SandboxInfo | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("app");

  // Per-action loading flags
  const [starting,  setStarting]  = useState(false);
  const [stopping,  setStopping]  = useState(false);
  const [deleting,  setDeleting]  = useState(false);
  const [resetting, setResetting] = useState(false);

  // Iframe ref for Reload control
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/sandbox/${env_name}`)
      .then((r) => r.json())
      .then(setInfo)
      .catch(() => setError("Failed to load sandbox info"));
  }, [env_name]);

  function showError(msg: string) {
    setError(msg);
    setTimeout(() => setError(null), 5000);
  }

  async function start() {
    setStarting(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${env_name}/start`, { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        showError(body.detail ?? "Failed to start container");
        return;
      }
      const data = await res.json();
      setInfo((prev) => prev ? { ...prev, status: "running", container_port: data.container_port } : prev);
    } catch {
      showError("Network error while starting container");
    } finally {
      setStarting(false);
    }
  }

  async function stop() {
    setStopping(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${env_name}/stop`, { method: "POST" });
      if (!res.ok) {
        showError("Failed to stop container");
        return;
      }
      setInfo((prev) => prev ? { ...prev, status: "stopped", container_port: null } : prev);
    } catch {
      showError("Network error while stopping container");
    } finally {
      setStopping(false);
    }
  }

  async function deleteSandbox() {
    if (!confirm(`Delete environment "${env_name}"? This cannot be undone.`)) return;
    setDeleting(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${env_name}`, { method: "DELETE" });
      if (!res.ok) {
        showError("Failed to delete environment");
        return;
      }
      window.location.href = "/environments";
    } catch {
      showError("Network error while deleting environment");
      setDeleting(false);
    }
  }

  async function reset() {
    setResetting(true);
    setError(null);
    try {
      const res = await fetch(`/api/proxy/${env_name}/forge/reset`, { method: "POST" });
      if (!res.ok) showError("Reset failed");
    } catch {
      showError("Network error while resetting");
    } finally {
      setResetting(false);
    }
  }

  function reloadIframe() {
    if (iframeRef.current) {
      // eslint-disable-next-line no-self-assign
      iframeRef.current.src = iframeRef.current.src;
    }
  }

  const daysLeft = info?.expires_at
    ? Math.max(0, Math.ceil((new Date(info.expires_at).getTime() - Date.now()) / 86400000))
    : null;

  const envType = info?.env_type ?? "general";
  const hasAppUI = envType === "general" || envType.startsWith("premade:");
  const isRunning = info?.status === "running";
  const canStart  = !isRunning && !starting && !stopping && !deleting
    && info?.status !== "building" && info?.status !== "queued";

  const visibleTabs = envType === "cli"
    ? TABS.filter((t) => t.id === "terminal" || t.id === "observability")
    : envType === "browser"
    ? TABS.filter((t) => t.id === "app" || t.id === "observability")
    : TABS;

  const effectiveTab = visibleTabs.find((t) => t.id === activeTab)
    ? activeTab
    : visibleTabs[0]?.id ?? "app";

  return (
    <div className="h-screen flex flex-col">
      <header className="border-b px-4 py-2 flex items-center gap-3 shrink-0 bg-white">
        <h1 className="font-bold text-lg mr-2">{env_name}</h1>

        {/* Status badge */}
        <span className={`text-xs px-2 py-0.5 rounded-full ${
          isRunning             ? "bg-green-100 text-green-700"  :
          info?.status === "building" || info?.status === "queued"
                                ? "bg-yellow-100 text-yellow-700" :
          info?.status === "error"    ? "bg-red-100 text-red-600"    :
                                        "bg-gray-100 text-gray-500"
        }`}>
          {info?.status ?? "loading…"}
        </span>

        {/* Env-type badge */}
        {envType !== "general" && !envType.startsWith("premade:") && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700">
            {envType}
          </span>
        )}
        {envType.startsWith("premade:") && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-orange-100 text-orange-700">
            {envType.slice("premade:".length)}
          </span>
        )}

        {daysLeft !== null && (
          <span className="text-xs text-gray-400 ml-1">expires in {daysLeft}d</span>
        )}

        {/* Controls */}
        <div className="flex gap-2 ml-auto items-center">

          {/* General / premade: Reset */}
          {hasAppUI && isRunning && (
            <button
              onClick={reset}
              disabled={resetting}
              className="px-3 py-1 border rounded text-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {resetting ? "Resetting…" : "Reset"}
            </button>
          )}

          {/* Browser: Reload iframe + Open in new tab */}
          {envType === "browser" && isRunning && info?.container_port && (
            <>
              <button
                onClick={reloadIframe}
                className="px-3 py-1 border rounded text-sm hover:bg-gray-50"
                title="Reload browser view"
              >
                Reload
              </button>
              <a
                href={`http://localhost:${info.container_port}/`}
                target="_blank"
                rel="noreferrer"
                className="px-3 py-1 border rounded text-sm hover:bg-gray-50"
                title="Open in a new browser tab"
              >
                Open ↗
              </a>
            </>
          )}

          {/* Start */}
          {canStart && (
            <button
              onClick={start}
              disabled={starting}
              className="px-3 py-1 border border-green-300 text-green-700 rounded text-sm hover:bg-green-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {starting ? "Starting…" : "Start"}
            </button>
          )}

          {/* Stop */}
          {isRunning && (
            <button
              onClick={stop}
              disabled={stopping}
              className="px-3 py-1 border rounded text-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {stopping ? "Stopping…" : "Stop"}
            </button>
          )}

          {/* Delete */}
          <button
            onClick={deleteSandbox}
            disabled={deleting}
            className="px-3 py-1 border border-red-300 text-red-600 rounded text-sm hover:bg-red-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {deleting ? "Deleting…" : "Delete"}
          </button>
        </div>

        {/* Error banner */}
        {error && (
          <span className="text-red-500 text-sm ml-2 shrink-0">{error}</span>
        )}
      </header>

      {/* Tab bar */}
      {visibleTabs.length > 1 && (
        <div className="border-b bg-white flex shrink-0">
          {visibleTabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-5 py-2 text-sm font-medium border-b-2 transition-colors ${
                effectiveTab === tab.id
                  ? "border-blue-600 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      )}

      {/* Content area */}
      <div className="flex-1 min-h-0">

        {/* Terminal (CLI + general) */}
        {effectiveTab === "terminal" && (
          <SandboxTerminal envName={env_name} />
        )}

        {/* App tab — browser */}
        {effectiveTab === "app" && envType === "browser" && (
          isRunning && info?.container_port ? (
            <iframe
              ref={iframeRef}
              src={`http://localhost:${info.container_port}/`}
              className="w-full h-full border-0"
              title={`${env_name} browser`}
            />
          ) : (
            <div className="h-full flex items-center justify-center text-gray-400 text-sm">
              {info?.status === "building" || info?.status === "queued"
                ? "Container is starting up…"
                : "Container not running — press Start to launch"}
            </div>
          )
        )}

        {/* App tab — general / premade */}
        {effectiveTab === "app" && hasAppUI && (
          info?.container_port ? (
            <iframe
              src={`/api/proxy/${env_name}/ui`}
              className="w-full h-full border-0"
              title={`${env_name} live UI`}
            />
          ) : (
            <div className="h-full flex items-center justify-center text-gray-400 text-sm">
              {info?.status === "building" || info?.status === "queued"
                ? "Container is starting up…"
                : "Container not running"}
            </div>
          )
        )}

        {/* Observability — general / premade use rich event feed */}
        {effectiveTab === "observability" && hasAppUI && (
          <SandboxEventFeed envName={env_name} />
        )}

        {/* Observability — CLI / browser use activity log */}
        {effectiveTab === "observability" && !hasAppUI && (
          <ActivityLog envName={env_name} />
        )}
      </div>
    </div>
  );
}
