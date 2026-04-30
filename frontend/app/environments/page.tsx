"use client";
import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

interface SandboxInfo {
  id: string;
  status: string;
  expires_at: string;
}

const STATUS_COLORS: Record<string, string> = {
  running:  "bg-green-100 text-green-700",
  building: "bg-blue-100 text-blue-700",
  queued:   "bg-yellow-100 text-yellow-700",
  stopped:  "bg-gray-100 text-gray-500",
  error:    "bg-red-100 text-red-600",
};

const IN_PROGRESS = new Set(["queued", "building"]);

export default function EnvironmentsPage() {
  const [allNames, setAllNames] = useState<string[]>([]);
  const [sandboxMap, setSandboxMap] = useState<Map<string, SandboxInfo>>(new Map());
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const signal = AbortSignal.timeout(8000);
      const [fileEnvs, sandboxes]: [string[], SandboxInfo[]] = await Promise.all([
        fetch(`${API_BASE}/api/envs/`, { signal })
          .then((r) => (r.ok ? r.json() : [])).catch(() => []),
        fetch(`${API_BASE}/api/sandbox/`, { signal })
          .then((r) => (r.ok ? r.json() : [])).catch(() => []),
      ]);
      const safeEnvs: string[] = Array.isArray(fileEnvs) ? fileEnvs : [];
      const safeSandboxes: SandboxInfo[] = Array.isArray(sandboxes) ? sandboxes : [];
      const map = new Map(safeSandboxes.map((s) => [s.id, s]));
      const names = Array.from(new Set([...safeEnvs, ...safeSandboxes.map((s) => s.id)])).sort();
      setSandboxMap(map);
      setAllNames(names);
    } catch {
      // backend unreachable — show empty list
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  async function handleDelete(name: string, hasSandbox: boolean, e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm(`Delete environment "${name}"? This cannot be undone.`)) return;
    setDeleting(name);
    const url = hasSandbox
      ? `${API_BASE}/api/sandbox/${name}`
      : `${API_BASE}/api/envs/${name}`;
    await fetch(url, { method: "DELETE" }).catch(() => {});
    setAllNames((prev) => prev.filter((n) => n !== name));
    setSandboxMap((prev) => { const next = new Map(prev); next.delete(name); return next; });
    setDeleting(null);
  }

  // Poll every 3 s while any env is still building/queued
  useEffect(() => {
    const anyInProgress = Array.from(sandboxMap.values()).some((s) => IN_PROGRESS.has(s.status));
    if (!anyInProgress) return;
    const id = setInterval(fetchData, 3000);
    return () => clearInterval(id);
  }, [sandboxMap, fetchData]);

  if (loading) {
    return <div className="py-20 text-center text-sm text-muted-foreground">Loading…</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Environments</h1>
          <p className="text-muted-foreground text-sm mt-1.5">
            {allNames.length} environment{allNames.length !== 1 ? "s" : ""}
          </p>
        </div>
        <Link
          href="/environments/new"
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 transition-colors"
        >
          + New Environment
        </Link>
      </div>

      {allNames.length === 0 ? (
        <div className="text-center py-24 border rounded-lg">
          <p className="text-muted-foreground">No environments yet.</p>
          <Link href="/environments/new" className="text-sm text-primary mt-2 block hover:underline">
            Create your first environment →
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {allNames.map((name) => {
            const sandbox = sandboxMap.get(name);
            const inProgress = sandbox ? IN_PROGRESS.has(sandbox.status) : false;

            return (
              <div
                key={name}
                className="relative group border rounded-lg hover:border-primary/40 hover:bg-muted/30 transition-colors"
              >
                <Link
                  href={inProgress ? `/environments/${name}/progress` : `/environments/${name}`}
                  className="block p-4"
                >
                  <div className="flex items-start justify-between gap-2 pr-6">
                    <span className="font-medium text-sm truncate">{name}</span>
                    {sandbox && (
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full shrink-0 font-medium ${
                          STATUS_COLORS[sandbox.status] ?? "bg-gray-100 text-gray-500"
                        }`}
                      >
                        {sandbox.status}
                      </span>
                    )}
                  </div>

                  <p className="text-xs text-muted-foreground mt-3">
                    {inProgress
                      ? sandbox?.status === "queued"
                        ? "Waiting for worker…"
                        : "Generating environment…"
                      : sandbox
                      ? `Expires ${new Date(sandbox.expires_at).toLocaleDateString()}`
                      : "File-based environment"}
                  </p>

                  <div className="mt-4 flex gap-2 text-xs text-muted-foreground">
                    {sandbox?.status === "running" && (
                      <span className="text-green-600 font-medium">● Live</span>
                    )}
                    <span>{inProgress ? "View progress →" : "View controls →"}</span>
                  </div>
                </Link>

                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={(e) => handleDelete(name, !!sandbox, e)}
                    disabled={deleting === name}
                    title={`Delete ${name}`}
                    className="p-1 text-muted-foreground hover:text-red-600 hover:bg-red-50 rounded transition-colors disabled:opacity-50"
                  >
                    {deleting === name ? (
                      <span className="text-xs">…</span>
                    ) : (
                      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M2 3.5h10M5 3.5V2.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v1M11.5 3.5l-.7 8a1 1 0 0 1-1 .9H4.2a1 1 0 0 1-1-.9l-.7-8" />
                        <path d="M5.5 6.5v3M8.5 6.5v3" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
