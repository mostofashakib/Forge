"use client";
import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

interface SandboxInfo {
  id: string;
  status: string;
  expires_at: string;
  ttl_days: number;
}

type Status = "running" | "building" | "queued" | "stopped" | "error";

const STATUS_META: Record<Status, { dot: string; badge: string; label: string }> = {
  running:  { dot: "bg-green-500",               badge: "bg-green-50 text-green-700 ring-1 ring-green-200",   label: "running"  },
  building: { dot: "bg-blue-500 animate-pulse",  badge: "bg-blue-50 text-blue-700 ring-1 ring-blue-200",     label: "building" },
  queued:   { dot: "bg-amber-400 animate-pulse", badge: "bg-amber-50 text-amber-700 ring-1 ring-amber-200",  label: "queued"   },
  stopped:  { dot: "bg-slate-300",               badge: "bg-slate-50 text-slate-500 ring-1 ring-slate-200",  label: "stopped"  },
  error:    { dot: "bg-red-500",                 badge: "bg-red-50 text-red-600 ring-1 ring-red-200",        label: "error"    },
};

const LEFT_ACCENT: Record<Status, string> = {
  running:  "before:bg-green-400",
  building: "before:bg-blue-400",
  queued:   "before:bg-amber-400",
  stopped:  "before:bg-slate-200",
  error:    "before:bg-red-400",
};

const IN_PROGRESS = new Set<Status>(["queued", "building"]);

function formatExpiry(isoDate: string): string {
  const d = new Date(isoDate);
  const now = new Date();
  const diff = Math.round((d.getTime() - now.getTime()) / 86400000);
  if (diff <= 0) return "Expired";
  if (diff === 1) return "Expires tomorrow";
  if (diff <= 7) return `Expires in ${diff}d`;
  return `Expires ${d.toLocaleDateString("en-US", { month: "short", day: "numeric" })}`;
}

export default function EnvironmentsPage() {
  const [allNames, setAllNames] = useState<string[]>([]);
  const [sandboxMap, setSandboxMap] = useState<Map<string, SandboxInfo>>(new Map());
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const signal = AbortSignal.timeout(8000);
      const [fileResponse, sandboxResponse] = await Promise.all([
        fetch(`${API_BASE}/api/envs/`, { signal }),
        fetch(`${API_BASE}/api/sandbox/`, { signal }),
      ]);
      if (!fileResponse.ok || !sandboxResponse.ok) {
        throw new Error(`Backend request failed (${fileResponse.status}/${sandboxResponse.status})`);
      }
      const [fileEnvs, sandboxes]: [string[], SandboxInfo[]] = await Promise.all([
        fileResponse.json(),
        sandboxResponse.json(),
      ]);
      const map = new Map((sandboxes as SandboxInfo[]).map((s) => [s.id, s]));
      const names = Array.from(
        new Set([...(fileEnvs as string[]), ...(sandboxes as SandboxInfo[]).map((s) => s.id)])
      ).sort();
      setSandboxMap(map);
      setAllNames(names);
      setRequestError(null);
    } catch (cause) {
      setRequestError(cause instanceof Error ? cause.message : "Could not reach the backend");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const initialFetch = window.setTimeout(fetchData, 0);
    return () => window.clearTimeout(initialFetch);
  }, [fetchData]);

  async function handleDelete(name: string, hasSandbox: boolean, e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm(`Delete environment "${name}"? This cannot be undone.`)) return;
    setDeleting(name);
    const url = hasSandbox ? `${API_BASE}/api/sandbox/${name}` : `${API_BASE}/api/envs/${name}`;
    try {
      const response = await fetch(url, { method: "DELETE" });
      if (!response.ok) throw new Error(`Delete failed (${response.status})`);
      setAllNames((prev) => prev.filter((n) => n !== name));
      setSandboxMap((prev) => { const next = new Map(prev); next.delete(name); return next; });
      setRequestError(null);
    } catch (cause) {
      setRequestError(cause instanceof Error ? cause.message : "Could not reach the backend");
    } finally {
      setDeleting(null);
    }
  }

  useEffect(() => {
    const anyInProgress = Array.from(sandboxMap.values()).some((s) => IN_PROGRESS.has(s.status as Status));
    if (!anyInProgress) return;
    const id = setInterval(fetchData, 3000);
    return () => clearInterval(id);
  }, [sandboxMap, fetchData]);

  if (loading) {
    return (
      <div className="py-32 flex flex-col items-center gap-3">
        <div className="w-6 h-6 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
        <p className="text-sm text-muted-foreground">Loading environments…</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="blueprint-panel p-6 sm:p-8 flex items-end justify-between gap-6 before:absolute before:top-0 before:left-0 before:h-1 before:w-28 before:bg-primary">
        <div>
          <span className="signal-chip mb-4"><span className="size-1.5 rounded-full bg-foreground animate-pulse" /> system inventory</span>
          <h1 className="text-4xl sm:text-5xl font-semibold tracking-[-0.045em] leading-none">Environments</h1>
          <p className="text-sm text-muted-foreground mt-3 max-w-lg">
            {allNames.length === 0
              ? "Turn application behavior into reproducible training grounds."
              : `${allNames.length} active training ground${allNames.length !== 1 ? "s" : ""} in your foundry.`}
          </p>
        </div>
        <Link
          href="/environments/new"
          className="flex items-center gap-2 px-5 py-3 bg-primary text-primary-foreground text-xs uppercase tracking-[0.12em] font-semibold hover:-translate-y-0.5 shadow-[4px_4px_0_var(--foreground)] transition-all"
        >
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M6 1v10M1 6h10" />
          </svg>
          New
        </Link>
      </div>

      {requestError && (
        <div role="alert" className="border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          Could not update environments: {requestError}
        </div>
      )}

      {allNames.length === 0 ? (
        /* Empty state */
        <div className="flex flex-col items-center justify-center py-24 border-2 border-dashed border-foreground/25 bg-card/70">
          <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center mb-5 ring-1 ring-primary/15">
            <svg width="24" height="24" viewBox="0 0 14 14" fill="none">
              <path d="M7 1L13 4V10L7 13L1 10V4L7 1Z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" className="text-primary" />
              <path d="M7 5L9 6.5V9L7 10.5L5 9V6.5L7 5Z" fill="currentColor" className="text-primary" />
            </svg>
          </div>
          <p className="text-sm font-semibold text-foreground mb-1.5">No environments yet</p>
          <p className="text-xs text-muted-foreground mb-6 text-center max-w-xs">
            Create your first environment to start collecting training data.
          </p>
          <Link
            href="/environments/new"
            className="px-5 py-2.5 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 shadow-sm transition-all"
          >
            Create environment →
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {allNames.map((name) => {
            const sandbox = sandboxMap.get(name);
            const status = (sandbox?.status ?? "stopped") as Status;
            const meta = STATUS_META[status] ?? STATUS_META.stopped;
            const accent = LEFT_ACCENT[status] ?? LEFT_ACCENT.stopped;
            const inProgress = sandbox ? IN_PROGRESS.has(sandbox.status as Status) : false;

            return (
              <div
                key={name}
                className={`relative group card-shadow hover:card-shadow-hover hover:-translate-y-1 transition-all duration-200 bg-card border border-foreground/20 overflow-hidden before:absolute before:left-0 before:top-0 before:bottom-0 before:w-1 ${accent}`}
              >
                <Link
                  href={inProgress ? `/environments/${name}/progress` : `/environments/${name}`}
                  className="block p-5 pr-11"
                >
                  {/* Name row */}
                  <div className="flex items-center justify-between gap-2 mb-2.5">
                    <span className="font-semibold text-sm truncate">{name}</span>
                    {sandbox && (
                      <span className={`inline-flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full shrink-0 font-medium ${meta.badge}`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${meta.dot}`} />
                        {meta.label}
                      </span>
                    )}
                  </div>

                  {/* Sub-line */}
                  <p className="text-xs text-muted-foreground">
                    {inProgress
                      ? status === "queued" ? "Waiting for worker…" : "Generating environment…"
                      : sandbox
                      ? formatExpiry(sandbox.expires_at)
                      : "File-based environment"}
                  </p>

                  {/* Footer */}
                  <div className="mt-4 pt-3.5 border-t border-border/40 flex items-center justify-between">
                    <span className="text-xs text-muted-foreground/60 group-hover:text-muted-foreground transition-colors">
                      {inProgress ? "View progress" : "Open"}
                    </span>
                    <svg
                      width="12" height="12" viewBox="0 0 12 12" fill="none"
                      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
                      className="text-muted-foreground/40 group-hover:text-muted-foreground group-hover:translate-x-0.5 transition-all"
                    >
                      <path d="M2 6h8M6.5 2.5L10 6l-3.5 3.5" />
                    </svg>
                  </div>
                </Link>

                {/* Delete button */}
                <div className="absolute top-4 right-3 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={(e) => handleDelete(name, !!sandbox, e)}
                    disabled={deleting === name}
                    title={`Delete ${name}`}
                    className="p-1.5 text-muted-foreground hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
                  >
                    {deleting === name ? (
                      <span className="text-xs">…</span>
                    ) : (
                      <svg width="13" height="13" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
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
