"use client";

import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

interface QuarantinedTask {
  id: string;
  task_id: string;
  reason: string;
  environment_version: string | null;
  created_at: string;
  resolved: boolean;
}

interface FlaggedVersion {
  id: string;
  environment_version: string;
  reason: string;
  created_at: string;
  resolved: boolean;
}

interface ReliabilityData {
  quarantined_tasks: QuarantinedTask[];
  flagged_versions: FlaggedVersion[];
}

export function ReliabilityPanel({ envName }: { envName: string }) {
  const [data, setData] = useState<ReliabilityData>({ quarantined_tasks: [], flagged_versions: [] });
  const [loading, setLoading] = useState(true);
  const [actioningId, setActioningId] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/reliability/environments/${envName}`);
      if (res.ok) {
        const json = await res.json();
        setData(json);
      }
    } catch {
      // ignore fetch errors
    } finally {
      setLoading(false);
    }
  }, [envName]);

  useEffect(() => {
    load();
  }, [load]);

  const handleRelease = async (id: string, taskId: string) => {
    setActioningId(id);
    try {
      const res = await fetch(`${API_BASE}/api/reliability/quarantine/${id}/release`, {
        method: "POST",
      });
      if (res.ok) {
        setMsg(`Released task ${taskId}`);
        setTimeout(() => setMsg(null), 3000);
        await load();
      }
    } finally {
      setActioningId(null);
    }
  };

  const handleResolveVersion = async (id: string, ver: string) => {
    setActioningId(id);
    try {
      const res = await fetch(`${API_BASE}/api/reliability/flagged-versions/${id}/resolve`, {
        method: "POST",
      });
      if (res.ok) {
        setMsg(`Resolved version ${ver.slice(0, 12)}`);
        setTimeout(() => setMsg(null), 3000);
        await load();
      }
    } finally {
      setActioningId(null);
    }
  };

  const hasIssues = data.quarantined_tasks.length > 0 || data.flagged_versions.length > 0;

  if (loading) {
    return (
      <div className="rounded-xl border border-border bg-card p-6 text-sm text-muted-foreground animate-pulse">
        Loading reliability status…
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className={`w-2.5 h-2.5 rounded-full ${hasIssues ? "bg-amber-500 animate-pulse" : "bg-emerald-500"}`} />
          <h2 className="text-base font-semibold text-foreground">Reliability & Quarantined Tasks</h2>
        </div>
        <button
          onClick={() => {
            setLoading(true);
            load();
          }}
          className="text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          Refresh
        </button>
      </div>

      {msg && (
        <div className="mb-4 px-3 py-2 rounded-lg bg-emerald-50 text-emerald-700 text-xs font-medium border border-emerald-200">
          {msg}
        </div>
      )}

      {!hasIssues ? (
        <p className="text-sm text-muted-foreground">
          No quarantined tasks or flagged versions. All episodes and environments are operating reliably.
        </p>
      ) : (
        <div className="space-y-6">
          {data.quarantined_tasks.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                Quarantined Tasks ({data.quarantined_tasks.length})
              </h3>
              <div className="divide-y divide-border border border-border rounded-lg overflow-hidden bg-background">
                {data.quarantined_tasks.map((task) => (
                  <div key={task.id} className="p-3.5 flex items-center justify-between gap-4">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-medium text-foreground truncate">
                          {task.task_id}
                        </span>
                        {task.environment_version && (
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                            v:{task.environment_version.slice(0, 10)}
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-rose-600 dark:text-rose-400 mt-1 truncate">
                        {task.reason}
                      </p>
                      <span className="text-[10px] text-muted-foreground">
                        Quarantined {new Date(task.created_at).toLocaleString()}
                      </span>
                    </div>
                    <button
                      disabled={actioningId === task.id}
                      onClick={() => handleRelease(task.id, task.task_id)}
                      className="shrink-0 px-3 py-1.5 text-xs font-medium rounded-md bg-secondary text-secondary-foreground hover:bg-secondary/80 disabled:opacity-50 transition-colors"
                    >
                      {actioningId === task.id ? "Releasing…" : "Release"}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {data.flagged_versions.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                Flagged Versions ({data.flagged_versions.length})
              </h3>
              <div className="divide-y divide-border border border-border rounded-lg overflow-hidden bg-background">
                {data.flagged_versions.map((ver) => (
                  <div key={ver.id} className="p-3.5 flex items-center justify-between gap-4">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-semibold text-amber-600 dark:text-amber-400">
                          {ver.environment_version}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1 truncate">
                        {ver.reason}
                      </p>
                      <span className="text-[10px] text-muted-foreground">
                        Flagged {new Date(ver.created_at).toLocaleString()}
                      </span>
                    </div>
                    <button
                      disabled={actioningId === ver.id}
                      onClick={() => handleResolveVersion(ver.id, ver.environment_version)}
                      className="shrink-0 px-3 py-1.5 text-xs font-medium rounded-md bg-secondary text-secondary-foreground hover:bg-secondary/80 disabled:opacity-50 transition-colors"
                    >
                      {actioningId === ver.id ? "Resolving…" : "Resolve"}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
