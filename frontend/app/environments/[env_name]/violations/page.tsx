"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

interface AuditLog {
  id: string;
  episode_id: string;
  step_index: number | null;
  actor: string | null;
  action_type: string | null;
  rule_id: string | null;
  violation: boolean;
  severity: string | null;
  message: string | null;
  created_at: string;
}

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-red-100 text-red-700",
  high:     "bg-orange-100 text-orange-700",
  medium:   "bg-yellow-100 text-yellow-700",
  low:      "bg-gray-100 text-gray-500",
  info:     "bg-blue-100 text-blue-700",
};

const SEVERITIES = ["all", "critical", "high", "medium", "low", "info"] as const;

function fmt(ts: string) {
  return new Date(ts).toLocaleString();
}

export default function ViolationsPage() {
  const params = useParams<{ env_name: string }>();
  const envName = params.env_name;

  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [severity, setSeverity] = useState<string>("all");
  const [episodeFilter, setEpisodeFilter] = useState("");
  const [ruleFilter, setRuleFilter] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    const data: AuditLog[] = await fetch(
      `${API_BASE}/api/audit/?env_name=${encodeURIComponent(envName)}&limit=500`,
      { cache: "no-store" }
    )
      .then((r) => (r.ok ? r.json() : []))
      .catch(() => []);
    setLogs(data);
    setLoading(false);
  }, [envName]);

  useEffect(() => { load(); }, [load]);

  const violations = logs.filter((l) => l.violation);

  const filtered = violations.filter((l) => {
    if (severity !== "all" && (l.severity ?? "").toLowerCase() !== severity) return false;
    if (episodeFilter && !l.episode_id.toLowerCase().includes(episodeFilter.toLowerCase())) return false;
    if (ruleFilter && !(l.rule_id ?? "").toLowerCase().includes(ruleFilter.toLowerCase())) return false;
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Policy Violations</h1>
          <p className="text-sm text-muted-foreground mt-1">
            {violations.length} violation{violations.length !== 1 ? "s" : ""} · {logs.length} total audit entries
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={load}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            title="Refresh"
          >
            ↻ Refresh
          </button>
          <Link
            href={`/environments/${envName}`}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            ← {envName}
          </Link>
        </div>
      </div>

      {/* Filter row */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-muted-foreground mr-0.5">Severity</span>
          {SEVERITIES.map((s) => (
            <button
              key={s}
              onClick={() => setSeverity(s)}
              className={`px-2.5 py-1 rounded text-xs font-medium capitalize transition-colors ${
                severity === s
                  ? "bg-foreground text-background"
                  : "bg-muted text-muted-foreground hover:text-foreground hover:bg-muted/80"
              }`}
            >
              {s}
            </button>
          ))}
        </div>

        <input
          type="text"
          placeholder="Episode ID…"
          value={episodeFilter}
          onChange={(e) => setEpisodeFilter(e.target.value)}
          className="border rounded px-2.5 py-1 text-xs w-36 focus:outline-none focus:ring-1 focus:ring-foreground/30"
        />

        <input
          type="text"
          placeholder="Rule ID…"
          value={ruleFilter}
          onChange={(e) => setRuleFilter(e.target.value)}
          className="border rounded px-2.5 py-1 text-xs w-32 focus:outline-none focus:ring-1 focus:ring-foreground/30"
        />

        {(severity !== "all" || episodeFilter || ruleFilter) && (
          <button
            onClick={() => { setSeverity("all"); setEpisodeFilter(""); setRuleFilter(""); }}
            className="text-xs text-muted-foreground hover:text-foreground underline transition-colors"
          >
            Clear filters
          </button>
        )}

        <span className="ml-auto text-xs text-muted-foreground tabular-nums">
          {filtered.length} result{filtered.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Table */}
      {loading ? (
        <p className="text-sm text-muted-foreground py-10 text-center">Loading…</p>
      ) : filtered.length === 0 ? (
        <div className="border rounded-lg p-10 text-center text-sm text-muted-foreground">
          {violations.length === 0
            ? "No policy violations recorded for this environment."
            : "No violations match the current filters."}
        </div>
      ) : (
        <div className="border rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/40 text-xs text-muted-foreground">
                <th className="px-4 py-2.5 text-left">Time</th>
                <th className="px-4 py-2.5 text-left">Episode</th>
                <th className="px-4 py-2.5 text-left">Step</th>
                <th className="px-4 py-2.5 text-left">Action</th>
                <th className="px-4 py-2.5 text-left">Rule</th>
                <th className="px-4 py-2.5 text-left">Severity</th>
                <th className="px-4 py-2.5 text-left">Message</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((log) => (
                <tr key={log.id} className="border-b last:border-0 hover:bg-muted/20">
                  <td className="px-4 py-3 text-xs text-muted-foreground whitespace-nowrap">
                    {fmt(log.created_at)}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                    {log.episode_id.slice(0, 8)}
                  </td>
                  <td className="px-4 py-3 text-xs">{log.step_index ?? "—"}</td>
                  <td className="px-4 py-3 font-mono text-xs">{log.action_type ?? "—"}</td>
                  <td className="px-4 py-3 font-mono text-xs">{log.rule_id ?? "—"}</td>
                  <td className="px-4 py-3">
                    {log.severity ? (
                      <span
                        className={`px-2 py-0.5 rounded text-xs font-medium ${
                          SEVERITY_COLORS[log.severity.toLowerCase()] ?? "bg-gray-100 text-gray-500"
                        }`}
                      >
                        {log.severity}
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-xs max-w-xs truncate" title={log.message ?? undefined}>
                    {log.message ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
