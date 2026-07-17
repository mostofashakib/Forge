"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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

interface DetectionFinding {
  category: string;
  severity: "high" | "medium" | "low";
  episode_ids: string[];
  description: string;
  evidence: string;
}

interface DetectionResult {
  episodes_analysed: number;
  is_clean: boolean;
  summary: string;
  findings: DetectionFinding[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-red-100 text-red-700",
  high:     "bg-red-100 text-red-700",
  medium:   "bg-orange-100 text-orange-700",
  low:      "bg-gray-100 text-gray-500",
  info:     "bg-blue-100 text-blue-700",
};

const CATEGORY_LABELS: Record<string, string> = {
  reward_hacking:      "Reward Hacking",
  distribution_drift:  "Distribution Drift",
  policy_gaming:       "Policy Gaming",
  anomalous_pattern:   "Anomalous Pattern",
  reward_collapse:     "Reward Collapse",
};

const CATEGORY_COLORS: Record<string, string> = {
  reward_hacking:      "bg-red-50 border-red-200",
  distribution_drift:  "bg-orange-50 border-orange-200",
  policy_gaming:       "bg-purple-50 border-purple-200",
  anomalous_pattern:   "bg-blue-50 border-blue-200",
  reward_collapse:     "bg-yellow-50 border-yellow-200",
};

const SEVERITIES = ["all", "critical", "high", "medium", "low", "info"] as const;

function fmt(ts: string) {
  return new Date(ts).toLocaleString();
}

// ---------------------------------------------------------------------------
// Detection panel
// ---------------------------------------------------------------------------

function DetectionPanel({ envName }: { envName: string }) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState("");

  async function runDetection() {
    setRunning(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${envName}/detect`, {
        method: "POST",
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setResult(data as DetectionResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detection failed");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-muted/30 border-b flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold">Trajectory Analysis</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Detect reward hacking, distribution drift, policy gaming, and anomalous patterns in agent trajectories.
          </p>
        </div>
        <button
          onClick={runDetection}
          disabled={running}
          className="px-3 py-1.5 text-xs font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors shrink-0"
        >
          {running ? "Analysing…" : "Run Detection"}
        </button>
      </div>

      {/* Results */}
      {(result || error) && (
        <div className="p-4 space-y-3">
          {error && <p className="text-xs text-red-600">{error}</p>}

          {result && (
            <>
              {/* Summary banner */}
              <div className={`rounded-lg px-4 py-3 text-sm ${result.is_clean ? "bg-green-50 text-green-800 border border-green-200" : "bg-orange-50 text-orange-800 border border-orange-200"}`}>
                <span className="font-medium">{result.is_clean ? "✓ No issues detected" : `${result.findings.length} issue${result.findings.length !== 1 ? "s" : ""} detected`}</span>
                {" — "}{result.summary}
                <span className="text-xs opacity-70 ml-2">({result.episodes_analysed} episodes analysed)</span>
              </div>

              {/* Findings */}
              {result.findings.map((f, i) => (
                <div key={i} className={`rounded-lg border p-4 ${CATEGORY_COLORS[f.category] ?? "bg-gray-50 border-gray-200"}`}>
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs font-semibold uppercase tracking-wide">
                        {CATEGORY_LABELS[f.category] ?? f.category}
                      </span>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${SEVERITY_COLORS[f.severity] ?? SEVERITY_COLORS.low}`}>
                        {f.severity}
                      </span>
                      {f.episode_ids.length > 0 && (
                        <span className="text-xs text-muted-foreground font-mono">
                          {f.episode_ids.join(", ")}
                        </span>
                      )}
                    </div>
                  </div>
                  <p className="text-sm mb-1">{f.description}</p>
                  <p className="text-xs text-muted-foreground border-l-2 border-current pl-2 opacity-70">{f.evidence}</p>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

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

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

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
          <h1 className="text-xl font-semibold tracking-tight">Violations</h1>
          <p className="text-sm text-muted-foreground mt-1">
            {violations.length} policy violation{violations.length !== 1 ? "s" : ""} · {logs.length} total audit entries
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

      {/* Detection panel */}
      <DetectionPanel envName={envName} />

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
            className="text-xs text-muted-foreground hover:text-foreground underline"
          >
            Clear filters
          </button>
        )}
        <span className="ml-auto text-xs text-muted-foreground tabular-nums">
          {filtered.length} result{filtered.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Violations table */}
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
                  <td className="px-4 py-3 text-xs text-muted-foreground whitespace-nowrap">{fmt(log.created_at)}</td>
                  <td className="px-4 py-3 font-mono text-xs text-muted-foreground">{log.episode_id.slice(0, 8)}</td>
                  <td className="px-4 py-3 text-xs">{log.step_index ?? "—"}</td>
                  <td className="px-4 py-3 font-mono text-xs">{log.action_type ?? "—"}</td>
                  <td className="px-4 py-3 font-mono text-xs">{log.rule_id ?? "—"}</td>
                  <td className="px-4 py-3">
                    {log.severity ? (
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${SEVERITY_COLORS[log.severity.toLowerCase()] ?? "bg-gray-100 text-gray-500"}`}>
                        {log.severity}
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-xs max-w-xs truncate" title={log.message ?? undefined}>{log.message ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
