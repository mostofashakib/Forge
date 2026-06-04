"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

interface RunSummary {
  id: string;
  status: string;
  domains: string;
  created_at: string;
}

interface MetricRow {
  env_name: string;
  state_coverage_score: number;
  reward_density: number;
  dead_end_rate: number;
  action_diversity: number;
  num_episodes: number;
  num_steps: number;
}

function colorClass(value: number, inverted = false): string {
  const v = inverted ? 1 - value : value;
  if (v >= 0.7) return "text-green-600 font-medium";
  if (v >= 0.4) return "text-amber-600 font-medium";
  return "text-red-500 font-medium";
}

function fmt(v: number) { return v.toFixed(3); }

export default function BenchmarkReportPage() {
  const [runId, setRunId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[] | null>(null);
  const [createdAt, setCreatedAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const runsRes = await fetch(`${API_BASE}/api/benchmark/runs`, { cache: "no-store" });
        if (!runsRes.ok) throw new Error("Failed to fetch runs");
        const runs: RunSummary[] = await runsRes.json();
        const latest = runs.find((r) => r.status === "done");
        if (!latest) { setLoading(false); return; }

        setRunId(latest.id);
        setCreatedAt(latest.created_at);

        const reportRes = await fetch(`${API_BASE}/api/benchmark/runs/${latest.id}/report`, { cache: "no-store" });
        if (!reportRes.ok) throw new Error("Failed to fetch report");
        setMetrics(await reportRes.json());
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  function handleDownload() {
    if (!runId) return;
    window.location.href = `${API_BASE}/api/benchmark/runs/${runId}/report/download`;
  }

  if (loading) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold tracking-tight">Quality Report</h1>
        <p className="text-sm text-muted-foreground">Loading…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold tracking-tight">Quality Report</h1>
        <div className="border border-red-200 bg-red-50 rounded-lg p-4">
          <p className="text-sm text-red-600">{error}</p>
        </div>
      </div>
    );
  }

  if (!metrics || metrics.length === 0) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold tracking-tight">Quality Report</h1>
        <div className="border rounded-lg p-8 text-center space-y-3">
          <p className="text-muted-foreground text-sm">No completed benchmark runs yet.</p>
          <Link href="/benchmark/run" className="text-sm text-primary hover:underline font-medium">
            Run a benchmark →
          </Link>
        </div>
      </div>
    );
  }

  const runDate = createdAt ? new Date(createdAt).toLocaleString() : "";

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Quality Report</h1>
          <p className="text-sm text-muted-foreground mt-1">Last run: {runDate}</p>
        </div>
        <button
          onClick={handleDownload}
          className="shrink-0 border rounded-lg px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
        >
          ↓ Download CSV
        </button>
      </div>

      <div className="border rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/40">
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">Environment</th>
              <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">Coverage ↑</th>
              <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">Reward ↑</th>
              <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">Dead-ends ↓</th>
              <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">Diversity ↑</th>
              <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">Episodes</th>
              <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">Steps</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {metrics.map((row) => (
              <tr key={row.env_name} className="hover:bg-muted/20 transition-colors">
                <td className="px-4 py-3 font-medium">{row.env_name}</td>
                <td className={`px-4 py-3 text-right font-mono ${colorClass(row.state_coverage_score)}`}>
                  {fmt(row.state_coverage_score)}
                </td>
                <td className={`px-4 py-3 text-right font-mono ${colorClass(row.reward_density)}`}>
                  {fmt(row.reward_density)}
                </td>
                <td className={`px-4 py-3 text-right font-mono ${colorClass(row.dead_end_rate, true)}`}>
                  {fmt(row.dead_end_rate)}
                </td>
                <td className={`px-4 py-3 text-right font-mono ${colorClass(row.action_diversity)}`}>
                  {fmt(row.action_diversity)}
                </td>
                <td className="px-4 py-3 text-right text-muted-foreground tabular-nums">{row.num_episodes}</td>
                <td className="px-4 py-3 text-right text-muted-foreground tabular-nums">{row.num_steps}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex gap-4 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-green-500" />≥ 0.7 good</span>
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-amber-400" />0.4 – 0.7 fair</span>
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-red-400" />{"< 0.4 poor"}</span>
        <span className="text-muted-foreground/60">(dead-end rate: lower is better)</span>
      </div>
    </div>
  );
}
