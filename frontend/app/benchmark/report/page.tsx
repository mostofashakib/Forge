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

interface GraphData {
  k_labels: number[];
  pass_at_k: number[];
  pass_pow_k: number[];
  cohens_kappa: number;
  gap_pass_vs_pow: number[];
  diagnostics: {
    reward_hacking_risk: string;
    memorization_risk: string;
    contamination_risk: string;
    cohens_agreement_rate: number;
    memorization_performance_gap: number;
    robustness_gap_k: number;
    genuine_learning_index: number;
  };
  chart_configs: Record<string, any>;
}

function colorClass(value: number, inverted = false): string {
  const v = inverted ? 1 - value : value;
  if (v >= 0.7) return "text-green-600 font-medium";
  if (v >= 0.4) return "text-amber-600 font-medium";
  return "text-red-500 font-medium";
}

function riskBadgeClass(risk: string) {
  if (risk === "high") return "bg-red-500/10 text-red-500 border-red-500/20";
  if (risk === "moderate") return "bg-amber-500/10 text-amber-500 border-amber-500/20";
  return "bg-emerald-500/10 text-emerald-500 border-emerald-500/20";
}

function fmt(v: number) {
  return v.toFixed(3);
}

export default function BenchmarkReportPage() {
  const [runId, setRunId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[] | null>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [graphOption, setGraphOption] = useState<"all" | "pass_curve" | "agreement_bar" | "risk_breakdown">("all");
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
        if (!latest) {
          setLoading(false);
          return;
        }

        setRunId(latest.id);
        setCreatedAt(latest.created_at);

        const reportRes = await fetch(`${API_BASE}/api/benchmark/runs/${latest.id}/report`, { cache: "no-store" });
        if (!reportRes.ok) throw new Error("Failed to fetch report");
        setMetrics(await reportRes.json());

        const graphRes = await fetch(`${API_BASE}/api/benchmark/runs/${latest.id}/graphs`, { cache: "no-store" });
        if (graphRes.ok) {
          setGraphData(await graphRes.json());
        }
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
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Quality & Learning Diagnostics Report</h1>
          <p className="text-sm text-muted-foreground mt-1">Run {runId} • {runDate}</p>
        </div>
        <button
          onClick={handleDownload}
          className="shrink-0 border rounded-lg px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
        >
          ↓ Download CSV
        </button>
      </div>

      {/* Diagnostics & Metric Cards */}
      {graphData && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold tracking-tight">Learning & Contamination Analytics</h2>
            <div className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground">Graph view:</span>
              {(["all", "pass_curve", "agreement_bar", "risk_breakdown"] as const).map((opt) => (
                <button
                  key={opt}
                  onClick={() => setGraphOption(opt)}
                  className={`px-2.5 py-1 rounded-md border transition-colors ${
                    graphOption === opt
                      ? "bg-primary text-primary-foreground border-primary"
                      : "border-border text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {opt === "all" ? "All Graphs" : opt === "pass_curve" ? "Pass@k Curves" : opt === "agreement_bar" ? "Cohen's Agreement" : "Risk Breakdown"}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="border rounded-xl p-4 bg-card shadow-sm space-y-1">
              <div className="text-xs text-muted-foreground uppercase font-medium">Genuine Learning Index</div>
              <div className="text-2xl font-bold font-mono text-emerald-500">
                {(graphData.diagnostics.genuine_learning_index * 100).toFixed(1)}%
              </div>
              <p className="text-[11px] text-muted-foreground">Balances pass@1, robustness and unseen generalization.</p>
            </div>

            <div className="border rounded-xl p-4 bg-card shadow-sm space-y-1">
              <div className="text-xs text-muted-foreground uppercase font-medium">Cohen&apos;s Kappa (κ)</div>
              <div className="text-2xl font-bold font-mono text-blue-500">
                {graphData.diagnostics.cohens_agreement_rate.toFixed(3)}
              </div>
              <p className="text-[11px] text-muted-foreground">Inter-evaluator and paraphrase agreement.</p>
            </div>

            <div className="border rounded-xl p-4 bg-card shadow-sm space-y-2">
              <div className="text-xs text-muted-foreground uppercase font-medium">Reward Hacking Risk</div>
              <span className={`inline-block text-xs px-2.5 py-0.5 rounded-full border font-semibold uppercase ${riskBadgeClass(graphData.diagnostics.reward_hacking_risk)}`}>
                {graphData.diagnostics.reward_hacking_risk}
              </span>
              <p className="text-[11px] text-muted-foreground">Gap (pass@k - pass^k): {graphData.diagnostics.robustness_gap_k.toFixed(3)}</p>
            </div>

            <div className="border rounded-xl p-4 bg-card shadow-sm space-y-2">
              <div className="text-xs text-muted-foreground uppercase font-medium">Memorization vs Contamination</div>
              <div className="flex gap-2">
                <span className={`text-[11px] px-2 py-0.5 rounded-full border font-semibold ${riskBadgeClass(graphData.diagnostics.memorization_risk)}`}>
                  Mem: {graphData.diagnostics.memorization_risk}
                </span>
                <span className={`text-[11px] px-2 py-0.5 rounded-full border font-semibold ${riskBadgeClass(graphData.diagnostics.contamination_risk)}`}>
                  Contam: {graphData.diagnostics.contamination_risk}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground">Paraphrase drop: {(graphData.diagnostics.memorization_performance_gap * 100).toFixed(1)}%</p>
            </div>
          </div>

          {/* Graphical Visualizations */}
          {(graphOption === "all" || graphOption === "pass_curve") && (
            <div className="border rounded-xl p-5 bg-card shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-sm">Pass@k vs Pass^k Robustness Curve</h3>
                  <p className="text-xs text-muted-foreground">
                    Distinguishes genuine reliable learning from lucky guesses or reward exploitation.
                  </p>
                </div>
                <div className="flex gap-4 text-xs font-mono">
                  <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-emerald-500" /> pass@k (at least 1)</span>
                  <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-indigo-500" /> pass^k (all k pass)</span>
                  <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-amber-500" /> Robustness Gap</span>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-5 gap-3 pt-2">
                {graphData.k_labels.map((k, idx) => (
                  <div key={k} className="border rounded-lg p-3 bg-muted/20 space-y-2 text-xs">
                    <div className="font-semibold text-muted-foreground">k = {k}</div>
                    <div className="space-y-1">
                      <div className="flex justify-between font-mono">
                        <span className="text-emerald-500">pass@{k}:</span>
                        <span>{(graphData.pass_at_k[idx] * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-muted rounded-full h-1.5 overflow-hidden">
                        <div className="bg-emerald-500 h-1.5 rounded-full" style={{ width: `${graphData.pass_at_k[idx] * 100}%` }} />
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="flex justify-between font-mono">
                        <span className="text-indigo-500">pass^{k}:</span>
                        <span>{(graphData.pass_pow_k[idx] * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-muted rounded-full h-1.5 overflow-hidden">
                        <div className="bg-indigo-500 h-1.5 rounded-full" style={{ width: `${graphData.pass_pow_k[idx] * 100}%` }} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Environment Quality Table */}
      <div className="space-y-3">
        <h2 className="text-lg font-semibold tracking-tight">Environment Quality Summary</h2>
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
    </div>
  );
}
