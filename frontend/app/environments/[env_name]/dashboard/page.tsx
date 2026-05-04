"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

interface AgentRun {
  id: string;
  objective: string;
  status: string;
  num_episodes: number;
  episodes_completed: number;
  created_at: string;
  completed_at: string | null;
}

interface AgentEpisode {
  id: string;
  run_id: string;
  status: string;
  total_steps: number;
  total_reward: number;
  final_objective_score: number;
  termination_reason: string | null;
}

const STATUS_BADGE: Record<string, string> = {
  pending:   "bg-amber-50 text-amber-700 ring-1 ring-amber-200",
  running:   "bg-blue-50 text-blue-700 ring-1 ring-blue-200",
  completed: "bg-green-50 text-green-700 ring-1 ring-green-200",
  failed:    "bg-red-50 text-red-600 ring-1 ring-red-200",
  stopped:   "bg-slate-50 text-slate-500 ring-1 ring-slate-200",
};

// ---------------------------------------------------------------------------
// KPI card
// ---------------------------------------------------------------------------

function KpiCard({
  label,
  value,
  sub,
  icon,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  icon: React.ReactNode;
  accent: string;
}) {
  return (
    <div className="kpi-card flex gap-4 items-start">
      <div className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${accent}`}>
        {icon}
      </div>
      <div className="min-w-0">
        <p className="text-xs text-muted-foreground font-medium mb-0.5">{label}</p>
        <p className="text-2xl font-semibold tracking-tight leading-none">{value}</p>
        {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Termination bar
// ---------------------------------------------------------------------------

const TERM_COLORS: Record<string, string> = {
  success:       "bg-green-500",
  max_steps:     "bg-amber-400",
  diverged:      "bg-orange-400",
  dead_end:      "bg-slate-300",
  loop_detected: "bg-red-400",
  stuck_failing: "bg-red-300",
};

function TermBar({ reason, count, total }: { reason: string; count: number; total: number }) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-muted-foreground w-28 shrink-0 truncate" title={reason}>
        {reason.replace(/_/g, " ")}
      </span>
      <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${TERM_COLORS[reason] ?? "bg-primary/60"} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-medium tabular-nums w-8 text-right text-muted-foreground">{count}</span>
      <span className="text-xs text-muted-foreground/50 w-9 text-right tabular-nums">{pct.toFixed(0)}%</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function DashboardPage() {
  const params = useParams<{ env_name: string }>();
  const envName = params.env_name;

  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [episodes, setEpisodes] = useState<AgentEpisode[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    else setRefreshing(true);

    const runsData: AgentRun[] = await fetch(
      `${API_BASE}/api/sandbox/${envName}/agent-runs`
    )
      .then((r) => (r.ok ? r.json() : []))
      .catch(() => []);
    setRuns(runsData);

    const allEps: AgentEpisode[] = [];
    await Promise.all(
      runsData.map(async (run) => {
        const eps: AgentEpisode[] = await fetch(
          `${API_BASE}/api/sandbox/${envName}/agent-runs/${run.id}/episodes`
        )
          .then((r) => (r.ok ? r.json() : []))
          .catch(() => []);
        allEps.push(...eps);
      })
    );
    setEpisodes(allEps);

    if (!silent) setLoading(false);
    else setRefreshing(false);
  }, [envName]);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    const hasActive = runs.some((r) => r.status === "running" || r.status === "pending");
    if (!hasActive) return;
    const t = setInterval(() => load(true), 5000);
    return () => clearInterval(t);
  }, [runs, load]);

  const completed = episodes.filter((e) => e.status === "completed");
  const totalRuns = runs.length;
  const totalEps = episodes.length;
  const successCount = completed.filter((e) => e.termination_reason === "success").length;
  const passRate = completed.length > 0 ? (successCount / completed.length) * 100 : 0;
  const avgReward =
    completed.length > 0
      ? completed.reduce((s, e) => s + e.total_reward, 0) / completed.length
      : 0;
  const avgSteps =
    completed.length > 0
      ? completed.reduce((s, e) => s + e.total_steps, 0) / completed.length
      : 0;

  const termCounts: Record<string, number> = {};
  for (const ep of completed) {
    const r = ep.termination_reason ?? "unknown";
    termCounts[r] = (termCounts[r] ?? 0) + 1;
  }

  const recentRuns = runs.slice(0, 10);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Dashboard</h1>
          <p className="text-sm text-muted-foreground mt-0.5">{envName}</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => load(true)}
            disabled={refreshing}
            className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors disabled:opacity-40"
          >
            <svg
              width="13" height="13" viewBox="0 0 13 13" fill="none"
              stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
              className={refreshing ? "animate-spin" : ""}
            >
              <path d="M11 6.5A4.5 4.5 0 1 1 9.5 3L11 1.5" />
              <path d="M11 1.5v3h-3" />
            </svg>
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
          <Link href={`/environments/${envName}`} className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            ← {envName}
          </Link>
        </div>
      </div>

      {loading ? (
        <div className="py-20 flex flex-col items-center gap-3">
          <div className="w-5 h-5 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
          <p className="text-sm text-muted-foreground">Loading metrics…</p>
        </div>
      ) : totalRuns === 0 ? (
        <div className="border border-dashed border-border rounded-xl p-14 text-center">
          <p className="text-sm font-medium text-foreground mb-1">No agent runs yet</p>
          <p className="text-xs text-muted-foreground mb-5">Run agents to start collecting metrics.</p>
          <Link
            href={`/environments/${envName}/agent`}
            className="text-sm text-primary font-medium hover:underline"
          >
            Launch a run →
          </Link>
        </div>
      ) : (
        <>
          {/* KPI + Termination side-by-side */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* KPI cards — left 8 cols */}
            <div className="lg:col-span-8 grid grid-cols-2 gap-4">
              <KpiCard
                label="Agent Runs"
                value={String(totalRuns)}
                sub={`${totalEps} total episodes`}
                accent="bg-indigo-50 text-indigo-600"
                icon={
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <polygon points="4,2.5 13.5,8 4,13.5" fill="currentColor" />
                  </svg>
                }
              />
              <KpiCard
                label="Pass Rate"
                value={`${passRate.toFixed(1)}%`}
                sub={`${successCount} / ${completed.length} completed`}
                accent="bg-green-50 text-green-600"
                icon={
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="8" cy="8" r="6" />
                    <path d="M5.5 8.5l2 2 3.5-4" strokeWidth="1.4" />
                  </svg>
                }
              />
              <KpiCard
                label="Avg Reward"
                value={avgReward.toFixed(3)}
                sub="per completed episode"
                accent="bg-amber-50 text-amber-600"
                icon={
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M8 1.5l1.7 3.4 3.8.55-2.75 2.68.65 3.77L8 9.85l-3.4 1.99.65-3.77L2.5 5.45l3.8-.55L8 1.5Z" />
                  </svg>
                }
              />
              <KpiCard
                label="Avg Steps"
                value={avgSteps.toFixed(1)}
                sub="per completed episode"
                accent="bg-violet-50 text-violet-600"
                icon={
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <rect x="1.5" y="9.5" width="3.5" height="5" rx="0.5" fill="currentColor" />
                    <rect x="6.25" y="6.5" width="3.5" height="8" rx="0.5" fill="currentColor" />
                    <rect x="11" y="3.5" width="3.5" height="11" rx="0.5" fill="currentColor" />
                  </svg>
                }
              />
            </div>

            {/* Termination breakdown — right 4 cols */}
            {Object.keys(termCounts).length > 0 ? (
              <div className="lg:col-span-4 border border-border/60 rounded-xl p-5 bg-card card-shadow flex flex-col">
                <h2 className="section-label mb-4">Termination Reasons</h2>
                <div className="space-y-3.5 flex-1">
                  {Object.entries(termCounts)
                    .sort((a, b) => b[1] - a[1])
                    .map(([reason, count]) => (
                      <TermBar key={reason} reason={reason} count={count} total={completed.length} />
                    ))}
                </div>
              </div>
            ) : (
              <div className="lg:col-span-4 border border-border/40 rounded-xl p-5 bg-card/50 flex items-center justify-center">
                <p className="text-xs text-muted-foreground">No completed episodes yet</p>
              </div>
            )}
          </div>

          {/* Recent runs */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="section-label">Recent Runs</h2>
              <Link href={`/environments/${envName}/agent`} className="text-xs text-primary hover:underline font-medium">
                View all →
              </Link>
            </div>
            <div className="border border-border/60 rounded-xl overflow-hidden bg-card card-shadow">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/60 bg-muted/30">
                    <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground">Objective</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground">Status</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground">Progress</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground">Started</th>
                  </tr>
                </thead>
                <tbody>
                  {recentRuns.map((run, i) => (
                    <tr
                      key={run.id}
                      className={`border-b border-border/40 last:border-0 hover:bg-muted/20 cursor-pointer transition-colors ${i % 2 === 0 ? "" : "bg-muted/10"}`}
                      onClick={() => (window.location.href = `/environments/${envName}/agent`)}
                    >
                      <td className="px-4 py-3 max-w-xs">
                        <p className="truncate text-sm font-medium">{run.objective}</p>
                        <p className="text-xs text-muted-foreground/60 font-mono mt-0.5">{run.id.slice(0, 8)}</p>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${STATUS_BADGE[run.status] ?? STATUS_BADGE.stopped}`}>
                          {run.status}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-20 h-1.5 bg-muted rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary/70 rounded-full transition-all"
                              style={{
                                width: `${run.num_episodes > 0 ? (run.episodes_completed / run.num_episodes) * 100 : 0}%`,
                              }}
                            />
                          </div>
                          <span className="text-xs text-muted-foreground tabular-nums">
                            {run.episodes_completed}/{run.num_episodes}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-xs text-muted-foreground whitespace-nowrap">
                        {new Date(run.created_at).toLocaleString("en-US", {
                          month: "short", day: "numeric",
                          hour: "2-digit", minute: "2-digit",
                        })}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
