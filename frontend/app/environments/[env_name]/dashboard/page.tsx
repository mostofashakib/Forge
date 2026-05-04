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

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------

const STATUS_BADGE: Record<string, { bg: string; text: string }> = {
  pending:   { bg: "bg-amber-50  ring-1 ring-amber-200",  text: "text-amber-700" },
  running:   { bg: "bg-blue-50   ring-1 ring-blue-200",   text: "text-blue-700"  },
  completed: { bg: "bg-green-50  ring-1 ring-green-200",  text: "text-green-700" },
  failed:    { bg: "bg-red-50    ring-1 ring-red-200",    text: "text-red-600"   },
  stopped:   { bg: "bg-slate-50  ring-1 ring-slate-200",  text: "text-slate-500" },
};

const STATUS_BAR: Record<string, string> = {
  running:   "bg-blue-500",
  completed: "bg-green-500",
  failed:    "bg-red-500",
  pending:   "bg-amber-400",
  stopped:   "bg-slate-300",
};

// ---------------------------------------------------------------------------
// KPI card — icon + label on top, large value, sub text below
// ---------------------------------------------------------------------------

function KpiCard({
  label,
  value,
  sub,
  icon,
  bgIcon,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  icon: React.ReactNode;
  bgIcon: React.ReactNode;
  accent: string;
}) {
  return (
    <div className="relative overflow-hidden border border-border/60 rounded-xl p-5 bg-card card-shadow">
      {/* Watermark */}
      <div className="absolute top-3 right-3 opacity-[0.06] pointer-events-none w-16 h-16 flex items-center justify-center">
        {bgIcon}
      </div>
      {/* Icon + label */}
      <div className="flex items-center gap-2.5 mb-4 relative">
        <div className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${accent}`}>
          {icon}
        </div>
        <span className="text-sm font-medium text-muted-foreground">{label}</span>
      </div>
      {/* Value */}
      <p className="text-3xl font-bold tracking-tight leading-none mb-2 relative">{value}</p>
      {/* Sub */}
      {sub && <p className="text-xs text-muted-foreground/70 relative">{sub}</p>}
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

const TERM_PCT_COLORS: Record<string, string> = {
  success:       "text-green-600",
  max_steps:     "text-amber-600",
  diverged:      "text-orange-500",
  dead_end:      "text-slate-400",
  loop_detected: "text-red-500",
  stuck_failing: "text-red-400",
};

function TermBar({ reason, count, total }: { reason: string; count: number; total: number }) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground capitalize">{reason.replace(/_/g, " ")}</span>
        <span className={`text-xs font-semibold tabular-nums ${TERM_PCT_COLORS[reason] ?? "text-muted-foreground"}`}>
          {pct.toFixed(0)}%
        </span>
      </div>
      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${TERM_COLORS[reason] ?? "bg-primary/60"} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Duration helper
// ---------------------------------------------------------------------------

function formatDuration(start: string, end: string | null): string {
  if (!end) return "—";
  const ms = new Date(end).getTime() - new Date(start).getTime();
  if (ms < 0) return "—";
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
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

    const runsData: AgentRun[] = await fetch(`${API_BASE}/api/sandbox/${envName}/agent-runs`)
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
  const avgReward = completed.length > 0
    ? completed.reduce((s, e) => s + e.total_reward, 0) / completed.length : 0;
  const avgSteps = completed.length > 0
    ? completed.reduce((s, e) => s + e.total_steps, 0) / completed.length : 0;

  const termCounts: Record<string, number> = {};
  for (const ep of completed) {
    const r = ep.termination_reason ?? "unknown";
    termCounts[r] = (termCounts[r] ?? 0) + 1;
  }

  const recentRuns = runs.slice(0, 10);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Dashboard</h1>
          <p className="text-sm text-muted-foreground mt-0.5 flex items-center gap-1.5">
            <svg width="13" height="13" viewBox="0 0 13 13" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
              <path d="M6.5 1L12 3.5V7C12 10 9.75 11.75 6.5 13C3.25 11.75 1 10 1 7V3.5L6.5 1Z" />
            </svg>
            Environment:{" "}
            <span className="font-medium text-primary">{envName}</span>
          </p>
        </div>
        <button
          onClick={() => load(true)}
          disabled={refreshing}
          className="flex items-center gap-1.5 px-3.5 py-2 text-sm font-medium border border-border rounded-lg hover:bg-muted/50 disabled:opacity-40 transition-colors"
        >
          <svg
            width="13" height="13" viewBox="0 0 13 13" fill="none"
            stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
            className={refreshing ? "animate-spin" : ""}
          >
            <path d="M11 6.5A4.5 4.5 0 1 1 9.5 3L11 1.5" />
            <path d="M11 1.5v3h-3" />
          </svg>
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="py-20 flex flex-col items-center gap-3">
          <div className="w-5 h-5 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
          <p className="text-sm text-muted-foreground">Loading metrics…</p>
        </div>
      ) : totalRuns === 0 ? (
        <div className="border border-dashed border-border rounded-xl p-14 text-center">
          <p className="text-sm font-medium mb-1">No agent runs yet</p>
          <p className="text-xs text-muted-foreground mb-5">Run agents to start collecting metrics.</p>
          <Link href={`/environments/${envName}/agent`} className="text-sm text-primary font-medium hover:underline">
            Launch a run →
          </Link>
        </div>
      ) : (
        <>
          {/* KPI + Termination side-by-side */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            {/* KPI 2×2 grid — left 8 cols */}
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
                bgIcon={
                  <svg width="64" height="64" viewBox="0 0 16 16" fill="none">
                    <polygon points="4,2.5 13.5,8 4,13.5" fill="currentColor" />
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
                bgIcon={
                  <svg width="64" height="64" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M8 1.5l1.7 3.4 3.8.55-2.75 2.68.65 3.77L8 9.85l-3.4 1.99.65-3.77L2.5 5.45l3.8-.55L8 1.5Z" />
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
                bgIcon={
                  <svg width="64" height="64" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="8" cy="8" r="6" />
                    <path d="M5.5 8.5l2 2 3.5-4" strokeWidth="1.4" />
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
                bgIcon={
                  <svg width="64" height="64" viewBox="0 0 16 16" fill="none">
                    <rect x="1.5" y="9.5" width="3.5" height="5" rx="0.5" fill="currentColor" />
                    <rect x="6.25" y="6.5" width="3.5" height="8" rx="0.5" fill="currentColor" />
                    <rect x="11" y="3.5" width="3.5" height="11" rx="0.5" fill="currentColor" />
                  </svg>
                }
              />
            </div>

            {/* Termination reasons — right 4 cols */}
            <div className="lg:col-span-4 border border-border/60 rounded-xl p-5 bg-card card-shadow">
              <div className="flex items-center justify-between mb-5">
                <h2 className="text-sm font-semibold">Termination Reasons</h2>
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                  <circle cx="7" cy="2" r="0.75" fill="currentColor" stroke="none" />
                  <circle cx="7" cy="7" r="0.75" fill="currentColor" stroke="none" />
                  <circle cx="7" cy="12" r="0.75" fill="currentColor" stroke="none" />
                </svg>
              </div>
              {Object.keys(termCounts).length === 0 ? (
                <p className="text-xs text-muted-foreground text-center py-6">No completed episodes</p>
              ) : (
                <div className="space-y-4">
                  {Object.entries(termCounts)
                    .sort((a, b) => b[1] - a[1])
                    .map(([reason, count]) => (
                      <TermBar key={reason} reason={reason} count={count} total={completed.length} />
                    ))}
                </div>
              )}
            </div>
          </div>

          {/* Recent runs */}
          <div className="border border-border/60 rounded-xl overflow-hidden bg-card card-shadow">
            <div className="flex items-center justify-between px-5 py-4 border-b border-border/60">
              <h2 className="text-sm font-semibold">Recent Runs</h2>
              <Link href={`/environments/${envName}/agent`} className="text-xs font-medium text-primary hover:underline">
                View All
              </Link>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/40">
                  <th className="px-5 py-3 text-left text-xs font-semibold text-muted-foreground">Run ID</th>
                  <th className="px-5 py-3 text-left text-xs font-semibold text-muted-foreground">Status</th>
                  <th className="px-5 py-3 text-left text-xs font-semibold text-muted-foreground">Progress</th>
                  <th className="px-5 py-3 text-right text-xs font-semibold text-muted-foreground">Duration</th>
                </tr>
              </thead>
              <tbody>
                {recentRuns.map((run) => {
                  const pct = run.num_episodes > 0 ? (run.episodes_completed / run.num_episodes) * 100 : 0;
                  const badge = STATUS_BADGE[run.status] ?? STATUS_BADGE.stopped;
                  const barColor = STATUS_BAR[run.status] ?? "bg-slate-300";
                  return (
                    <tr
                      key={run.id}
                      className="border-b border-border/30 last:border-0 hover:bg-muted/30 transition-colors cursor-pointer"
                      onClick={() => (window.location.href = `/environments/${envName}/agent`)}
                    >
                      <td className="px-5 py-4">
                        <span className="text-sm font-medium text-primary font-mono">
                          {run.id.slice(0, 8)}
                        </span>
                        <p className="text-xs text-muted-foreground/60 mt-0.5 truncate max-w-40">{run.objective}</p>
                      </td>
                      <td className="px-5 py-4">
                        <span className={`inline-flex px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wide ${badge.bg} ${badge.text}`}>
                          {run.status}
                        </span>
                      </td>
                      <td className="px-5 py-4">
                        <div className="flex items-center gap-3">
                          <div className="flex-1 max-w-36 h-1.5 bg-muted rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${barColor} transition-all`}
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                          <span className="text-xs text-muted-foreground tabular-nums w-8">{pct.toFixed(0)}%</span>
                        </div>
                      </td>
                      <td className="px-5 py-4 text-right">
                        <span className="text-sm font-mono text-muted-foreground tabular-nums">
                          {formatDuration(run.created_at, run.completed_at)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
