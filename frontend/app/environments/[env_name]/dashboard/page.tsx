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
  pending:   "bg-yellow-100 text-yellow-700",
  running:   "bg-blue-100 text-blue-700",
  completed: "bg-green-100 text-green-700",
  failed:    "bg-red-100 text-red-600",
  stopped:   "bg-gray-100 text-gray-500",
};

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="border rounded-lg p-5">
      <p className="text-xs text-muted-foreground uppercase tracking-widest mb-1">{label}</p>
      <p className="text-2xl font-semibold">{value}</p>
      {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
    </div>
  );
}

function TermBar({ reason, count, total }: { reason: string; count: number; total: number }) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  const colors: Record<string, string> = {
    success:       "bg-green-500",
    max_steps:     "bg-yellow-400",
    diverged:      "bg-orange-400",
    dead_end:      "bg-gray-300",
    loop_detected: "bg-red-400",
    stuck_failing: "bg-red-300",
  };
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-muted-foreground w-28 shrink-0 truncate" title={reason}>{reason}</span>
      <div className="flex-1 h-2 bg-muted rounded overflow-hidden">
        <div
          className={`h-full rounded ${colors[reason] ?? "bg-blue-400"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-medium w-10 text-right">{count}</span>
    </div>
  );
}

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

  // Auto-refresh while any run is still in progress.
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
          <p className="text-sm text-muted-foreground mt-1">{envName}</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => load(true)}
            disabled={refreshing}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors disabled:opacity-40"
            title="Refresh"
          >
            {refreshing ? "Refreshing…" : "↻ Refresh"}
          </button>
          <Link
            href={`/environments/${envName}`}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            ← {envName}
          </Link>
        </div>
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading metrics…</p>
      ) : totalRuns === 0 ? (
        <div className="border rounded-lg p-10 text-center text-sm text-muted-foreground">
          No agent runs yet.{" "}
          <Link href={`/environments/${envName}/agent`} className="text-primary hover:underline">
            Launch a run →
          </Link>
        </div>
      ) : (
        <>
          {/* KPI row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard
              label="Agent Runs"
              value={String(totalRuns)}
              sub={`${totalEps} total episodes`}
            />
            <StatCard
              label="Pass Rate"
              value={`${passRate.toFixed(1)}%`}
              sub={`${successCount} / ${completed.length} completed`}
            />
            <StatCard
              label="Avg Reward"
              value={avgReward.toFixed(3)}
              sub="across completed episodes"
            />
            <StatCard
              label="Avg Steps"
              value={avgSteps.toFixed(1)}
              sub="per completed episode"
            />
          </div>

          {/* Termination breakdown */}
          {Object.keys(termCounts).length > 0 && (
            <div>
              <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-widest mb-3">
                Termination Reasons
              </h2>
              <div className="border rounded-lg p-4 space-y-3">
                {Object.entries(termCounts)
                  .sort((a, b) => b[1] - a[1])
                  .map(([reason, count]) => (
                    <TermBar key={reason} reason={reason} count={count} total={completed.length} />
                  ))}
              </div>
            </div>
          )}

          {/* Recent runs */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
                Recent Runs
              </h2>
              <Link
                href={`/environments/${envName}/agent`}
                className="text-xs text-primary hover:underline"
              >
                View all →
              </Link>
            </div>
            <div className="border rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/40 text-xs text-muted-foreground">
                    <th className="px-4 py-2.5 text-left">Objective</th>
                    <th className="px-4 py-2.5 text-left">Status</th>
                    <th className="px-4 py-2.5 text-left">Progress</th>
                    <th className="px-4 py-2.5 text-left">Started</th>
                  </tr>
                </thead>
                <tbody>
                  {recentRuns.map((run) => (
                    <tr
                      key={run.id}
                      className="border-b last:border-0 hover:bg-muted/20 cursor-pointer"
                      onClick={() =>
                        (window.location.href = `/environments/${envName}/agent`)
                      }
                    >
                      <td className="px-4 py-3 max-w-xs">
                        <p className="truncate text-sm">{run.objective}</p>
                        <p className="text-xs text-muted-foreground font-mono mt-0.5">{run.id.slice(0, 8)}</p>
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={`px-2 py-0.5 rounded text-xs font-medium ${
                            STATUS_BADGE[run.status] ?? "bg-gray-100 text-gray-500"
                          }`}
                        >
                          {run.status}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-20 h-1.5 bg-muted rounded overflow-hidden">
                            <div
                              className="h-full bg-blue-500 rounded"
                              style={{
                                width: `${run.num_episodes > 0 ? (run.episodes_completed / run.num_episodes) * 100 : 0}%`,
                              }}
                            />
                          </div>
                          <span className="text-xs text-muted-foreground">
                            {run.episodes_completed}/{run.num_episodes}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-xs text-muted-foreground whitespace-nowrap">
                        {new Date(run.created_at).toLocaleString()}
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
