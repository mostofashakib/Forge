"use client";

import { useEffect, useState } from "react";
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
    success: "bg-green-500",
    max_steps: "bg-yellow-400",
    diverged: "bg-orange-400",
    dead_end: "bg-gray-300",
  };
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-muted-foreground w-20 shrink-0">{reason}</span>
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

  useEffect(() => {
    async function load() {
      const runsRes = await fetch(`${API_BASE}/api/sandbox/${envName}/agent-runs`).then((r) =>
        r.ok ? r.json() : []
      );
      setRuns(runsRes);

      const allEps: AgentEpisode[] = [];
      for (const run of runsRes) {
        const eps = await fetch(
          `${API_BASE}/api/sandbox/${envName}/agent-runs/${run.id}/episodes`
        )
          .then((r) => (r.ok ? r.json() : []))
          .catch(() => []);
        allEps.push(...eps);
      }
      setEpisodes(allEps);
      setLoading(false);
    }
    load();
  }, [envName]);

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

  const recentRuns = [...runs].slice(0, 5);

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Dashboard</h1>
          <p className="text-sm text-muted-foreground mt-1">{envName}</p>
        </div>
        <Link
          href={`/environments/${envName}`}
          className="text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          ← {envName}
        </Link>
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
            <StatCard label="Agent Runs" value={String(totalRuns)} />
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
              sub={`${totalEps} total episodes`}
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
                  </tr>
                </thead>
                <tbody>
                  {recentRuns.map((run) => (
                    <tr key={run.id} className="border-b last:border-0 hover:bg-muted/20">
                      <td className="px-4 py-3 max-w-xs truncate">{run.objective}</td>
                      <td className="px-4 py-3">
                        <span className="text-xs text-muted-foreground">{run.status}</span>
                      </td>
                      <td className="px-4 py-3 text-xs text-muted-foreground">
                        {run.episodes_completed}/{run.num_episodes}
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
