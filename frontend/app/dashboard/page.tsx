import Link from "next/link";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface FailureCluster {
  check_name: string;
  count: number;
  episode_ids: string[];
}

interface Stats {
  pass_rate: number;
  avg_reward: number;
  avg_steps: number;
  policy_violation_count: number;
  top_failures: FailureCluster[];
}

interface EpisodeSummary {
  id: string;
  env_name: string;
  status: string;
  passed: boolean;
  total_reward: number;
  total_steps: number;
  started_at: string | null;
}

async function getStats(envName: string): Promise<Stats | null> {
  try {
    const res = await fetch(`${API}/api/envs/${envName}/stats`, { cache: "no-store" });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function getEpisodes(envName: string): Promise<EpisodeSummary[]> {
  try {
    const res = await fetch(`${API}/api/episodes/?env_name=${envName}`, { cache: "no-store" });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export default async function DashboardPage({
  searchParams,
}: {
  searchParams: Promise<{ env?: string }>;
}) {
  const { env } = await searchParams;
  const envName = env ?? "default";

  const [stats, episodes] = await Promise.all([
    getStats(envName),
    getEpisodes(envName),
  ]);

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Episode metrics for{" "}
            <span className="font-mono text-foreground">{envName}</span>
          </p>
        </div>
      </div>

      {stats ? (
        <div className="grid grid-cols-4 gap-3">
          <StatCard
            label="Pass Rate"
            value={`${(stats.pass_rate * 100).toFixed(1)}%`}
            accent={stats.pass_rate >= 0.7 ? "green" : "red"}
          />
          <StatCard label="Avg Reward" value={stats.avg_reward.toFixed(3)} />
          <StatCard label="Avg Steps" value={stats.avg_steps.toFixed(1)} />
          <StatCard
            label="Policy Violations"
            value={String(stats.policy_violation_count)}
            accent={stats.policy_violation_count > 0 ? "red" : undefined}
          />
        </div>
      ) : (
        <p className="text-muted-foreground text-sm">No stats available for {envName}.</p>
      )}

      {stats && stats.top_failures.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
            Top Failure Modes
          </h2>
          <div className="rounded-lg border border-border overflow-hidden">
            {stats.top_failures.slice(0, 5).map((f, i) => {
              const maxCount = stats.top_failures[0].count;
              const pct = maxCount > 0 ? (f.count / maxCount) * 100 : 0;
              return (
                <div
                  key={f.check_name}
                  className={`flex items-center gap-4 px-4 py-3 text-sm ${i < stats.top_failures.length - 1 ? "border-b border-border" : ""}`}
                >
                  <span className="w-48 truncate font-mono text-xs text-muted-foreground">
                    {f.check_name}
                  </span>
                  <div className="flex-1 bg-muted rounded-full h-1.5 overflow-hidden">
                    <div
                      className="h-full bg-destructive rounded-full"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-6 text-right tabular-nums">
                    {f.count}
                  </span>
                </div>
              );
            })}
          </div>
        </section>
      )}

      <section className="space-y-3">
        <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
          Recent Episodes
        </h2>
        {episodes.length === 0 ? (
          <div className="rounded-lg border border-border px-4 py-8 text-center text-muted-foreground text-sm">
            No episodes recorded yet.
          </div>
        ) : (
          <div className="rounded-lg border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/30">
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">Episode</th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">Result</th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">Reward</th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">Steps</th>
                </tr>
              </thead>
              <tbody>
                {episodes.slice(0, 20).map((ep, i) => (
                  <tr
                    key={ep.id}
                    className={`hover:bg-muted/20 transition-colors ${i < episodes.length - 1 ? "border-b border-border/50" : ""}`}
                  >
                    <td className="px-4 py-2.5">
                      <Link
                        href={`/environments/${ep.env_name}/replay/${ep.id}`}
                        className="font-mono text-xs text-primary hover:underline"
                      >
                        {ep.id}
                      </Link>
                    </td>
                    <td className="px-4 py-2.5">
                      <span
                        className={`inline-flex items-center gap-1 text-xs font-medium ${
                          ep.passed ? "text-emerald-400" : "text-destructive"
                        }`}
                      >
                        {ep.passed ? "● pass" : "○ fail"}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 font-mono text-xs tabular-nums text-muted-foreground">
                      {ep.total_reward.toFixed(3)}
                    </td>
                    <td className="px-4 py-2.5 font-mono text-xs tabular-nums text-muted-foreground">
                      {ep.total_steps}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function StatCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: "green" | "red";
}) {
  return (
    <div className="rounded-lg border border-border bg-card px-4 py-4 space-y-1.5">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div
        className={`text-2xl font-semibold font-mono tabular-nums ${
          accent === "green"
            ? "text-emerald-400"
            : accent === "red"
            ? "text-destructive"
            : "text-foreground"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
