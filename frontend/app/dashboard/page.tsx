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
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <span className="text-sm text-muted-foreground font-mono">{envName}</span>
      </div>

      {/* Stat cards */}
      {stats ? (
        <div className="grid grid-cols-4 gap-4">
          <StatCard
            label="Pass Rate"
            value={`${(stats.pass_rate * 100).toFixed(1)}%`}
            highlight={stats.pass_rate >= 0.7 ? "green" : "red"}
          />
          <StatCard label="Avg Reward" value={stats.avg_reward.toFixed(3)} />
          <StatCard label="Avg Steps" value={stats.avg_steps.toFixed(1)} />
          <StatCard
            label="Policy Violations"
            value={String(stats.policy_violation_count)}
            highlight={stats.policy_violation_count > 0 ? "red" : undefined}
          />
        </div>
      ) : (
        <p className="text-muted-foreground text-sm">No stats available for {envName}.</p>
      )}

      {/* Top Failure Modes */}
      {stats && stats.top_failures.length > 0 && (
        <section>
          <h2 className="text-sm font-semibold text-muted-foreground uppercase mb-3">Top Failure Modes</h2>
          <div className="space-y-2">
            {stats.top_failures.slice(0, 5).map((f) => {
              const maxCount = stats.top_failures[0].count;
              const pct = maxCount > 0 ? (f.count / maxCount) * 100 : 0;
              return (
                <div key={f.check_name} className="flex items-center gap-3 text-sm">
                  <span className="w-48 truncate font-mono text-xs text-foreground">{f.check_name}</span>
                  <div className="flex-1 bg-muted rounded h-3 overflow-hidden">
                    <div
                      className="h-full bg-red-500"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-8 text-right">{f.count}</span>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Recent Episodes */}
      <section>
        <h2 className="text-sm font-semibold text-muted-foreground uppercase mb-3">Recent Episodes</h2>
        {episodes.length === 0 ? (
          <p className="text-muted-foreground text-sm">No episodes yet.</p>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-muted-foreground border-b">
                <th className="pb-2 font-medium">Episode</th>
                <th className="pb-2 font-medium">Status</th>
                <th className="pb-2 font-medium">Reward</th>
                <th className="pb-2 font-medium">Steps</th>
              </tr>
            </thead>
            <tbody>
              {episodes.slice(0, 20).map((ep) => (
                <tr key={ep.id} className="border-b border-muted/30 hover:bg-muted/20">
                  <td className="py-1.5">
                    <Link
                      href={`/environments/${ep.env_name}/replay/${ep.id}`}
                      className="font-mono text-xs text-blue-400 hover:underline"
                    >
                      {ep.id}
                    </Link>
                  </td>
                  <td className="py-1.5">
                    <span className={`text-xs ${ep.passed ? "text-green-400" : "text-red-400"}`}>
                      {ep.passed ? "✓ pass" : "✗ fail"}
                    </span>
                  </td>
                  <td className="py-1.5 font-mono text-xs">{ep.total_reward.toFixed(3)}</td>
                  <td className="py-1.5 font-mono text-xs">{ep.total_steps}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}

function StatCard({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: "green" | "red";
}) {
  return (
    <div className="bg-muted rounded-lg p-4 space-y-1">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div
        className={`text-2xl font-bold font-mono ${
          highlight === "green"
            ? "text-green-400"
            : highlight === "red"
            ? "text-red-400"
            : "text-foreground"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
