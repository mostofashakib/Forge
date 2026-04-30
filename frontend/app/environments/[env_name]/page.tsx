import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { DeleteEnvironmentButton } from "@/components/DeleteEnvironmentButton";

interface SandboxInfo {
  id: string;
  status: string;
  container_port: number | null;
  expires_at: string;
  ttl_days: number;
}

interface EnvStats {
  pass_rate: number;
  avg_reward: number;
  avg_steps: number;
  policy_violation_count: number;
}

const STATUS_COLORS: Record<string, string> = {
  running:  "bg-green-100 text-green-700",
  building: "bg-blue-100 text-blue-700",
  queued:   "bg-yellow-100 text-yellow-700",
  stopped:  "bg-gray-100 text-gray-500",
  error:    "bg-red-100 text-red-600",
};

const ACTIONS = [
  {
    id: "sandbox",
    label: "Sandbox",
    description: "Live app preview, interactive terminal, and observability feed",
    href: (env: string) => `/environments/${env}/sandbox`,
    requiresSandbox: true,
  },
  {
    id: "config",
    label: "Config",
    description: "View and edit the environment configuration YAML",
    href: (env: string) => `/environments/${env}/config`,
    requiresSandbox: false,
  },
  {
    id: "graph",
    label: "Entity Graph",
    description: "Visual graph of entities, actions, and state transitions",
    href: (env: string) => `/environments/${env}/graph`,
    requiresSandbox: false,
  },
  {
    id: "rollouts",
    label: "Rollouts",
    description: "Launch parallel episode rollouts and track agent performance",
    href: (_env: string) => `/rollouts`,
    requiresSandbox: false,
  },
  {
    id: "violations",
    label: "Violations",
    description: "Policy violation audit log and episode-level breakdowns",
    href: (env: string) => `/violations?env=${encodeURIComponent(env)}`,
    requiresSandbox: false,
  },
  {
    id: "dashboard",
    label: "Dashboard",
    description: "Pass rates, reward curves, and failure cluster analysis",
    href: (_env: string) => `/dashboard`,
    requiresSandbox: false,
  },
];

export default async function EnvironmentHubPage({
  params,
}: {
  params: Promise<{ env_name: string }>;
}) {
  const { env_name } = await params;

  const [sandbox, stats] = await Promise.all([
    fetch(`${API_BASE}/api/sandbox/${env_name}`, { cache: "no-store" })
      .then((r) => (r.ok ? (r.json() as Promise<SandboxInfo>) : null))
      .catch(() => null),
    fetch(`${API_BASE}/api/envs/${env_name}/stats`, { cache: "no-store" })
      .then((r) => (r.ok ? (r.json() as Promise<EnvStats>) : null))
      .catch(() => null),
  ]);

  const hasSandbox = sandbox !== null;
  const isLive = sandbox?.status === "running";
  const inProgress = sandbox?.status === "queued" || sandbox?.status === "building";

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-tight">{env_name}</h1>
            {sandbox && (
              <span
                className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  STATUS_COLORS[sandbox.status] ?? "bg-gray-100 text-gray-500"
                }`}
              >
                {sandbox.status}
              </span>
            )}
          </div>
          <p className="text-sm text-muted-foreground mt-1.5">
            {hasSandbox
              ? `Sandbox · TTL ${sandbox!.ttl_days}d · expires ${new Date(
                  sandbox!.expires_at
                ).toLocaleDateString()}`
              : "File-based environment"}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <DeleteEnvironmentButton envName={env_name} hasSandbox={hasSandbox} />
          <Link
            href="/environments"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            ← All environments
          </Link>
        </div>
      </div>

      {/* Build progress */}
      {inProgress && (
        <div className="border rounded-lg p-5 flex items-center justify-between">
          <p className="text-sm font-medium">
            {sandbox!.status === "queued" ? "Waiting for worker…" : "Generating environment — five agents running in parallel"}
          </p>
          <Link
            href={`/environments/${env_name}/progress`}
            className="text-sm text-primary hover:underline shrink-0 ml-4"
          >
            View progress →
          </Link>
        </div>
      )}

      {/* Stats row */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { label: "Pass Rate", value: `${Math.round(stats.pass_rate * 100)}%` },
            { label: "Avg Reward", value: stats.avg_reward.toFixed(2) },
            { label: "Avg Steps", value: stats.avg_steps.toFixed(1) },
            { label: "Violations", value: stats.policy_violation_count.toString() },
          ].map(({ label, value }) => (
            <div key={label} className="border rounded-lg p-4 text-center">
              <div className="text-2xl font-semibold">{value}</div>
              <div className="text-xs text-muted-foreground mt-1">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Action cards */}
      <div>
        <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-widest mb-3">
          Controls
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {ACTIONS.map((action) => {
            const disabled = action.requiresSandbox && !isLive;
            const href = action.href(env_name);

            const card = (
              <div
                className={`border rounded-lg p-4 h-full transition-colors ${
                  disabled
                    ? "opacity-50 cursor-not-allowed"
                    : "hover:border-primary/40 hover:bg-muted/30 cursor-pointer"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-sm">{action.label}</span>
                  {action.requiresSandbox && isLive && (
                    <span className="text-xs text-green-600 font-medium">● Live</span>
                  )}
                  {action.requiresSandbox && !hasSandbox && (
                    <span className="text-xs text-muted-foreground">No sandbox</span>
                  )}
                  {action.requiresSandbox && hasSandbox && !isLive && (
                    <span className="text-xs text-muted-foreground">{sandbox!.status}</span>
                  )}
                </div>
                <p className="text-xs text-muted-foreground">{action.description}</p>
              </div>
            );

            return disabled ? (
              <div key={action.id}>{card}</div>
            ) : (
              <Link key={action.id} href={href}>
                {card}
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}
