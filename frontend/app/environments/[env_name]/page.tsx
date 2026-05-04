import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { SandboxControls } from "@/components/SandboxControls";

interface SandboxInfo {
  id: string;
  status: string;
  container_port: number | null;
  expires_at: string;
  ttl_days: number;
}

interface EvalConfig {
  policy_requirements: string;
  reward_requirements: string;
}

const STATUS_COLORS: Record<string, string> = {
  running:  "bg-green-100 text-green-700",
  building: "bg-blue-100 text-blue-700",
  queued:   "bg-yellow-100 text-yellow-700",
  stopped:  "bg-gray-100 text-gray-500",
  error:    "bg-red-100 text-red-600",
};

export default async function EnvironmentHubPage({
  params,
}: {
  params: Promise<{ env_name: string }>;
}) {
  const { env_name } = await params;

  const [sandbox, evalConfig] = await Promise.all([
    fetch(`${API_BASE}/api/sandbox/${env_name}`, { cache: "no-store" })
      .then((r) => (r.ok ? (r.json() as Promise<SandboxInfo>) : null))
      .catch(() => null),
    fetch(`${API_BASE}/api/sandbox/${env_name}/evaluate`, { cache: "no-store" })
      .then((r) => (r.ok ? (r.json() as Promise<EvalConfig>) : null))
      .catch(() => null),
  ]);

  const hasSandbox = sandbox !== null;
  const isLive = sandbox?.status === "running";
  const inProgress = sandbox?.status === "queued" || sandbox?.status === "building";

  const policyConfigured = !!(evalConfig?.policy_requirements?.trim());
  const rewardConfigured = !!(evalConfig?.reward_requirements?.trim());

  const ACTIONS = [
    {
      id: "agent",
      label: "Agent Runs",
      description: "Run agents inside the sandbox, record trajectories, and collect data for policy training",
      href: `/environments/${env_name}/agent`,
      requiresSandbox: true,
      badge: null as string | null,
    },
    {
      id: "dashboard",
      label: "Dashboard",
      description: "Aggregated metrics across all agent runs — pass rate, reward distribution, and step efficiency",
      href: `/environments/${env_name}/dashboard`,
      requiresSandbox: false,
      badge: null,
    },
    {
      id: "policy",
      label: "Policy",
      description: "Define rules that constrain agent behaviour — what the agent must not do",
      href: `/environments/${env_name}/policy`,
      requiresSandbox: false,
      badge: policyConfigured ? "Custom" : "Default",
    },
    {
      id: "reward",
      label: "Reward",
      description: "Define how success is measured — what a good trajectory looks like",
      href: `/environments/${env_name}/reward`,
      requiresSandbox: false,
      badge: rewardConfigured ? "Custom" : "Default",
    },
    {
      id: "evaluate",
      label: "Evaluate",
      description: "Re-run policy and reward evaluation against recent agent trajectories",
      href: `/environments/${env_name}/evaluate`,
      requiresSandbox: false,
      badge: null,
    },
    {
      id: "synthetic",
      label: "Synthetic Data",
      description: "Generate a synthetic epoch — a bundle of episodes each with a trajectory — from a research goal",
      href: `/environments/${env_name}/synthetic`,
      requiresSandbox: false,
      badge: null,
    },
    {
      id: "violations",
      label: "Violations",
      description: "Policy audit log — view rule violations, run trajectory analysis, and detect reward hacking",
      href: `/environments/${env_name}/violations`,
      requiresSandbox: false,
      badge: null,
    },
    {
      id: "export",
      label: "Export Dataset",
      description: "Export trajectories as SFT pairs, DPO preferences, RL rollouts, and failure datasets for training.",
      href: `/environments/${env_name}/export`,
      requiresSandbox: false,
      badge: null,
    },
  ];

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
          <SandboxControls
            envName={env_name}
            status={sandbox?.status ?? ""}
            hasSandbox={hasSandbox}
          />
          <Link
            href="/environments"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            ← All environments
          </Link>
        </div>
      </div>

      {/* Ready banner */}
      {isLive && (
        <div className="border border-green-200 bg-green-50 rounded-lg p-5 flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold text-green-800">Environment is ready</p>
            <p className="text-xs text-green-700 mt-0.5">
              Your app is running on port {sandbox!.container_port} and available via the Sandbox below.
            </p>
          </div>
          <Link
            href={`/environments/${env_name}/sandbox`}
            className="shrink-0 ml-4 px-4 py-2 bg-green-700 text-white text-sm font-medium rounded-lg hover:bg-green-800 transition-colors"
          >
            Open Sandbox →
          </Link>
        </div>
      )}

      {/* Build progress */}
      {inProgress && (
        <div className="border rounded-lg p-5 flex items-center justify-between">
          <p className="text-sm font-medium">
            {sandbox!.status === "queued"
              ? "Waiting for worker…"
              : "Generating environment — agents running in parallel"}
          </p>
          <Link
            href={`/environments/${env_name}/progress`}
            className="text-sm text-primary hover:underline shrink-0 ml-4"
          >
            View progress →
          </Link>
        </div>
      )}

      {/* Controls grid */}
      <div>
        <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-widest mb-3">
          Controls
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {ACTIONS.map((action) => {
            const disabled = action.requiresSandbox && !isLive;

            const card = (
              <div
                className={`border rounded-lg p-4 transition-colors ${
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
                  {action.badge && (
                    <span className={`text-xs font-medium ${action.badge === "Custom" ? "text-green-600" : "text-muted-foreground"}`}>
                      {action.badge === "Custom" ? "● Custom" : "Default"}
                    </span>
                  )}
                </div>
                <p className="text-xs text-muted-foreground">{action.description}</p>
              </div>
            );

            return disabled ? (
              <div key={action.id}>{card}</div>
            ) : (
              <Link key={action.id} href={action.href}>
                {card}
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}
