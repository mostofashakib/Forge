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

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------

const STATUS_STYLES: Record<string, { badge: string; dot?: string }> = {
  running:  { badge: "bg-green-50 text-green-700 ring-1 ring-green-200",  dot: "bg-green-500" },
  building: { badge: "bg-blue-50 text-blue-700 ring-1 ring-blue-200",    dot: "bg-blue-500 animate-pulse" },
  queued:   { badge: "bg-amber-50 text-amber-700 ring-1 ring-amber-200", dot: "bg-amber-400 animate-pulse" },
  stopped:  { badge: "bg-slate-50 text-slate-500 ring-1 ring-slate-200", dot: "bg-slate-300" },
  error:    { badge: "bg-red-50 text-red-600 ring-1 ring-red-200",       dot: "bg-red-500" },
};

function StatusBadge({ status }: { status: string }) {
  const s = STATUS_STYLES[status] ?? STATUS_STYLES.stopped;
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium ${s.badge}`}>
      {s.dot && <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />}
      {status}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------

function AgentIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <polygon points="4,2.5 13.5,8 4,13.5" fill="currentColor" />
    </svg>
  );
}

function ChartIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <rect x="1.5" y="9.5" width="3.5" height="5" rx="0.5" fill="currentColor" />
      <rect x="6.25" y="6.5" width="3.5" height="8" rx="0.5" fill="currentColor" />
      <rect x="11" y="3.5" width="3.5" height="11" rx="0.5" fill="currentColor" />
    </svg>
  );
}

function ShieldIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 1.5L13.5 3.5V8C13.5 11.5 11 13.5 8 14.5C5 13.5 2.5 11.5 2.5 8V3.5L8 1.5Z" />
      <path d="M5.5 8.5L7 10l3.5-4" strokeWidth="1.3" />
    </svg>
  );
}

function StarIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 1.5l1.7 3.4 3.8.55-2.75 2.68.65 3.77L8 9.85l-3.4 1.99.65-3.77L2.5 5.45l3.8-.55L8 1.5Z" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="8" cy="8" r="6" />
      <path d="M5.5 8.5l2 2 3.5-4" strokeWidth="1.4" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 1.5L14.5 13.5H1.5L8 1.5Z" />
      <path d="M8 6.5V10" strokeWidth="1.3" />
      <circle cx="8" cy="11.5" r="0.5" fill="currentColor" stroke="none" />
    </svg>
  );
}

function SparkleIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path
        d="M8 1.5C8.3 3.8 9.7 5.2 12 5.5C9.7 5.8 8.3 7.2 8 9.5C7.7 7.2 6.3 5.8 4 5.5C6.3 5.2 7.7 3.8 8 1.5Z"
        fill="currentColor"
      />
      <path
        d="M12.5 10C12.7 11.1 13.3 11.8 14.5 12C13.3 12.2 12.7 12.9 12.5 14C12.3 12.9 11.7 12.2 10.5 12C11.7 11.8 12.3 11.1 12.5 10Z"
        fill="currentColor"
        opacity="0.7"
      />
    </svg>
  );
}

function ExportIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 2v8M5.5 7.5L8 10l2.5-2.5" />
      <path d="M2.5 12h11" />
      <path d="M2.5 12v1.5h11V12" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Section + action config
// ---------------------------------------------------------------------------

type ActionId = "agent" | "dashboard" | "policy" | "reward" | "evaluate" | "violations" | "synthetic" | "export";

const ACTION_ICONS: Record<ActionId, React.ReactNode> = {
  agent:      <AgentIcon />,
  dashboard:  <ChartIcon />,
  policy:     <ShieldIcon />,
  reward:     <StarIcon />,
  evaluate:   <CheckIcon />,
  violations: <AlertIcon />,
  synthetic:  <SparkleIcon />,
  export:     <ExportIcon />,
};

const ICON_THEME: Record<ActionId, { bg: string; color: string }> = {
  agent:      { bg: "bg-indigo-50",  color: "text-indigo-600" },
  dashboard:  { bg: "bg-indigo-50",  color: "text-indigo-600" },
  policy:     { bg: "bg-violet-50",  color: "text-violet-600" },
  reward:     { bg: "bg-violet-50",  color: "text-violet-600" },
  evaluate:   { bg: "bg-amber-50",   color: "text-amber-600"  },
  violations: { bg: "bg-amber-50",   color: "text-amber-600"  },
  synthetic:  { bg: "bg-emerald-50", color: "text-emerald-600" },
  export:     { bg: "bg-emerald-50", color: "text-emerald-600" },
};

interface Section {
  id: string;
  label: string;
  description: string;
  actions: ActionId[];
}

const SECTIONS: Section[] = [
  { id: "training",      label: "Training",      description: "Run, observe, and measure agent behavior.", actions: ["agent", "dashboard"] },
  { id: "configuration", label: "Configuration", description: "Shape the rules and definition of success.", actions: ["policy", "reward"] },
  { id: "analysis",      label: "Analysis",      description: "Interrogate trajectories and surface risk.", actions: ["evaluate", "violations"] },
  { id: "data",          label: "Data",          description: "Create and package learning-ready datasets.", actions: ["synthetic", "export"] },
];

interface ActionDef {
  id: ActionId;
  label: string;
  description: string;
  href: (env: string) => string;
  requiresSandbox: boolean;
  badge?: (cfg: { policyConfigured: boolean; rewardConfigured: boolean; isLive: boolean; hasSandbox: boolean; sandbox: SandboxInfo | null }) => string | null;
}

const ACTIONS: ActionDef[] = [
  {
    id: "agent",
    label: "Agent Runs",
    description: "Run agents inside the sandbox and record trajectories for policy training.",
    href: (e) => `/environments/${e}/agent`,
    requiresSandbox: true,
  },
  {
    id: "dashboard",
    label: "Dashboard",
    description: "Pass rate, reward distribution, and step efficiency across all agent runs.",
    href: (e) => `/environments/${e}/dashboard`,
    requiresSandbox: false,
  },
  {
    id: "policy",
    label: "Policy",
    description: "Define rules that constrain agent behaviour — what the agent must not do.",
    href: (e) => `/environments/${e}/policy`,
    requiresSandbox: false,
    badge: ({ policyConfigured }) => policyConfigured ? "Custom" : "Default",
  },
  {
    id: "reward",
    label: "Reward",
    description: "Define how success is measured — what a good trajectory looks like.",
    href: (e) => `/environments/${e}/reward`,
    requiresSandbox: false,
    badge: ({ rewardConfigured }) => rewardConfigured ? "Custom" : "Default",
  },
  {
    id: "evaluate",
    label: "Evaluate",
    description: "Re-run policy and reward checks against recent agent trajectories.",
    href: (e) => `/environments/${e}/evaluate`,
    requiresSandbox: false,
  },
  {
    id: "violations",
    label: "Violations",
    description: "Policy audit log — view rule violations and detect reward hacking.",
    href: (e) => `/environments/${e}/violations`,
    requiresSandbox: false,
  },
  {
    id: "synthetic",
    label: "Synthetic Data",
    description: "Generate synthetic episodes from a research goal for agent replay.",
    href: (e) => `/environments/${e}/synthetic`,
    requiresSandbox: false,
  },
  {
    id: "export",
    label: "Export Dataset",
    description: "Package trajectories as SFT pairs, DPO preferences, RL rollouts, and more.",
    href: (e) => `/environments/${e}/export`,
    requiresSandbox: false,
  },
];

const ACTION_MAP = Object.fromEntries(ACTIONS.map((a) => [a.id, a])) as Record<ActionId, ActionDef>;

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

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

  const badgeCtx = { policyConfigured, rewardConfigured, isLive, hasSandbox, sandbox };
  const expiryDate = sandbox
    ? new Date(sandbox.expires_at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
    : null;

  return (
    <div className="environment-hub">
      {/* ------------------------------------------------------------------ */}
      {/* Header */}
      {/* ------------------------------------------------------------------ */}
      <section className="environment-hub__header">
        <div className="environment-hub__topline">
          <span>Environment workspace</span>
          <Link href="/environments" className="environment-hub__back">← All environments</Link>
        </div>

        <div className="environment-hub__identity">
          <div className="environment-hub__mark">
            <svg width="24" height="24" viewBox="0 0 14 14" fill="none">
              <path d="M7 1L13 4V10L7 13L1 10V4L7 1Z" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" className="text-primary" />
              <path d="M7 5L9 6.5V9L7 10.5L5 9V6.5L7 5Z" fill="currentColor" className="text-primary" />
            </svg>
          </div>
          <div className="environment-hub__name">
            <div className="flex items-center gap-3 flex-wrap">
              <h1>{env_name}</h1>
              {sandbox && <StatusBadge status={sandbox.status} />}
            </div>
            <p>{hasSandbox ? "Containerized agent training environment" : "File-based training environment"}</p>
          </div>

          <div className="environment-hub__controls">
          <SandboxControls
            envName={env_name}
            status={sandbox?.status ?? ""}
            hasSandbox={hasSandbox}
          />
          </div>
        </div>

        <div className="environment-hub__meta">
          <div><span>Runtime</span><strong>{hasSandbox ? "Sandbox" : "File"}</strong></div>
          <div><span>Retention</span><strong>{sandbox ? `${sandbox.ttl_days} days` : "Persistent"}</strong></div>
          <div><span>Expires</span><strong>{expiryDate ?? "Never"}</strong></div>
          <div><span>Core state</span><strong className={isLive ? "text-accent" : "text-background"}>{sandbox?.status ?? "local"}</strong></div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Status banners */}
      {/* ------------------------------------------------------------------ */}
      {isLive && (
        <div className="border border-green-200 bg-green-50/60 rounded-xl p-4 flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold text-green-800">Environment is ready</p>
            <p className="text-xs text-green-700 mt-0.5">
              App running on port {sandbox!.container_port} · available via Sandbox
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

      {inProgress && (
        <div className="border border-border/60 rounded-xl p-4 flex items-center justify-between bg-muted/20">
          <div className="flex items-center gap-3">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse shrink-0" />
            <p className="text-sm font-medium">
              {sandbox!.status === "queued" ? "Waiting for worker…" : "Generating environment — agents running in parallel"}
            </p>
          </div>
          <Link
            href={`/environments/${env_name}/progress`}
            className="text-sm text-primary hover:underline shrink-0 ml-4 font-medium"
          >
            View progress →
          </Link>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* All action cards in one flat grid — section labels as col-span-2 separators */}
      {/* ------------------------------------------------------------------ */}
      <div className="environment-hub__sections">
        {SECTIONS.map((section, sectionIndex) => (
          <section key={section.id} className="hub-section">
            <header className="hub-section__header">
              <span className="hub-section__number">0{sectionIndex + 1}</span>
              <div>
                <h2>{section.label}</h2>
                <p>{section.description}</p>
              </div>
            </header>

            <div className="hub-section__grid">
            {section.actions.map((actionId) => {
            const action = ACTION_MAP[actionId];
            const theme = ICON_THEME[actionId];
            const disabled = action.requiresSandbox && !isLive;
            const badge = action.badge?.(badgeCtx);

            const card = (
              <div
                data-action={actionId}
                className={`hub-action-card group ${disabled ? "hub-action-card--disabled" : ""}`}
              >
                <div className="hub-action-card__content">
                  {/* Icon well */}
                  <div className={`hub-action-card__icon ${theme.bg} ${theme.color}`}>
                    {ACTION_ICONS[actionId]}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="hub-action-card__title-row">
                      <h3>{action.label}</h3>
                      <span className="hub-action-card__badge">
                        {action.requiresSandbox && isLive && (
                          <span className="text-green-600 font-medium flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-green-500 inline-block" />
                            Live
                          </span>
                        )}
                        {action.requiresSandbox && !isLive && hasSandbox && (
                          <span className="text-muted-foreground">{sandbox!.status}</span>
                        )}
                        {action.requiresSandbox && !hasSandbox && (
                          <span className="text-muted-foreground">No sandbox</span>
                        )}
                        {badge && (
                          <span className={badge === "Custom" ? "text-emerald-600 font-medium" : "text-muted-foreground"}>
                            {badge === "Custom" ? "● Custom" : "Default"}
                          </span>
                        )}
                      </span>
                    </div>
                    <p>{action.description}</p>
                  </div>
                </div>

                {/* Hover arrow */}
                {!disabled && (
                  <span className="hub-action-card__arrow">↗</span>
                )}
              </div>
            );

            return disabled ? (
              <div key={actionId} className="h-full">{card}</div>
            ) : (
              <Link key={actionId} href={action.href(env_name)} className="h-full block">
                {card}
              </Link>
            );
            })}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
