"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

const ENV_LIMIT = 10;

// ---------------------------------------------------------------------------
// Quick-create modal (Browser / CLI)
// ---------------------------------------------------------------------------

interface QuickCreateDef {
  label: string;
  envType: string;
  icon: string;
  placeholder: string;
}

function QuickCreateModal({
  def,
  atLimit,
  activeCount,
  onClose,
}: {
  def: QuickCreateDef;
  atLimit: boolean;
  activeCount: number | null;
  onClose: () => void;
}) {
  const router = useRouter();
  const [envName, setEnvName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!envName.trim()) return;
    setSubmitting(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          env_name: envName.trim(),
          env_type: def.envType,
          ttl_days: 30,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const detail = body.detail;
        setError(
          Array.isArray(detail)
            ? detail.map((e: { msg?: string }) => e.msg ?? String(e)).join("; ")
            : (detail ?? `Request failed (${res.status})`)
        );
        return;
      }
      const data = await res.json();
      router.push(`/environments/${data.env_name}/progress`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error — is the backend running?");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-background border border-border rounded-2xl shadow-xl w-full max-w-sm">
        <div className="flex items-center justify-between px-6 pt-6 pb-4 border-b border-border">
          <div className="flex items-center gap-2.5">
            <span className="font-mono text-lg">{def.icon}</span>
            <h2 className="font-semibold text-sm">{def.label}</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground text-lg leading-none"
          >
            ✕
          </button>
        </div>

        <form onSubmit={handleCreate} className="px-6 py-5 space-y-4">
          {atLimit && (
            <div className="border border-red-200 bg-red-50 rounded-lg p-3">
              <p className="text-xs font-semibold text-red-700">Environment limit reached</p>
              <p className="text-xs text-red-600 mt-0.5">
                {activeCount} / {ENV_LIMIT} used.{" "}
                <Link href="/environments" className="underline font-medium">Delete one</Link> to continue.
              </p>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium mb-1.5">Environment name</label>
            <input
              className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-foreground/20 disabled:opacity-50"
              placeholder={def.placeholder}
              value={envName}
              onChange={(e) => setEnvName(e.target.value.replace(/\s+/g, "_"))}
              disabled={atLimit || submitting}
              required
              autoFocus
            />
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          <div className="flex items-center gap-3 pt-1">
            <button
              type="submit"
              disabled={submitting || atLimit || !envName.trim()}
              className="px-5 py-2 text-sm font-medium text-white bg-foreground rounded-lg hover:opacity-90 disabled:opacity-50 transition-opacity"
            >
              {submitting ? "Creating…" : "Create environment →"}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="text-sm text-muted-foreground hover:text-foreground"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Option card
// ---------------------------------------------------------------------------

const CARD_THEMES: Record<string, { iconBg: string; iconText: string; accent: string }> = {
  cli:     { iconBg: "bg-emerald-50",  iconText: "text-emerald-600",  accent: "group-hover:border-emerald-300/60" },
  browser: { iconBg: "bg-blue-50",     iconText: "text-blue-600",     accent: "group-hover:border-blue-300/60" },
  custom:  { iconBg: "bg-violet-50",   iconText: "text-violet-600",   accent: "group-hover:border-violet-300/60" },
  premade: { iconBg: "bg-orange-50",   iconText: "text-orange-500",   accent: "group-hover:border-orange-300/60" },
};

const CARD_ICONS: Record<string, React.ReactNode> = {
  cli: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="1.5" y="2.5" width="15" height="13" rx="2" />
      <path d="M5 7l3 2.5L5 12" />
      <path d="M10 12h3" />
    </svg>
  ),
  browser: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="9" cy="9" r="7" />
      <path d="M2 9h14" />
      <path d="M9 2c-2 2-3 4-3 7s1 5 3 7" />
      <path d="M9 2c2 2 3 4 3 7s-1 5-3 7" />
    </svg>
  ),
  custom: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 1.5C9.4 4.8 10.8 6.2 14 6.5C10.8 6.8 9.4 8.2 9 11.5C8.6 8.2 7.2 6.8 4 6.5C7.2 6.2 8.6 4.8 9 1.5Z" fill="currentColor" stroke="none" />
      <path d="M13.5 12C13.7 13.1 14.3 13.7 15.5 14C14.3 14.3 13.7 14.9 13.5 16C13.3 14.9 12.7 14.3 11.5 14C12.7 13.7 13.3 13.1 13.5 12Z" fill="currentColor" stroke="none" opacity="0.7" />
    </svg>
  ),
  premade: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="2" width="6" height="6" rx="1.5" fill="currentColor" opacity="0.2" />
      <rect x="10" y="2" width="6" height="6" rx="1.5" fill="currentColor" />
      <rect x="2" y="10" width="6" height="6" rx="1.5" fill="currentColor" />
      <rect x="10" y="10" width="6" height="6" rx="1.5" fill="currentColor" opacity="0.2" />
    </svg>
  ),
};

function OptionCard({
  icon,
  label,
  description,
  themeKey,
  onClick,
  href,
}: {
  icon: string;
  label: string;
  description: string;
  themeKey: string;
  onClick?: () => void;
  href?: string;
}) {
  const theme = CARD_THEMES[themeKey] ?? CARD_THEMES.custom;

  const inner = (
    <div className={`group h-full card-shadow hover:card-shadow-hover hover:-translate-y-0.5 bg-card border border-border/60 rounded-2xl p-6 flex flex-col gap-4 transition-all duration-200 cursor-pointer ${theme.accent}`}>
      <div className={`w-10 h-10 rounded-xl ${theme.iconBg} ${theme.iconText} flex items-center justify-center shrink-0`}>
        {CARD_ICONS[themeKey]}
      </div>
      <div className="flex-1">
        <div className="font-semibold text-sm mb-1.5">{label}</div>
        <p className="text-xs text-muted-foreground leading-relaxed">{description}</p>
      </div>
      <span className="text-xs text-muted-foreground/60 group-hover:text-muted-foreground flex items-center gap-1 transition-colors">
        Select
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M2 5h6M5.5 2.5L8 5l-2.5 2.5" />
        </svg>
      </span>
    </div>
  );

  if (href) {
    return <Link href={href} className="block h-full">{inner}</Link>;
  }
  return (
    <button type="button" onClick={onClick} className="text-left w-full h-full">
      {inner}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

const QUICK_CREATE: Record<string, QuickCreateDef> = {
  cli: {
    label: "Terminal / CLI",
    envType: "cli",
    icon: ">_",
    placeholder: "e.g. cli_training_env",
  },
  browser: {
    label: "Browser",
    envType: "browser",
    icon: "⬡",
    placeholder: "e.g. browser_training_env",
  },
};

export default function NewEnvironmentPage() {
  const [modal, setModal] = useState<"cli" | "browser" | null>(null);
  const [activeCount, setActiveCount] = useState<number | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/sandbox/`, { cache: "no-store" })
      .then((r) => r.json())
      .then((list: unknown[]) => setActiveCount(list.length))
      .catch(() => {});
  }, []);

  const atLimit = activeCount !== null && activeCount >= ENV_LIMIT;

  return (
    <>
      {modal && (
        <QuickCreateModal
          def={QUICK_CREATE[modal]}
          atLimit={atLimit}
          activeCount={activeCount}
          onClose={() => setModal(null)}
        />
      )}

      <div className="max-w-3xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">New Environment</h1>
            <p className="text-sm text-muted-foreground mt-1.5">
              Choose an environment type to get started.
            </p>
          </div>
          {activeCount !== null && (
            <span className={`text-xs px-2.5 py-1 rounded-full font-medium shrink-0 ${
              atLimit ? "bg-red-100 text-red-700" : "bg-muted text-muted-foreground"
            }`}>
              {activeCount} / {ENV_LIMIT}
            </span>
          )}
        </div>

        {atLimit && (
          <div className="border border-red-200 bg-red-50 rounded-lg p-4">
            <p className="text-sm font-semibold text-red-700">Environment limit reached</p>
            <p className="text-xs text-red-600 mt-1">
              You have {activeCount} active environments (max {ENV_LIMIT}).{" "}
              <Link href="/environments" className="underline font-medium">Delete one</Link> to continue.
            </p>
          </div>
        )}

        {/* 2×2 grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <OptionCard
            icon="cli"
            label="CLI Terminal"
            themeKey="cli"
            description="Full Ubuntu 22.04 terminal in Docker for shell scripting, package management, and system administration tasks."
            onClick={() => !atLimit && setModal("cli")}
          />
          <OptionCard
            icon="browser"
            label="Browser"
            themeKey="browser"
            description="Real Chromium browser in Docker for web automation, multi-step form filling, scraping, and navigation tasks."
            onClick={() => !atLimit && setModal("browser")}
          />
          <OptionCard
            icon="custom"
            label="Custom Environment"
            themeKey="custom"
            description="Describe any real-world application and Forge generates a complete RL environment with policy, reward, and observability."
            href="/environments/new/custom"
          />
          <OptionCard
            icon="premade"
            label="Premade"
            themeKey="premade"
            description="Ready-to-use environments pre-configured with realistic apps, policies, and reward functions. Gmail and Slack available now."
            href="/environments/new/premade"
          />
        </div>
      </div>
    </>
  );
}
