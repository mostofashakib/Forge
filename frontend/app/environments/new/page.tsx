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

function OptionCard({
  icon,
  label,
  description,
  onClick,
  href,
}: {
  icon: string;
  label: string;
  description: string;
  onClick?: () => void;
  href?: string;
}) {
  const inner = (
    <div className="h-full border-2 border-border rounded-2xl p-7 flex flex-col gap-4 hover:border-foreground/30 hover:bg-muted/10 transition-all group cursor-pointer">
      <span className="text-3xl font-mono leading-none">{icon}</span>
      <div>
        <div className="font-semibold text-base mb-1.5">{label}</div>
        <p className="text-sm text-muted-foreground leading-relaxed">{description}</p>
      </div>
      <span className="mt-auto text-xs text-muted-foreground group-hover:text-foreground transition-colors">
        Select →
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
            icon=">_"
            label="CLI"
            description="Full Ubuntu 22.04 terminal in Docker for shell scripting, package management, and system administration tasks."
            onClick={() => !atLimit && setModal("cli")}
          />
          <OptionCard
            icon="⬡"
            label="Browser"
            description="Real Chromium browser in Docker for web automation, multi-step form filling, scraping, and navigation tasks."
            onClick={() => !atLimit && setModal("browser")}
          />
          <OptionCard
            icon="◈"
            label="Custom Environment"
            description="Describe any real-world application and Forge generates a complete RL environment with policy, reward, and observability."
            href="/environments/new/custom"
          />
          <OptionCard
            icon="▤"
            label="Premade"
            description="Ready-to-use environments pre-configured with realistic apps, policies, and reward functions. Gmail and Slack available now."
            href="/environments/new/premade"
          />
        </div>
      </div>
    </>
  );
}
