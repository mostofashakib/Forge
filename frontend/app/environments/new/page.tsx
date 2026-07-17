"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { useSandboxCapacity } from "@/lib/useSandboxCapacity";

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
  limit,
  onClose,
}: {
  def: QuickCreateDef;
  atLimit: boolean;
  activeCount: number | null;
  limit: number | null;
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
      className="new-env-modal-backdrop"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="new-env-modal">
        <div className="new-env-modal__header">
          <div className="flex items-center gap-2.5">
            <span>{def.icon}</span>
            <div><small>Quick runtime</small><h2>{def.label}</h2></div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="new-env-modal__close"
          >
            ✕
          </button>
        </div>

        <form onSubmit={handleCreate} className="new-env-modal__form">
          {atLimit && (
            <div className="border border-red-200 bg-red-50 rounded-lg p-3">
              <p className="text-xs font-semibold text-red-700">Environment limit reached</p>
              <p className="text-xs text-red-600 mt-0.5">
                {activeCount} / {limit} used.{" "}
                <Link href="/environments" className="underline font-medium">Delete one</Link> to continue.
              </p>
            </div>
          )}

          <div>
            <label className="new-env-modal__label">Environment name</label>
            <input
              className="new-env-modal__input"
              placeholder={def.placeholder}
              value={envName}
              onChange={(e) => setEnvName(e.target.value.replace(/\s+/g, "_"))}
              disabled={atLimit || submitting}
              required
              autoFocus
            />
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          <div className="new-env-modal__actions">
            <button
              type="submit"
              disabled={submitting || atLimit || !envName.trim()}
              className="new-env-modal__submit"
            >
              {submitting ? "Creating…" : "Create environment →"}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="new-env-modal__cancel"
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

const CARD_INDEX: Record<string, string> = {
  cli: "01",
  browser: "02",
  custom: "03",
  premade: "04",
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
  label,
  description,
  themeKey,
  onClick,
  href,
}: {
  label: string;
  description: string;
  themeKey: string;
  onClick?: () => void;
  href?: string;
}) {
  const inner = (
    <div className="new-env-card" data-theme={themeKey}>
      <div className="new-env-card__topline">
        <span>{CARD_INDEX[themeKey] ?? "00"} / BUILD PATH</span>
        <span>↗</span>
      </div>
      <div className="new-env-card__body">
        <div className="new-env-card__icon">{CARD_ICONS[themeKey]}</div>
        <div>
          <h2>{label}</h2>
          <p>{description}</p>
        </div>
      </div>
      <div className="new-env-card__action"><span>Select path</span><i /></div>
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
  const { activeCount, limit, error: capacityError } = useSandboxCapacity();
  const atLimit = activeCount !== null && limit !== null && activeCount >= limit;

  return (
    <>
      {modal && (
        <QuickCreateModal
          def={QUICK_CREATE[modal]}
          atLimit={atLimit}
          activeCount={activeCount}
          limit={limit}
          onClose={() => setModal(null)}
        />
      )}

      <div className="new-env-page">
        <header className="new-env-hero">
          <div className="new-env-hero__copy">
            <span>Build sequence / 01</span>
            <h1>CHOOSE YOUR<br /><em>STARTING POINT.</em></h1>
            <p>Launch a live runtime, describe a custom system, or deploy a proven template. Every path ends in a training-ready environment.</p>
          </div>
          <div className="new-env-hero__capacity">
            <span>Local capacity</span>
            {activeCount !== null && limit !== null ? (
              <>
                <strong>{String(activeCount).padStart(2, "0")}<small>/ {String(limit).padStart(2, "0")}</small></strong>
                <div><i style={{ width: `${Math.min((activeCount / limit) * 100, 100)}%` }} /></div>
                <p>{atLimit ? "Capacity reached" : `${limit - activeCount} build slots available`}</p>
              </>
            ) : (
              <><strong>--<small>/ --</small></strong><p>Reading local core</p></>
            )}
          </div>
        </header>

        {capacityError && (
          <div role="alert" className="border border-red-200 bg-red-50 rounded-lg p-4 text-sm text-red-700">
            Could not load environment capacity: {capacityError}
          </div>
        )}

        {atLimit && (
          <div className="border border-red-200 bg-red-50 rounded-lg p-4">
            <p className="text-sm font-semibold text-red-700">Environment limit reached</p>
            <p className="text-xs text-red-600 mt-1">
              You have {activeCount} active environments (max {limit}).{" "}
              <Link href="/environments" className="underline font-medium">Delete one</Link> to continue.
            </p>
          </div>
        )}

        <div className="new-env-heading">
          <div><span />Available build paths</div>
          <p>04 routes / select one</p>
        </div>

        <div className="new-env-grid">
          <OptionCard
            label="CLI Terminal"
            themeKey="cli"
            description="Full Ubuntu 22.04 terminal in Docker for shell scripting, package management, and system administration tasks."
            onClick={() => !atLimit && setModal("cli")}
          />
          <OptionCard
            label="Browser"
            themeKey="browser"
            description="Real Chromium browser in Docker for web automation, multi-step form filling, scraping, and navigation tasks."
            onClick={() => !atLimit && setModal("browser")}
          />
          <OptionCard
            label="Custom Environment"
            themeKey="custom"
            description="Describe any real-world application and Forge generates a complete RL environment with policy, reward, and observability."
            href="/environments/new/custom"
          />
          <OptionCard
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
