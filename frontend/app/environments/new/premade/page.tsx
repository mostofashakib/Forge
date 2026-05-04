"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

const ENV_LIMIT = 10;

// ---------------------------------------------------------------------------
// Premade templates
// ---------------------------------------------------------------------------

interface Template {
  id: string;
  label: string;
  subtitle: string;
  icon: string;
  envType: string;
  badgeColor: string;
  blurb: string;
  details: string;
  policyRequirements: string;
  rewardRequirements: string;
}

const TEMPLATES: Template[] = [
  {
    id: "gmail",
    label: "Gmail",
    subtitle: "Email client · General",
    icon: "✉",
    envType: "premade:gmail",
    badgeColor: "bg-red-100 text-red-700",
    blurb: "Full-featured email client for inbox triage and communication workflows.",
    details:
      "Agents manage a realistic inbox — archiving newsletters, starring urgent messages, composing replies, and organising with labels. Tasks range from simple inbox-zero objectives to multi-step label-and-reply sequences. Reward is highest for correct inbox management without accidental deletions, with bonuses for efficiency and correct label usage. Agents are penalised for permanently deleting emails, replying to the wrong sender, or marking important messages as spam.",
    policyRequirements:
      "The agent must not permanently delete emails — only move to trash.\nThe agent must not send emails to unintended recipients.\nReplies must include the original subject line.",
    rewardRequirements:
      "Reward completing inbox-zero objectives efficiently.\nPenalise destructive actions (permanent deletion) heavily.\nBonus reward for using labels correctly.",
  },
  {
    id: "slack",
    label: "Slack",
    subtitle: "Team messaging · General",
    icon: "#",
    envType: "premade:slack",
    badgeColor: "bg-purple-100 text-purple-700",
    blurb: "Team messaging workspace for channel management and coordination tasks.",
    details:
      "Agents operate inside a workspace with channels, threads, reactions, pins, and DMs. Typical tasks include pinning announcements, sending status updates, archiving quiet channels, and replying to threads. Reward is highest for correct channel management without data loss, with bonuses for clear and concise communication. Agents are penalised for sending duplicate messages, archiving active channels, or deleting pinned announcements.",
    policyRequirements:
      "The agent must not delete messages in #announcements.\nThe agent must not archive channels with unread messages unless explicitly instructed.\nStatus updates must be sent to #general before performing other tasks.",
    rewardRequirements:
      "Reward efficient channel organisation and message delivery.\nPenalise sending duplicate messages heavily.\nBonus for correctly using reactions to acknowledge messages.",
  },
];

// ---------------------------------------------------------------------------
// Template card
// ---------------------------------------------------------------------------

function TemplateCard({
  template,
  onSelect,
}: {
  template: Template;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className="text-left w-full border-2 border-border rounded-xl p-5 hover:border-foreground/30 hover:bg-muted/10 transition-all group"
    >
      <div className="flex items-start justify-between gap-2 mb-3">
        <div className="flex items-center gap-3">
          <span className="text-2xl font-mono leading-none">{template.icon}</span>
          <div>
            <div className="font-semibold text-sm">{template.label}</div>
            <div className="text-xs text-muted-foreground">{template.subtitle}</div>
          </div>
        </div>
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium shrink-0 ${template.badgeColor}`}>
          Select →
        </span>
      </div>
      <p className="text-xs text-muted-foreground leading-relaxed">{template.blurb}</p>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Create modal
// ---------------------------------------------------------------------------

function CreateModal({
  template,
  atLimit,
  activeCount,
  onClose,
}: {
  template: Template;
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
          env_type: template.envType,
          description: `Premade environment: ${template.label}`,
          policy_requirements: template.policyRequirements,
          reward_requirements: template.rewardRequirements,
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
      <div className="bg-background border border-border rounded-2xl shadow-xl w-full max-w-md">
        <div className="flex items-center justify-between px-6 pt-6 pb-4 border-b border-border">
          <div className="flex items-center gap-2.5">
            <span className="font-mono text-lg">{template.icon}</span>
            <div>
              <h2 className="font-semibold text-sm">{template.label}</h2>
              <p className="text-xs text-muted-foreground">{template.subtitle}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground text-lg leading-none"
          >
            ✕
          </button>
        </div>

        <div className="px-6 py-5 space-y-5">
          <p className="text-sm text-muted-foreground leading-relaxed">{template.details}</p>

          {atLimit && (
            <div className="border border-red-200 bg-red-50 rounded-lg p-3">
              <p className="text-xs font-semibold text-red-700">Environment limit reached</p>
              <p className="text-xs text-red-600 mt-0.5">
                You have {activeCount} / {ENV_LIMIT} environments.{" "}
                <Link href="/environments" className="underline font-medium">Delete one</Link> to continue.
              </p>
            </div>
          )}

          <form onSubmit={handleCreate} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1.5">Environment name</label>
              <input
                className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-foreground/20 disabled:opacity-50"
                placeholder={`e.g. ${template.id}_training_env`}
                value={envName}
                onChange={(e) => setEnvName(e.target.value.replace(/\s+/g, "_"))}
                disabled={atLimit || submitting}
                required
                autoFocus
              />
              <p className="text-xs text-muted-foreground mt-1">
                Policy and reward requirements are pre-configured from the template.
              </p>
            </div>

            {error && <p className="text-sm text-red-600">{error}</p>}

            <div className="flex items-center gap-3 pt-1">
              <button
                type="submit"
                disabled={submitting || atLimit || !envName.trim()}
                className="px-5 py-2 text-sm font-medium text-white bg-foreground rounded-lg hover:opacity-90 disabled:opacity-50 transition-opacity"
              >
                {submitting ? "Creating…" : `Create ${template.label} →`}
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
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function PremadeEnvironmentsPage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeCount, setActiveCount] = useState<number | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/sandbox/`, { cache: "no-store" })
      .then((r) => r.json())
      .then((list: unknown[]) => setActiveCount(list.length))
      .catch(() => {});
  }, []);

  const atLimit = activeCount !== null && activeCount >= ENV_LIMIT;
  const selected = TEMPLATES.find((t) => t.id === selectedId) ?? null;

  return (
    <>
      {selected && (
        <CreateModal
          template={selected}
          atLimit={atLimit}
          activeCount={activeCount}
          onClose={() => setSelectedId(null)}
        />
      )}

      <div className="max-w-2xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Premade Environments</h1>
            <p className="text-sm text-muted-foreground mt-1.5">
              Pre-configured environments with realistic apps, policies, and reward functions.
            </p>
          </div>
          <Link
            href="/environments/new"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors shrink-0"
          >
            ← Back
          </Link>
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

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {TEMPLATES.map((t) => (
            <TemplateCard
              key={t.id}
              template={t}
              onSelect={() => !atLimit && setSelectedId(t.id)}
            />
          ))}
        </div>
      </div>
    </>
  );
}
