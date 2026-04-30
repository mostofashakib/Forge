"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { Toast } from "@/components/Toast";

const ENV_LIMIT = 10;

type EnvType = "general" | "cli" | "browser";

const ENV_TYPES: { id: EnvType; label: string; subtitle: string; description: string }[] = [
  {
    id: "browser",
    label: "Browser",
    subtitle: "Chromium + KasmVNC",
    description: "A full Chromium browser running in Docker, accessible via a web UI. Ideal for web automation and browser-based RL tasks.",
  },
  {
    id: "cli",
    label: "CLI",
    subtitle: "Ubuntu 22.04 terminal",
    description: "A Linux VM in Docker with a real bash terminal. Run scripts, install packages, and interact via the integrated terminal.",
  },
  {
    id: "general",
    label: "General Purpose",
    subtitle: "High fidelity app",
    description: "Describe a real-world application and our agents will generate a full-stack environment with observability, policy rules, and reward functions.",
  },
];

interface FormState {
  env_name: string;
  env_type: EnvType;
  description: string;
  domain: string;
  policy_requirements: string;
  reward_requirements: string;
  ttl_days: number;
}

export default function NewEnvironmentPage() {
  const router = useRouter();
  const [showToast, setShowToast] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [activeCount, setActiveCount] = useState<number | null>(null);
  const [form, setForm] = useState<FormState>({
    env_name: "",
    env_type: "general",
    description: "",
    domain: "",
    policy_requirements: "",
    reward_requirements: "",
    ttl_days: 30,
  });

  useEffect(() => {
    fetch(`${API_BASE}/api/sandbox/`, { cache: "no-store" })
      .then((r) => r.json())
      .then((list: unknown[]) => setActiveCount(list.length))
      .catch(() => {});
  }, []);

  const atLimit = activeCount !== null && activeCount >= ENV_LIMIT;

  function update<K extends keyof FormState>(field: K, value: FormState[K]) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  async function handleSubmit(e: { preventDefault(): void }) {
    e.preventDefault();
    setSubmitError(null);
    setSubmitting(true);
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setSubmitError(err.detail ?? `Request failed (${res.status})`);
        return;
      }
      const data = await res.json();
      setShowToast(true);
      setTimeout(() => router.push(`/environments/${data.env_name}/progress`), 1500);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Network error — is the backend running?");
    } finally {
      setSubmitting(false);
    }
  }

  const isGeneral = form.env_type === "general";

  return (
    <>
      {showToast && (
        <Toast message="Environment added to queue" onDismiss={() => setShowToast(false)} />
      )}
      <form onSubmit={handleSubmit} className="max-w-lg mx-auto p-8 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">New Environment</h1>
          {activeCount !== null && (
            <span className={`text-xs px-2 py-1 rounded-full font-medium ${
              atLimit ? "bg-red-100 text-red-700" : "bg-gray-100 text-gray-600"
            }`}>
              {activeCount} / {ENV_LIMIT} environments
            </span>
          )}
        </div>

        {atLimit && (
          <div className="border border-red-200 bg-red-50 rounded-lg p-4">
            <p className="text-sm font-semibold text-red-700">Environment limit reached</p>
            <p className="text-xs text-red-600 mt-1">
              You have {activeCount} active environments (max {ENV_LIMIT}).{" "}
              <Link href="/environments" className="underline font-medium">
                Delete an environment
              </Link>{" "}
              to create a new one.
            </p>
          </div>
        )}

        {/* Environment type selector */}
        <div>
          <label className="block text-sm font-medium mb-2">Environment Type</label>
          <div className="grid grid-cols-3 gap-3">
            {ENV_TYPES.map((t) => (
              <button
                key={t.id}
                type="button"
                disabled={atLimit}
                onClick={() => update("env_type", t.id)}
                className={`text-left p-3 rounded-lg border-2 transition-colors disabled:opacity-50 ${
                  form.env_type === t.id
                    ? "border-blue-500 bg-blue-50"
                    : "border-gray-200 hover:border-gray-300"
                }`}
              >
                <div className="font-medium text-sm">{t.label}</div>
                <div className="text-xs text-gray-500 mt-0.5">{t.subtitle}</div>
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {ENV_TYPES.find((t) => t.id === form.env_type)?.description}
          </p>
        </div>

        {/* Environment name */}
        <div>
          <label className="block text-sm font-medium mb-1">Environment Name</label>
          <input
            className="w-full border rounded px-3 py-2 text-sm disabled:opacity-50"
            placeholder="e.g. my_env"
            value={form.env_name}
            onChange={(e) => update("env_name", e.target.value)}
            disabled={atLimit}
            required
          />
        </div>

        {/* Description — General Purpose only */}
        {isGeneral && (
          <div>
            <label className="block text-sm font-medium mb-1">Description</label>
            <textarea
              className="w-full border rounded px-3 py-2 text-sm h-28 disabled:opacity-50"
              placeholder="Describe the real-world application you want to simulate..."
              value={form.description}
              onChange={(e) => update("description", e.target.value)}
              disabled={atLimit}
              required
            />
          </div>
        )}

        {/* Shared fields */}
        <div>
          <label className="block text-sm font-medium mb-1">
            Domain <span className="font-normal text-gray-400">(optional)</span>
          </label>
          <input
            className="w-full border rounded px-3 py-2 text-sm disabled:opacity-50"
            placeholder="e.g. support, email, crm — defaults to localhost"
            value={form.domain}
            onChange={(e) => update("domain", e.target.value)}
            disabled={atLimit}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">
            Policy Requirements <span className="font-normal text-gray-400">(optional)</span>
          </label>
          <textarea
            className="w-full border rounded px-3 py-2 text-sm h-20 disabled:opacity-50"
            placeholder="e.g. The agent cannot delete records."
            value={form.policy_requirements}
            onChange={(e) => update("policy_requirements", e.target.value)}
            disabled={atLimit}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">
            Reward Requirements <span className="font-normal text-gray-400">(optional)</span>
          </label>
          <textarea
            className="w-full border rounded px-3 py-2 text-sm h-20 disabled:opacity-50"
            placeholder="e.g. Reward speed. Penalize errors heavily."
            value={form.reward_requirements}
            onChange={(e) => update("reward_requirements", e.target.value)}
            disabled={atLimit}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">TTL (days)</label>
          <input
            type="number"
            min={1}
            max={365}
            className="w-full border rounded px-3 py-2 text-sm disabled:opacity-50"
            value={form.ttl_days}
            onChange={(e) => update("ttl_days", parseInt(e.target.value, 10))}
            disabled={atLimit}
          />
        </div>

        {submitError && (
          <div className="border border-red-200 bg-red-50 rounded-lg p-3">
            <p className="text-sm text-red-600">{submitError}</p>
            {submitError.includes("limit") && (
              <Link href="/environments" className="text-sm text-red-700 underline font-medium mt-1 block">
                Go to environments →
              </Link>
            )}
          </div>
        )}

        <button
          type="submit"
          disabled={submitting || atLimit}
          className="w-full py-2 bg-blue-600 text-white rounded hover:bg-blue-700 font-medium disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {submitting
            ? "Submitting…"
            : form.env_type === "cli"
            ? "Create CLI Environment"
            : form.env_type === "browser"
            ? "Create Browser Environment"
            : "Generate Environment"}
        </button>
      </form>
    </>
  );
}
