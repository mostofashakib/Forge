"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { API_BASE } from "@/lib/api";
import { Toast } from "@/components/Toast";

interface FormState {
  env_name: string;
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
  const [form, setForm] = useState<FormState>({
    env_name: "",
    description: "",
    domain: "",
    policy_requirements: "",
    reward_requirements: "",
    ttl_days: 30,
  });

  function update(field: keyof FormState, value: string | number) {
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

  return (
    <>
    {showToast && (
      <Toast message="Environment added to queue" onDismiss={() => setShowToast(false)} />
    )}
    <form onSubmit={handleSubmit} className="max-w-lg mx-auto p-8 space-y-5">
      <h1 className="text-2xl font-bold">New Sandbox Environment</h1>

      <div>
        <label className="block text-sm font-medium mb-1">Environment Name</label>
        <input
          className="w-full border rounded px-3 py-2 text-sm"
          placeholder="e.g. ticket_system"
          value={form.env_name}
          onChange={(e) => update("env_name", e.target.value)}
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">Description</label>
        <textarea
          className="w-full border rounded px-3 py-2 text-sm h-28"
          placeholder="Describe the real-world application you want to simulate..."
          value={form.description}
          onChange={(e) => update("description", e.target.value)}
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">
          Domain <span className="font-normal text-gray-400">(optional)</span>
        </label>
        <input
          className="w-full border rounded px-3 py-2 text-sm"
          placeholder="e.g. support, email, crm — defaults to localhost"
          value={form.domain}
          onChange={(e) => update("domain", e.target.value)}
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">
          Policy Requirements <span className="font-normal text-gray-400">(optional)</span>
        </label>
        <textarea
          className="w-full border rounded px-3 py-2 text-sm h-20"
          placeholder="e.g. The agent cannot delete records. Bulk operations are limited to 10 items."
          value={form.policy_requirements}
          onChange={(e) => update("policy_requirements", e.target.value)}
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">
          Reward Requirements <span className="font-normal text-gray-400">(optional)</span>
        </label>
        <textarea
          className="w-full border rounded px-3 py-2 text-sm h-20"
          placeholder="e.g. Reward speed. Penalize errors heavily. Binary: full reward only if task completes in under 5 steps."
          value={form.reward_requirements}
          onChange={(e) => update("reward_requirements", e.target.value)}
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">TTL (days)</label>
        <input
          type="number"
          min={1}
          max={365}
          className="w-full border rounded px-3 py-2 text-sm"
          value={form.ttl_days}
          onChange={(e) => update("ttl_days", parseInt(e.target.value, 10))}
        />
      </div>

      {submitError && (
        <p className="text-red-500 text-sm p-2 bg-red-50 rounded">{submitError}</p>
      )}

      <button
        type="submit"
        disabled={submitting}
        className="w-full py-2 bg-blue-600 text-white rounded hover:bg-blue-700 font-medium disabled:opacity-60 disabled:cursor-not-allowed"
      >
        {submitting ? "Submitting…" : "Generate Environment"}
      </button>
    </form>
    </>
  );
}
