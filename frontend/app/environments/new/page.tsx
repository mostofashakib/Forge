"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { API_BASE } from "@/lib/api";
import { AgentProgressChecklist } from "@/components/AgentProgressChecklist";

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
  const [step, setStep] = useState<"form" | "building" | "done" | "error">("form");
  const [jobId, setJobId] = useState("");
  const [envName, setEnvName] = useState("");
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

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const err = await res.json();
        setSubmitError(err.detail ?? "Failed to create sandbox");
        return;
      }
      const data = await res.json();
      setJobId(data.job_id);
      setEnvName(data.env_name);
      setStep("building");
    } catch {
      setSubmitError("Network error — is the backend running?");
    }
  }

  if (step === "building") {
    return (
      <div className="max-w-lg mx-auto p-8">
        <h1 className="text-xl font-bold mb-2">Building your environment</h1>
        <p className="text-gray-500 text-sm mb-6">
          Five agents are running in parallel. This takes 30–90 seconds.
        </p>
        <AgentProgressChecklist
          jobId={jobId}
          onDone={() => setStep("done")}
          onError={() => setStep("error")}
        />
      </div>
    );
  }

  if (step === "done") {
    return (
      <div className="max-w-lg mx-auto p-8 text-center space-y-4">
        <h1 className="text-2xl font-bold text-green-600">Environment ready</h1>
        <p className="text-gray-600">{envName} has been generated and is ready to launch.</p>
        <button
          onClick={() => router.push(`/environments/${envName}/sandbox`)}
          className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Open Sandbox →
        </button>
      </div>
    );
  }

  if (step === "error") {
    return (
      <div className="max-w-lg mx-auto p-8 text-center space-y-4">
        <h1 className="text-xl font-bold text-red-600">Generation failed</h1>
        <button onClick={() => setStep("form")} className="px-4 py-2 border rounded">
          Try again
        </button>
      </div>
    );
  }

  return (
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
        <label className="block text-sm font-medium mb-1">Domain</label>
        <input
          className="w-full border rounded px-3 py-2 text-sm"
          placeholder="e.g. support, email, crm"
          value={form.domain}
          onChange={(e) => update("domain", e.target.value)}
          required
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
        className="w-full py-2 bg-blue-600 text-white rounded hover:bg-blue-700 font-medium"
      >
        Generate Environment
      </button>
    </form>
  );
}
