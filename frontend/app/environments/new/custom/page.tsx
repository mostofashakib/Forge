"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { Toast } from "@/components/Toast";
import { useSandboxCapacity } from "@/lib/useSandboxCapacity";
const TTL_OPTIONS = [7, 30, 90, 365];

interface FormState {
  env_name: string;
  description: string;
  domain: string;
  policy_requirements: string;
  reward_requirements: string;
  reference_urls: string;
  use_user_researcher: boolean;
  source_product_name: string;
  source_product_url: string;
  ttl_days: number;
}

export default function CustomEnvironmentPage() {
  const router = useRouter();
  const [showToast, setShowToast] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const { activeCount, limit, error: capacityError } = useSandboxCapacity();
  const [form, setForm] = useState<FormState>({
    env_name: "",
    description: "",
    domain: "",
    policy_requirements: "",
    reward_requirements: "",
    reference_urls: "",
    use_user_researcher: false,
    source_product_name: "",
    source_product_url: "",
    ttl_days: 30,
  });

  const atLimit = activeCount !== null && limit !== null && activeCount >= limit;
  const researchIncomplete = form.use_user_researcher && (
    !form.source_product_name.trim() || !form.source_product_url.trim()
  );

  function update<K extends keyof FormState>(field: K, value: FormState[K]) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    setSubmitting(true);
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          reference_urls: form.use_user_researcher
            ? form.reference_urls
                .split("\n")
                .map((url) => url.trim())
                .filter(Boolean)
            : [],
          env_type: "general",
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const detail = err.detail;
        setSubmitError(
          Array.isArray(detail)
            ? detail.map((e: { msg?: string }) => e.msg ?? String(e)).join("; ")
            : (detail ?? `Request failed (${res.status})`)
        );
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

      <div className="max-w-3xl mx-auto space-y-8">
        {/* Header */}
        <div className="blueprint-panel p-7 sm:p-9 flex items-start justify-between gap-6 before:absolute before:left-0 before:top-0 before:h-full before:w-2 before:bg-primary">
          <div>
            <span className="signal-chip mb-4">Generative foundry</span>
            <h1 className="text-4xl sm:text-5xl font-semibold tracking-[-0.045em] leading-none">Specify the world</h1>
            <p className="text-sm text-muted-foreground mt-3 max-w-lg">
              Describe any real-world application — Forge generates a full-stack RL environment
              with policy rules and a reward function.
            </p>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            {activeCount !== null && limit !== null && (
              <span className={`text-xs px-2.5 py-1 rounded-full font-medium ${
                atLimit ? "bg-red-100 text-red-700" : "bg-muted text-muted-foreground"
              }`}>
                {activeCount} / {limit}
              </span>
            )}
            <Link
              href="/environments/new"
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              ← Back
            </Link>
          </div>
        </div>

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

        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Core section */}
          <div className="blueprint-panel p-6 space-y-5">
            {/* Name */}
            <div>
              <label className="block text-xs font-medium text-muted-foreground uppercase tracking-widest mb-2">
                Environment Name
              </label>
              <input
                className="w-full border rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-foreground/20 disabled:opacity-50"
                placeholder="e.g. crm_support_env"
                value={form.env_name}
                onChange={(e) => update("env_name", e.target.value.replace(/\s+/g, "_"))}
                disabled={atLimit}
                required
              />
            </div>

            {/* Description — most important field */}
            <div>
              <label className="block text-xs font-medium text-muted-foreground uppercase tracking-widest mb-2">
                Application Description
              </label>
              <textarea
                className="w-full border rounded-lg px-3 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-foreground/20 disabled:opacity-50"
                rows={6}
                placeholder={"Describe the real-world application to simulate.\n\nExamples:\n• A CRM where agents handle inbound support tickets and escalate urgent ones\n• A file storage service where agents organise uploads and remove duplicates\n• An e-commerce backend where agents process orders and update inventory"}
                value={form.description}
                onChange={(e) => update("description", e.target.value)}
                disabled={atLimit}
                required
              />
              <p className="text-xs text-muted-foreground mt-1.5">
                Be specific about what the agent does and what success looks like.
              </p>
            </div>
          </div>

          {/* Training config */}
          <div className="blueprint-panel p-6 space-y-5">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
              Training Config <span className="normal-case font-normal ml-1">— optional, AI fills defaults if blank</span>
            </p>

            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1.5">
                Policy Requirements
              </label>
              <textarea
                className="w-full border rounded-lg px-3 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-foreground/20 disabled:opacity-50"
                rows={3}
                placeholder={"Rules the agent must follow.\n\ne.g. The agent must not delete records permanently."}
                value={form.policy_requirements}
                onChange={(e) => update("policy_requirements", e.target.value)}
                disabled={atLimit}
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1.5">
                Reward Requirements
              </label>
              <textarea
                className="w-full border rounded-lg px-3 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-foreground/20 disabled:opacity-50"
                rows={3}
                placeholder={"How success is measured.\n\ne.g. Reward speed. Penalise errors and unnecessary steps heavily."}
                value={form.reward_requirements}
                onChange={(e) => update("reward_requirements", e.target.value)}
                disabled={atLimit}
              />
            </div>
          </div>

          {/* Settings */}
          <div className="blueprint-panel p-6 space-y-5">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
              Settings
            </p>

            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1.5">
                Domain <span className="font-normal">(optional)</span>
              </label>
              <input
                className="w-full border rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-foreground/20 disabled:opacity-50"
                placeholder="e.g. support, email, crm — defaults to localhost"
                value={form.domain}
                onChange={(e) => update("domain", e.target.value)}
                disabled={atLimit}
              />
            </div>

            <div className="border border-foreground/20 bg-muted/30 p-4 space-y-4">
              <label className="flex items-start gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  className="mt-0.5 size-4 accent-[var(--primary)]"
                  checked={form.use_user_researcher}
                  onChange={(e) => update("use_user_researcher", e.target.checked)}
                  disabled={atLimit}
                />
                <span>
                  <span className="block text-sm font-semibold">Use the user researcher agent</span>
                  <span className="block text-xs text-muted-foreground mt-1 leading-relaxed">
                    Research the original product before generation so its workflows, states,
                    and data model can inform the prototype.
                  </span>
                </span>
              </label>

              {form.use_user_researcher && (
                <div className="border-t border-foreground/15 pt-4 space-y-4">
                  <div>
                    <label htmlFor="source-product-name" className="block text-xs font-medium text-muted-foreground mb-1.5">
                      Original product name <span className="text-destructive">*</span>
                    </label>
                    <input
                      id="source-product-name"
                      className="forge-input"
                      placeholder="e.g. Linear"
                      value={form.source_product_name}
                      onChange={(e) => update("source_product_name", e.target.value)}
                      disabled={atLimit}
                      required
                    />
                  </div>

                  <div>
                    <label htmlFor="source-product-url" className="block text-xs font-medium text-muted-foreground mb-1.5">
                      Original product URL <span className="text-destructive">*</span>
                    </label>
                    <input
                      id="source-product-url"
                      type="url"
                      className="forge-input"
                      placeholder="https://linear.app"
                      value={form.source_product_url}
                      onChange={(e) => update("source_product_url", e.target.value)}
                      disabled={atLimit}
                      required
                    />
                  </div>

                  <div>
                    <label htmlFor="reference-urls" className="block text-xs font-medium text-muted-foreground mb-1.5">
                      Additional documentation <span className="font-normal">(optional, one URL per line)</span>
                    </label>
                    <textarea
                      id="reference-urls"
                      className="forge-input resize-none"
                      rows={3}
                      placeholder={"https://docs.example.com/product\nhttps://example.com/help/workflows"}
                      value={form.reference_urls}
                      onChange={(e) => update("reference_urls", e.target.value)}
                      disabled={atLimit}
                    />
                  </div>
                </div>
              )}
            </div>

            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-2">
                TTL
              </label>
              <div className="flex gap-2">
                {TTL_OPTIONS.map((d) => (
                  <button
                    key={d}
                    type="button"
                    onClick={() => update("ttl_days", d)}
                    disabled={atLimit}
                    className={`flex-1 py-2 text-xs font-medium rounded-lg border-2 transition-colors disabled:opacity-50 ${
                      form.ttl_days === d
                        ? "border-foreground bg-foreground text-background"
                        : "border-border text-muted-foreground hover:border-foreground/30"
                    }`}
                  >
                    {d}d
                  </button>
                ))}
              </div>
            </div>
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
            disabled={submitting || atLimit || researchIncomplete}
            className="w-full py-3 text-sm font-semibold bg-foreground text-background rounded-xl hover:opacity-90 disabled:opacity-50 transition-opacity"
          >
            {submitting ? "Generating…" : "Generate Environment →"}
          </button>
          <p className="text-xs text-center text-muted-foreground -mt-2">
            Forge will scaffold the app, policy, and reward function using AI. Takes 2–5 minutes.
          </p>
        </form>
      </div>
    </>
  );
}
