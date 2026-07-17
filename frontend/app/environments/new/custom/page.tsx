"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { Toast } from "@/components/Toast";
import { useSandboxCapacity } from "@/lib/useSandboxCapacity";
const TTL_OPTIONS = [7, 30, 90, 365];
type BuildMode = "research" | "manual";

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

function projectIdentityFromUrl(rawUrl: string): { envName: string; productName: string; domain: string } | null {
  try {
    const parsed = new URL(rawUrl);
    if (!(["http:", "https:"] as string[]).includes(parsed.protocol)) return null;
    const domain = parsed.hostname.replace(/^www\./, "");
    const pathParts = parsed.pathname.split("/").filter(Boolean);
    const sourceKey = ["github.com", "gitlab.com", "bitbucket.org"].includes(domain) && pathParts.length > 0
      ? pathParts[pathParts.length - 1].replace(/\.git$/, "")
      : domain.split(".")[0];
    const envName = `${sourceKey.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "")}_env`;
    const productName = sourceKey.replace(/[-_]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
    return envName && productName ? { envName, productName, domain } : null;
  } catch {
    return null;
  }
}

export default function CustomEnvironmentPage() {
  const router = useRouter();
  const [showToast, setShowToast] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [mode, setMode] = useState<BuildMode>("research");
  const { activeCount, limit, error: capacityError } = useSandboxCapacity();
  const [form, setForm] = useState<FormState>({
    env_name: "",
    description: "",
    domain: "",
    policy_requirements: "",
    reward_requirements: "",
    reference_urls: "",
    use_user_researcher: true,
    source_product_name: "",
    source_product_url: "",
    ttl_days: 30,
  });

  const atLimit = activeCount !== null && limit !== null && activeCount >= limit;
  const researchIdentity = projectIdentityFromUrl(form.source_product_url.trim());
  const researchIncomplete = mode === "research" && !researchIdentity;

  function update<K extends keyof FormState>(field: K, value: FormState[K]) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    if (mode === "research" && !researchIdentity) {
      setSubmitError("Enter a complete http or https project URL.");
      return;
    }
    setSubmitting(true);
    try {
      const researchPayload = mode === "research" && researchIdentity
        ? {
            env_name: researchIdentity.envName,
            description: form.description.trim(),
            domain: researchIdentity.domain,
            policy_requirements: "",
            reward_requirements: "",
            reference_urls: [],
            use_user_researcher: true,
            source_product_name: researchIdentity.productName,
            source_product_url: form.source_product_url.trim(),
            ttl_days: form.ttl_days,
          }
        : null;
      const manualPayload = {
        ...form,
        reference_urls: [],
        use_user_researcher: false,
        source_product_name: "",
        source_product_url: "",
      };
      const res = await fetch(`${API_BASE}/api/sandbox/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...(researchPayload ?? manualPayload), env_type: "general" }),
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

      <div className="custom-builder">
        <header className="custom-builder__hero">
          <div>
            <Link href="/environments/new" className="custom-builder__back">← Build paths</Link>
            <span>Generative foundry / Custom</span>
            <h1>BRING THE<br /><em>REAL WORLD.</em></h1>
            <p>Start with a live product and let the researcher map it, or specify the environment yourself with complete control.</p>
          </div>
          <aside>
            <span>Build capacity</span>
            <strong>{activeCount ?? "--"}<small>/ {limit ?? "--"}</small></strong>
            <p>{atLimit ? "No build slots available" : "Local compiler ready"}</p>
          </aside>
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

        <div className="custom-builder__modes" role="tablist" aria-label="Custom environment build method">
          <button type="button" role="tab" aria-selected={mode === "research"} onClick={() => setMode("research")} className={mode === "research" ? "is-active" : ""}>
            <span>01</span><div><strong>Research a real project</strong><small>Drop a link. Forge investigates the rest.</small></div><i>↗</i>
          </button>
          <button type="button" role="tab" aria-selected={mode === "manual"} onClick={() => setMode("manual")} className={mode === "manual" ? "is-active" : ""}>
            <span>02</span><div><strong>Specify it manually</strong><small>Define the system, policy, and reward yourself.</small></div><i>↗</i>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="custom-builder__form">
          {mode === "research" ? (
            <section className="research-launcher" role="tabpanel">
              <div className="research-launcher__intro">
                <span>User researcher / armed</span>
                <h2>Drop the source.<br />We map the system.</h2>
                <p>The researcher reads the public project, identifies workflows and states, then briefs the UI, backend, RL, and review agents.</p>
              </div>
              <div className="research-launcher__fields">
                <label>
                  <span>Real project URL <b>Required</b></span>
                  <div className="research-url-input"><i>↗</i><input type="url" value={form.source_product_url} onChange={(e) => update("source_product_url", e.target.value)} placeholder="https://your-project.com" disabled={atLimit} required autoFocus /></div>
                  {form.source_product_url && !researchIdentity && <small className="research-field-error">Enter a complete http or https URL.</small>}
                  {researchIdentity && <small className="research-field-preview">Environment identity: <strong>{researchIdentity.envName}</strong></small>}
                </label>
                <label>
                  <span>What should we focus on? <b>Optional</b></span>
                  <textarea rows={5} value={form.description} onChange={(e) => update("description", e.target.value)} placeholder="Add a workflow, audience, constraint, or research goal. Leave blank to let the researcher discover the product independently." disabled={atLimit} />
                </label>
                <div className="research-pipeline" aria-label="Research pipeline">
                  {[
                    ["01", "Inspect source"], ["02", "Map workflows"], ["03", "Brief agents"], ["04", "Build environment"],
                  ].map(([step, label]) => <div key={step}><span>{step}</span><strong>{label}</strong></div>)}
                </div>
              </div>
            </section>
          ) : (
            <div className="manual-builder" role="tabpanel">
              <section>
                <div className="manual-builder__heading"><span>01</span><div><h2>Application core</h2><p>Name the environment and describe the system the agent will operate.</p></div></div>
                <div className="manual-builder__fields">
                  <label><span>Environment name</span><input className="forge-input" placeholder="e.g. crm_support_env" value={form.env_name} onChange={(e) => update("env_name", e.target.value.replace(/\s+/g, "_"))} disabled={atLimit} required /></label>
                  <label><span>Application description</span><textarea className="forge-input resize-none" rows={6} placeholder="Describe the application, its users, core workflows, and what the agent should accomplish." value={form.description} onChange={(e) => update("description", e.target.value)} disabled={atLimit} required /></label>
                </div>
              </section>
              <section>
                <div className="manual-builder__heading"><span>02</span><div><h2>Training logic</h2><p>Optional constraints. Forge fills sensible defaults when left blank.</p></div></div>
                <div className="manual-builder__fields manual-builder__fields--split">
                  <label><span>Policy requirements</span><textarea className="forge-input resize-none" rows={4} placeholder="Rules the agent must follow." value={form.policy_requirements} onChange={(e) => update("policy_requirements", e.target.value)} disabled={atLimit} /></label>
                  <label><span>Reward requirements</span><textarea className="forge-input resize-none" rows={4} placeholder="How success should be measured." value={form.reward_requirements} onChange={(e) => update("reward_requirements", e.target.value)} disabled={atLimit} /></label>
                </div>
              </section>
              <section>
                <div className="manual-builder__heading"><span>03</span><div><h2>Runtime settings</h2><p>Set the environment namespace and retention window.</p></div></div>
                <div className="manual-builder__fields manual-builder__fields--split">
                  <label><span>Domain <b>Optional</b></span><input className="forge-input" placeholder="e.g. support, email, crm" value={form.domain} onChange={(e) => update("domain", e.target.value)} disabled={atLimit} /></label>
                  <div><span className="manual-builder__label">TTL</span><div className="manual-builder__ttl">{TTL_OPTIONS.map((days) => <button key={days} type="button" onClick={() => update("ttl_days", days)} disabled={atLimit} className={form.ttl_days === days ? "is-active" : ""}>{days}d</button>)}</div></div>
                </div>
              </section>
            </div>
          )}

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
            className="custom-builder__submit"
          >
            <span>{submitting ? (mode === "research" ? "Researching…" : "Generating…") : (mode === "research" ? "Research & generate environment" : "Generate from specification")}</span><span>↗</span>
          </button>
          <p className="custom-builder__estimate">{mode === "research" ? "The user researcher starts first, then hands evidence to the build agents." : "Forge scaffolds the app, policy, and reward function from your specification."} Usually takes 2–5 minutes.</p>
        </form>
      </div>
    </>
  );
}
