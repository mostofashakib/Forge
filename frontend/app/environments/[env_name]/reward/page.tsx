"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

export default function RewardPage() {
  const params = useParams<{ env_name: string }>();
  const envName = params.env_name;

  const [text, setText] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    const data = await fetch(`${API_BASE}/api/sandbox/${envName}/evaluate`)
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
    setText(data?.reward_requirements ?? "");
    setLoading(false);
  }, [envName]);

  useEffect(() => { load(); }, [load]);

  async function save() {
    setSaving(true);
    setError("");
    setSaved(false);
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${envName}/evaluate`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reward_requirements: text }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Reward Requirements</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Define how success is measured — what a good trajectory looks like.
          </p>
        </div>
        <Link
          href={`/environments/${envName}`}
          className="text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          ← {envName}
        </Link>
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : (
        <div className="space-y-4">
          <textarea
            className="w-full border rounded-lg px-3 py-2.5 text-sm font-mono resize-y min-h-[260px] focus:outline-none focus:ring-2 focus:ring-foreground/20"
            placeholder={
              "Describe reward criteria in plain English.\n\nExamples:\n• Prefer agents that complete the task in fewer than 10 steps\n• Penalise agents that produce non-zero exit codes more than 3 times in a row\n• Reward efficiency: ideal completion in 5 steps, penalise linearly beyond\n• A trajectory that partially completes the task is worth 0.3–0.5"
            }
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <p className="text-xs text-muted-foreground">
            Plain English is fine. The LLM applies these criteria when running a Reward evaluation.
          </p>

          <div className="flex items-center gap-3">
            <button
              onClick={save}
              disabled={saving}
              className="px-4 py-2 text-sm font-medium bg-foreground text-background rounded-lg hover:opacity-90 disabled:opacity-50 transition-opacity"
            >
              {saving ? "Saving…" : saved ? "Saved ✓" : "Save"}
            </button>
            {error && <p className="text-xs text-red-600">{error}</p>}
          </div>
        </div>
      )}
    </div>
  );
}
