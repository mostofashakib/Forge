"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

type ScoringMethod = "llm" | "embeddings" | "rouge" | "bleu";

const SCORING_METHODS: {
  id: ScoringMethod;
  label: string;
  description: string;
  noLLM?: boolean;
}[] = [
  {
    id: "llm",
    label: "LLM-as-judge",
    description: "Claude Haiku evaluates each trajectory against your requirements. Most flexible, requires API calls.",
  },
  {
    id: "embeddings",
    label: "Sentence Embeddings",
    description: "Cosine similarity between requirement text and trajectory using all-MiniLM-L6-v2. Fast, no LLM.",
    noLLM: true,
  },
  {
    id: "rouge",
    label: "ROUGE-L",
    description: "Longest common subsequence overlap between requirements and trajectory text. Deterministic, no LLM.",
    noLLM: true,
  },
  {
    id: "bleu",
    label: "BLEU",
    description: "N-gram precision overlap between requirements and trajectory. Best for short, structured outputs. No LLM.",
    noLLM: true,
  },
];

export default function RewardPage() {
  const params = useParams<{ env_name: string }>();
  const envName = params.env_name;

  const [text, setText] = useState("");
  const [selectedMethods, setSelectedMethods] = useState<ScoringMethod[]>(["llm"]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    const data = await fetch(`${API_BASE}/api/sandbox/${envName}/evaluate`)
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
    setText(data?.reward_requirements ?? "");
    const methods = data?.scoring_methods as ScoringMethod[] | undefined;
    setSelectedMethods(methods?.length ? methods : ["llm"]);
    setLoading(false);
  }, [envName]);

  useEffect(() => { load(); }, [load]);

  function toggleMethod(id: ScoringMethod) {
    setSelectedMethods((prev) => {
      if (prev.includes(id)) {
        // Prevent deselecting the last method.
        if (prev.length === 1) return prev;
        return prev.filter((m) => m !== id);
      }
      return [...prev, id];
    });
  }

  async function save() {
    setSaving(true);
    setError("");
    setSaved(false);
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${envName}/evaluate`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reward_requirements: text, scoring_methods: selectedMethods }),
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

  const hasLLM = selectedMethods.includes("llm");

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
        <div className="space-y-5">
          {/* Scoring methods */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <label className="text-sm font-medium">Scoring methods</label>
              <span className="text-xs text-muted-foreground">
                — {selectedMethods.length} selected
                {selectedMethods.length > 1 && ", scores averaged"}
              </span>
            </div>
            <div className="space-y-2">
              {SCORING_METHODS.map((m) => {
                const active = selectedMethods.includes(m.id);
                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => toggleMethod(m.id)}
                    className={`w-full text-left px-4 py-3 rounded-lg border-2 transition-colors ${
                      active
                        ? "border-foreground bg-muted/20"
                        : "border-border hover:border-foreground/30"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-0.5">
                      <span
                        className={`w-3.5 h-3.5 rounded border-2 flex items-center justify-center shrink-0 transition-colors ${
                          active ? "border-foreground bg-foreground" : "border-muted-foreground"
                        }`}
                      >
                        {active && (
                          <svg className="w-2 h-2 text-background" fill="currentColor" viewBox="0 0 8 8">
                            <path d="M1.5 4L3.5 6L6.5 2" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        )}
                      </span>
                      <span className="text-sm font-medium">{m.label}</span>
                      {m.noLLM && (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-green-100 text-green-700 font-medium">
                          No LLM
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground pl-5">{m.description}</p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Requirements text */}
          <div>
            <label className="block text-sm font-medium mb-1.5">Requirements</label>
            <textarea
              className="w-full border rounded-lg px-3 py-2.5 text-sm font-mono resize-y min-h-50 focus:outline-none focus:ring-2 focus:ring-foreground/20"
              placeholder={
                "Describe reward criteria in plain English.\n\nExamples:\n• Prefer agents that complete the task in fewer than 10 steps\n• Penalise agents that produce non-zero exit codes more than 3 times in a row\n• Reward efficiency: ideal completion in 5 steps, penalise linearly beyond\n• A trajectory that partially completes the task is worth 0.3–0.5"
              }
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <p className="text-xs text-muted-foreground mt-1">
              {hasLLM
                ? "Plain English is fine — the LLM interprets these criteria when scoring."
                : "Used as the reference text for similarity comparison during ML scoring."}
            </p>
          </div>

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
