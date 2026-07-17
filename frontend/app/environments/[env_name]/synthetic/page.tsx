"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface GenerateResult {
  objective: string;
  episodes: string[][];
  difficulty: number;
  difficulty_label: string;
  edge_cases: string[];
}

interface ReplayStatus {
  active: boolean;
  objective?: string;
  num_episodes?: number;
  difficulty?: number;
  difficulty_label?: string;
  edge_cases?: string[];
}

type Quality = "optimal" | "diverse" | "failure_cases";

const QUALITY_OPTIONS: { value: Quality; label: string; description: string }[] = [
  { value: "optimal",       label: "Optimal",       description: "Efficient, near-perfect sequences" },
  { value: "diverse",       label: "Diverse",        description: "Mix of exploratory and direct paths" },
  { value: "failure_cases", label: "Failure cases",  description: "Wrong turns and recovery attempts" },
];

const DIFFICULTY_LABELS: Record<number, string> = {
  1: "Trivial", 2: "Beginner", 3: "Intermediate", 4: "Advanced", 5: "Expert",
};

const DIFFICULTY_DESCRIPTIONS: Record<number, string> = {
  1: "2–4 commands · common utilities only",
  2: "4–7 commands · standard tools, single-service",
  3: "7–12 commands · installs + config, two components",
  4: "12–16 commands · multi-service, debugging required",
  5: "15–20 commands · system-level, cascading failures",
};

const EDGE_CASE_OPTIONS: { id: string; label: string; description: string }[] = [
  { id: "boundary_conditions", label: "Boundary conditions",  description: "Empty files, max-length inputs, threshold outputs" },
  { id: "permission_errors",   label: "Permission errors",    description: "Read-only paths, missing sudo — must adapt" },
  { id: "missing_deps",        label: "Missing dependencies", description: "Commands not installed; must install first" },
  { id: "conflicting_state",   label: "Conflicting state",    description: "File exists, port in use, service already running" },
  { id: "recovery",            label: "Recovery",             description: "Mid-trajectory failure the agent must diagnose" },
];

const DIFFICULTY_COLOR: Record<number, string> = {
  1: "bg-green-500", 2: "bg-lime-500", 3: "bg-amber-500", 4: "bg-orange-500", 5: "bg-red-500",
};

// ---------------------------------------------------------------------------
// Episode card
// ---------------------------------------------------------------------------

function EpisodeCard({ commands, index }: { commands: string[]; index: number }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="border border-border/60 rounded-xl overflow-hidden bg-card card-shadow">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full text-left px-4 py-3.5 flex items-center justify-between hover:bg-muted/20 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="w-6 h-6 rounded-md bg-muted flex items-center justify-center text-xs font-mono font-medium text-muted-foreground shrink-0">
            {index + 1}
          </span>
          <span className="text-sm font-medium">Episode {index + 1}</span>
          <span className="text-xs text-muted-foreground px-2 py-0.5 rounded-full bg-muted">{commands.length} steps</span>
        </div>
        <svg
          width="12" height="12" viewBox="0 0 12 12" fill="none"
          stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
          className={`text-muted-foreground transition-transform ${expanded ? "rotate-180" : ""}`}
        >
          <path d="M2 4l4 4 4-4" />
        </svg>
      </button>
      {expanded && (
        <div className="border-t border-border/40 bg-muted/10 px-4 py-3 space-y-1.5 scrollbar-thin max-h-64 overflow-y-auto">
          {commands.map((cmd, i) => (
            <div key={i} className="flex items-start gap-2 font-mono text-xs">
              <span className="text-muted-foreground/50 w-5 shrink-0 text-right tabular-nums">{i + 1}</span>
              <span className="text-emerald-600 shrink-0">$</span>
              <span className="break-all text-foreground/90">{cmd}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function SyntheticPage() {
  const params = useParams<{ env_name: string }>();
  const envName = params.env_name;

  const [goal, setGoal] = useState("");
  const [numEpisodes, setNumEpisodes] = useState(3);
  const [quality, setQuality] = useState<Quality>("diverse");
  const [difficulty, setDifficulty] = useState(3);
  const [edgeCases, setEdgeCases] = useState<string[]>([]);

  const [generating, setGenerating] = useState(false);
  const [importing, setImporting] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [suggestingGoals, setSuggestingGoals] = useState(false);
  const [suggestedGoals, setSuggestedGoals] = useState<string[]>([]);
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [imported, setImported] = useState(false);
  const [replayStatus, setReplayStatus] = useState<ReplayStatus | null>(null);
  const [error, setError] = useState("");

  const loadStatus = useCallback(async () => {
    const data = await fetch(`${API_BASE}/api/sandbox/${envName}/synthetic`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
    setReplayStatus(data);
  }, [envName]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadStatus(), 0);
    return () => window.clearTimeout(timer);
  }, [loadStatus]);

  const suggestGoals = useCallback(async () => {
    setSuggestingGoals(true);
    setSuggestedGoals([]);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${envName}/synthetic/suggest-goals`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ difficulty, existing_goals: goal.trim() ? [goal.trim()] : [], num_suggestions: 5 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setSuggestedGoals(data.goals ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Suggestion failed");
    } finally {
      setSuggestingGoals(false);
    }
  }, [envName, difficulty, goal]);

  const generate = useCallback(async () => {
    if (!goal.trim()) return;
    setGenerating(true);
    setError("");
    setResult(null);
    setImported(false);
    setSuggestedGoals([]);
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${envName}/synthetic/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ research_goal: goal, num_episodes: numEpisodes, quality, difficulty, edge_cases: edgeCases }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setResult(data as GenerateResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setGenerating(false);
    }
  }, [envName, goal, numEpisodes, quality, difficulty, edgeCases]);

  const importTrajectories = useCallback(async () => {
    if (!result) return;
    setImporting(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${envName}/synthetic/import`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          research_goal: goal,
          objective: result.objective,
          episodes: result.episodes,
          difficulty: result.difficulty,
          edge_cases: result.edge_cases,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setImported(true);
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Import failed");
    } finally {
      setImporting(false);
    }
  }, [envName, goal, result, loadStatus]);

  const clearTrajectories = useCallback(async () => {
    setClearing(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${envName}/synthetic`, { method: "DELETE" });
      if (!res.ok && res.status !== 204) throw new Error(`HTTP ${res.status}`);
      setImported(false);
      setResult(null);
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Clear failed");
    } finally {
      setClearing(false);
    }
  }, [envName, loadStatus]);

  function toggleEdgeCase(id: string) {
    setEdgeCases((prev) => prev.includes(id) ? prev.filter((e) => e !== id) : [...prev, id]);
  }

  return (
    <div className="space-y-6 max-w-2xl">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Synthetic Data</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-sm">
            Generate an epoch of synthetic episodes from a research goal. Trajectories are
            replayed during agent runs instead of live LLM inference.
          </p>
        </div>
        <Link href={`/environments/${envName}`} className="text-sm text-muted-foreground hover:text-foreground transition-colors shrink-0">
          ← {envName}
        </Link>
      </div>

      {/* Active epoch banner */}
      {replayStatus?.active && (
        <div className="border border-indigo-200/80 bg-indigo-50/60 rounded-xl px-4 py-3.5 flex items-center justify-between gap-4">
          <div className="space-y-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="w-2 h-2 rounded-full bg-indigo-500 inline-block" />
              <p className="text-sm font-semibold text-indigo-800">
                Active epoch — {replayStatus.num_episodes} episode{replayStatus.num_episodes !== 1 ? "s" : ""}
              </p>
              {replayStatus.difficulty_label && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-600 font-medium">
                  {replayStatus.difficulty_label}
                </span>
              )}
              {(replayStatus.edge_cases?.length ?? 0) > 0 && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-600 font-medium">
                  {replayStatus.edge_cases!.length} edge case{replayStatus.edge_cases!.length !== 1 ? "s" : ""}
                </span>
              )}
            </div>
            <p className="text-xs text-indigo-600/80 font-mono truncate">{replayStatus.objective}</p>
          </div>
          <button
            onClick={clearTrajectories}
            disabled={clearing}
            className="shrink-0 text-xs text-indigo-700 hover:text-indigo-900 font-medium underline underline-offset-2 disabled:opacity-50"
          >
            {clearing ? "Clearing…" : "Clear epoch"}
          </button>
        </div>
      )}

      {/* Config form */}
      <div className="space-y-4">

        {/* Research goal */}
        <div className="border border-border/60 rounded-xl p-5 bg-card card-shadow space-y-3">
          <div className="flex items-center justify-between">
            <p className="section-label">Research Goal</p>
            <button
              type="button"
              onClick={suggestGoals}
              disabled={suggestingGoals}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground disabled:opacity-50 transition-colors"
            >
              {suggestingGoals ? (
                <>
                  <span className="w-3 h-3 rounded-full border border-muted-foreground/50 border-t-foreground animate-spin" />
                  Suggesting…
                </>
              ) : (
                <>
                  <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
                    <path
                      d="M5.5 0.5C5.7 2 6.6 3 8 3.5C6.6 4 5.7 5 5.5 6.5C5.3 5 4.4 4 3 3.5C4.4 3 5.3 2 5.5 0.5Z"
                      fill="currentColor"
                    />
                    <path d="M9 7C9.1 7.6 9.5 8 10 8.2C9.5 8.4 9.1 8.8 9 9.4C8.9 8.8 8.5 8.4 8 8.2C8.5 8 8.9 7.6 9 7Z" fill="currentColor" opacity="0.7" />
                  </svg>
                  Suggest goals
                </>
              )}
            </button>
          </div>
          <textarea
            className="forge-input resize-none w-full"
            rows={3}
            placeholder="e.g. Generate diverse examples of a CLI agent setting up a Python web server and handling common errors"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
          />
          {suggestedGoals.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <p className="text-xs text-muted-foreground font-medium">Click to use:</p>
              {suggestedGoals.map((g, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => { setGoal(g); setSuggestedGoals([]); }}
                  className="w-full text-left text-xs px-3 py-2.5 rounded-lg border border-border hover:border-primary/40 hover:bg-primary/5 transition-all"
                >
                  {g}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Episodes + Quality */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Episodes */}
          <div className="border border-border/60 rounded-xl p-5 bg-card card-shadow">
            <p className="section-label mb-4">Episodes</p>
            <div className="flex items-center gap-4">
              <input
                type="range" min={1} max={10} step={1}
                value={numEpisodes}
                onChange={(e) => setNumEpisodes(Number(e.target.value))}
                className="flex-1 accent-primary"
              />
              <span className="text-2xl font-semibold w-6 text-center tabular-nums">{numEpisodes}</span>
            </div>
            <div className="flex justify-between mt-2">
              <span className="text-xs text-muted-foreground">1</span>
              <span className="text-xs text-muted-foreground">10</span>
            </div>
          </div>

          {/* Quality */}
          <div className="border border-border/60 rounded-xl p-5 bg-card card-shadow">
            <p className="section-label mb-4">Quality</p>
            <div className="space-y-2">
              {QUALITY_OPTIONS.map((opt) => (
                <label key={opt.value} className="flex items-start gap-2.5 cursor-pointer group">
                  <div className={`mt-0.5 w-4 h-4 rounded-full border-2 flex items-center justify-center shrink-0 transition-colors ${quality === opt.value ? "border-primary bg-primary" : "border-border group-hover:border-primary/50"}`}>
                    {quality === opt.value && <span className="w-1.5 h-1.5 rounded-full bg-white" />}
                  </div>
                  <input
                    type="radio" name="quality" value={opt.value}
                    checked={quality === opt.value}
                    onChange={() => setQuality(opt.value)}
                    className="sr-only"
                  />
                  <div>
                    <p className="text-sm font-medium leading-tight">{opt.label}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{opt.description}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Difficulty */}
        <div className="border border-border/60 rounded-xl p-5 bg-card card-shadow">
          <div className="flex items-center justify-between mb-4">
            <p className="section-label">Difficulty</p>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${DIFFICULTY_COLOR[difficulty]}`} />
              <span className="text-sm font-semibold">{DIFFICULTY_LABELS[difficulty]}</span>
            </div>
          </div>
          <input
            type="range" min={1} max={5} step={1}
            value={difficulty}
            onChange={(e) => setDifficulty(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <div className="flex justify-between mt-2 mb-3">
            {[1, 2, 3, 4, 5].map((d) => (
              <span key={d} className={`text-xs ${d === difficulty ? "text-foreground font-semibold" : "text-muted-foreground"}`}>
                {DIFFICULTY_LABELS[d]}
              </span>
            ))}
          </div>
          <p className="text-xs text-muted-foreground bg-muted/40 px-3 py-2 rounded-lg">
            {DIFFICULTY_DESCRIPTIONS[difficulty]}
          </p>
        </div>

        {/* Edge cases */}
        <div className="border border-border/60 rounded-xl p-5 bg-card card-shadow">
          <div className="flex items-center justify-between mb-4">
            <p className="section-label">Edge Case Injection</p>
            <span className="text-xs text-muted-foreground">optional</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {EDGE_CASE_OPTIONS.map((opt) => {
              const active = edgeCases.includes(opt.id);
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => toggleEdgeCase(opt.id)}
                  className={`text-left px-3.5 py-3 rounded-lg border-2 transition-all ${
                    active
                      ? "border-primary/60 bg-primary/5"
                      : "border-border hover:border-primary/30 hover:bg-muted/20"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className={`w-3.5 h-3.5 rounded border-2 shrink-0 flex items-center justify-center transition-colors ${active ? "border-primary bg-primary" : "border-muted-foreground/50"}`}>
                      {active && (
                        <svg className="w-2 h-2 text-white" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 8 8" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M1.5 4L3.5 6L6.5 2" />
                        </svg>
                      )}
                    </span>
                    <span className="text-xs font-semibold">{opt.label}</span>
                  </div>
                  <p className="text-xs text-muted-foreground pl-5">{opt.description}</p>
                </button>
              );
            })}
          </div>
          <p className={`text-xs text-muted-foreground mt-3 transition-opacity ${edgeCases.length > 0 ? "opacity-100" : "opacity-0"}`}>
            {edgeCases.length} edge case type{edgeCases.length !== 1 ? "s" : ""} will be woven into each trajectory.
          </p>
        </div>

        {/* Generate button */}
        <div className="flex items-center gap-3">
          <button
            onClick={generate}
            disabled={generating || !goal.trim()}
            className="flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white bg-primary rounded-lg hover:bg-primary/90 disabled:opacity-50 transition-all shadow-sm hover:shadow-md"
          >
            {generating ? (
              <>
                <span className="w-3.5 h-3.5 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                Generating…
              </>
            ) : (
              <>
                <svg width="13" height="13" viewBox="0 0 11 11" fill="none">
                  <path d="M5.5 0.5C5.7 2 6.6 3 8 3.5C6.6 4 5.7 5 5.5 6.5C5.3 5 4.4 4 3 3.5C4.4 3 5.3 2 5.5 0.5Z" fill="currentColor" />
                  <path d="M9 7C9.1 7.6 9.5 8 10 8.2C9.5 8.4 9.1 8.8 9 9.4C8.9 8.8 8.5 8.4 8 8.2C8.5 8 8.9 7.6 9 7Z" fill="currentColor" opacity="0.7" />
                </svg>
                Generate epoch
              </>
            )}
          </button>
          {error && (
            <p className="text-xs text-red-600 bg-red-50 px-3 py-1.5 rounded-lg border border-red-200">{error}</p>
          )}
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-4 pt-2">
          <div className="border border-border/60 rounded-xl p-5 bg-card card-shadow">
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-2 min-w-0">
                <h2 className="text-sm font-semibold">Generated epoch</h2>
                <p className="text-xs text-muted-foreground font-mono truncate">
                  {result.objective}
                </p>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">
                    {result.difficulty_label}
                  </span>
                  {result.edge_cases.map((ec) => (
                    <span key={ec} className="text-xs px-2 py-0.5 rounded-full bg-orange-50 text-orange-700 border border-orange-200/60">
                      {ec.replace(/_/g, " ")}
                    </span>
                  ))}
                </div>
              </div>
              <div className="shrink-0">
                {imported ? (
                  <div className="flex items-center gap-1.5 text-xs text-green-600 font-medium bg-green-50 px-3 py-1.5 rounded-lg border border-green-200">
                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="6" cy="6" r="4.5" />
                      <path d="M4 6.5l1.5 1.5 3-3" />
                    </svg>
                    Epoch created
                  </div>
                ) : (
                  <button
                    onClick={importTrajectories}
                    disabled={importing}
                    className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-foreground text-background rounded-lg hover:opacity-90 disabled:opacity-50 transition-opacity"
                  >
                    {importing ? "Creating…" : `Create epoch (${result.episodes.length})`}
                    {!importing && <span className="opacity-60">→</span>}
                  </button>
                )}
              </div>
            </div>
          </div>

          {result.episodes.map((commands, i) => (
            <EpisodeCard key={i} commands={commands} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}
