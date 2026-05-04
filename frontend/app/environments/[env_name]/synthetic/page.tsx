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
  { value: "optimal",       label: "Optimal",       description: "Efficient sequences that reach the goal in few steps" },
  { value: "diverse",       label: "Diverse",        description: "Mix of exploratory and direct command paths" },
  { value: "failure_cases", label: "Failure cases",  description: "Sequences with wrong turns and recovery attempts" },
];

const DIFFICULTY_LABELS: Record<number, string> = {
  1: "Trivial", 2: "Beginner", 3: "Intermediate", 4: "Advanced", 5: "Expert",
};

const DIFFICULTY_DESCRIPTIONS: Record<number, string> = {
  1: "2–4 commands, common utilities only, unambiguous success",
  2: "4–7 commands, standard tools, single-service goals",
  3: "7–12 commands, installs + config, two components",
  4: "12–16 commands, multi-service setup, debugging required",
  5: "15–20 commands, system-level, cascading failures, high ambiguity",
};

const EDGE_CASE_OPTIONS: { id: string; label: string; description: string }[] = [
  { id: "boundary_conditions", label: "Boundary conditions",  description: "Empty files, max-length inputs, threshold outputs" },
  { id: "permission_errors",   label: "Permission errors",    description: "Read-only paths, missing sudo — agent must adapt" },
  { id: "missing_deps",        label: "Missing dependencies", description: "Commands or packages not installed; must install first" },
  { id: "conflicting_state",   label: "Conflicting state",    description: "File already exists, port in use, service already running" },
  { id: "recovery",            label: "Recovery",             description: "Mid-trajectory failure the agent must diagnose and fix" },
];

// ---------------------------------------------------------------------------
// Episode card
// ---------------------------------------------------------------------------

function EpisodeCard({ commands, index }: { commands: string[]; index: number }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full text-left px-4 py-3 flex items-center justify-between hover:bg-muted/20 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-muted-foreground">Episode {index + 1}</span>
          <span className="text-xs text-muted-foreground">{commands.length} steps</span>
        </div>
        <span className="text-xs text-muted-foreground">{expanded ? "▲" : "▼"}</span>
      </button>
      {expanded && (
        <div className="border-t bg-muted/10 px-4 py-3 space-y-1">
          {commands.map((cmd, i) => (
            <div key={i} className="flex items-start gap-2 font-mono text-xs">
              <span className="text-muted-foreground w-5 shrink-0 text-right">{i + 1}</span>
              <span className="text-green-700 shrink-0">$</span>
              <span className="break-all">{cmd}</span>
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

  useEffect(() => { loadStatus(); }, [loadStatus]);

  // ---- actions ----

  const suggestGoals = useCallback(async () => {
    setSuggestingGoals(true);
    setSuggestedGoals([]);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${envName}/synthetic/suggest-goals`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          difficulty,
          existing_goals: goal.trim() ? [goal.trim()] : [],
          num_suggestions: 5,
        }),
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
    setEdgeCases((prev) =>
      prev.includes(id) ? prev.filter((e) => e !== id) : [...prev, id]
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Synthetic Data</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Generate an epoch of episodes from a research goal. Trajectories are replayed
            during agent runs instead of live LLM inference.
          </p>
        </div>
        <Link href={`/environments/${envName}`} className="text-sm text-muted-foreground hover:text-foreground transition-colors">
          ← {envName}
        </Link>
      </div>

      {/* Active epoch banner */}
      {replayStatus?.active && (
        <div className="border border-indigo-200 bg-indigo-50 rounded-lg px-4 py-3 flex items-center justify-between gap-4">
          <div className="space-y-0.5">
            <p className="text-sm font-medium text-indigo-800">
              ● Active epoch — {replayStatus.num_episodes} episode{replayStatus.num_episodes !== 1 ? "s" : ""}
              {replayStatus.difficulty_label && (
                <span className="ml-2 text-xs font-normal text-indigo-600">
                  · {replayStatus.difficulty_label}
                  {replayStatus.edge_cases?.length ? ` · ${replayStatus.edge_cases.length} edge case type${replayStatus.edge_cases.length !== 1 ? "s" : ""}` : ""}
                </span>
              )}
            </p>
            <p className="text-xs text-indigo-600 font-mono">{replayStatus.objective}</p>
          </div>
          <button
            onClick={clearTrajectories}
            disabled={clearing}
            className="shrink-0 text-xs text-indigo-700 hover:text-indigo-900 underline disabled:opacity-50"
          >
            {clearing ? "Clearing…" : "Clear epoch"}
          </button>
        </div>
      )}

      {/* Config form */}
      <div className="border rounded-lg p-5 space-y-5">

        {/* Research goal + suggest */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
              Research Goal
            </label>
            <button
              type="button"
              onClick={suggestGoals}
              disabled={suggestingGoals}
              className="text-xs text-muted-foreground hover:text-foreground underline disabled:opacity-50 transition-colors"
            >
              {suggestingGoals ? "Suggesting…" : "Suggest goals →"}
            </button>
          </div>
          <textarea
            className="w-full border rounded-lg px-3 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-foreground/20"
            rows={3}
            placeholder="e.g. Generate diverse examples of a CLI agent setting up a Python web server and handling common errors"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
          />
          {/* Suggested goals */}
          {suggestedGoals.length > 0 && (
            <div className="mt-2 space-y-1.5">
              <p className="text-xs text-muted-foreground font-medium">Click to use a suggestion:</p>
              {suggestedGoals.map((g, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => { setGoal(g); setSuggestedGoals([]); }}
                  className="w-full text-left text-xs px-3 py-2 rounded-lg border border-border hover:border-foreground/30 hover:bg-muted/10 transition-colors"
                >
                  {g}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          {/* Episodes */}
          <div>
            <label className="block text-xs font-medium text-muted-foreground uppercase tracking-widest mb-2">
              Episodes
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range" min={1} max={10} step={1}
                value={numEpisodes}
                onChange={(e) => setNumEpisodes(Number(e.target.value))}
                className="flex-1"
              />
              <span className="text-sm font-semibold w-4 text-right">{numEpisodes}</span>
            </div>
          </div>

          {/* Quality */}
          <div>
            <label className="block text-xs font-medium text-muted-foreground uppercase tracking-widest mb-2">
              Quality
            </label>
            <div className="flex flex-col gap-1.5">
              {QUALITY_OPTIONS.map((opt) => (
                <label key={opt.value} className="flex items-start gap-2 cursor-pointer">
                  <input
                    type="radio" name="quality" value={opt.value}
                    checked={quality === opt.value}
                    onChange={() => setQuality(opt.value)}
                    className="mt-0.5 accent-foreground"
                  />
                  <span className="text-sm">
                    <span className="font-medium">{opt.label}</span>
                    <span className="text-muted-foreground text-xs ml-1.5">— {opt.description}</span>
                  </span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Difficulty */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
              Difficulty
            </label>
            <span className="text-xs font-semibold">{DIFFICULTY_LABELS[difficulty]}</span>
          </div>
          <input
            type="range" min={1} max={5} step={1}
            value={difficulty}
            onChange={(e) => setDifficulty(Number(e.target.value))}
            className="w-full accent-foreground"
          />
          <div className="flex justify-between mt-1">
            {[1, 2, 3, 4, 5].map((d) => (
              <span key={d} className={`text-xs ${d === difficulty ? "text-foreground font-medium" : "text-muted-foreground"}`}>
                {DIFFICULTY_LABELS[d]}
              </span>
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-1.5">{DIFFICULTY_DESCRIPTIONS[difficulty]}</p>
        </div>

        {/* Edge cases */}
        <div>
          <label className="block text-xs font-medium text-muted-foreground uppercase tracking-widest mb-2">
            Edge Case Injection <span className="normal-case font-normal ml-1">— optional</span>
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {EDGE_CASE_OPTIONS.map((opt) => {
              const active = edgeCases.includes(opt.id);
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => toggleEdgeCase(opt.id)}
                  className={`text-left px-3 py-2.5 rounded-lg border-2 transition-colors ${
                    active ? "border-foreground bg-muted/20" : "border-border hover:border-foreground/30"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className={`w-3 h-3 rounded border-2 shrink-0 transition-colors ${active ? "border-foreground bg-foreground" : "border-muted-foreground"}`} />
                    <span className="text-xs font-medium">{opt.label}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5 pl-5">{opt.description}</p>
                </button>
              );
            })}
          </div>
          <p className={`text-xs text-muted-foreground mt-1.5 transition-opacity ${edgeCases.length > 0 ? "opacity-100" : "opacity-0"}`}>
            {edgeCases.length} edge case type{edgeCases.length !== 1 ? "s" : ""} will be woven into each trajectory.
          </p>
        </div>

        <div className="flex items-center gap-3 pt-1">
          <button
            onClick={generate}
            disabled={generating || !goal.trim()}
            className="px-5 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {generating ? "Generating…" : "Generate epoch"}
          </button>
          {error && <p className="text-xs text-red-600">{error}</p>}
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-4">
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-1">
              <h2 className="text-sm font-semibold">Generated epoch</h2>
              <p className="text-xs text-muted-foreground">
                Objective: <span className="font-mono">{result.objective}</span>
              </p>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                  {result.difficulty_label}
                </span>
                {result.edge_cases.map((ec) => (
                  <span key={ec} className="text-xs px-2 py-0.5 rounded-full bg-orange-100 text-orange-700">
                    {ec.replace(/_/g, " ")}
                  </span>
                ))}
              </div>
            </div>
            <div className="shrink-0">
              {imported ? (
                <span className="text-xs text-green-600 font-medium">✓ Epoch created — agent runs will replay these trajectories</span>
              ) : (
                <button
                  onClick={importTrajectories}
                  disabled={importing}
                  className="px-4 py-2 text-sm font-medium text-white bg-foreground rounded-lg hover:opacity-90 disabled:opacity-50 transition-opacity"
                >
                  {importing ? "Creating epoch…" : `Create epoch (${result.episodes.length} episode${result.episodes.length !== 1 ? "s" : ""}) →`}
                </button>
              )}
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
