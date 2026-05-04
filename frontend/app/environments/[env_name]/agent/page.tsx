"use client";

import { useEffect, useState, useCallback } from "react";
import { createPortal } from "react-dom";
import { useParams } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AgentRun {
  id: string;
  env_name: string;
  agent_id: string;
  objective: string;
  num_episodes: number;
  max_steps: number;
  divergence_threshold: number;
  consecutive_below_threshold: number;
  dead_end_patience: number;
  success_threshold: number;
  status: string;
  episodes_completed: number;
  error: string | null;
  created_at: string;
  completed_at: string | null;
}

interface AgentEpisode {
  id: string;
  run_id: string;
  episode_index: number;
  seed: number;
  status: string;
  total_steps: number;
  total_reward: number;
  final_objective_score: number;
  termination_reason: string | null;
  started_at: string;
  completed_at: string | null;
}

// General (HTTP) step
interface GeneralStep {
  step_index: number;
  state_before: Record<string, unknown>;
  action: { endpoint: string; payload: Record<string, unknown>; reasoning?: string };
  state_after: Record<string, unknown>;
  reward: number;
  objective_score: number;
  state_hash_before: string;
  state_hash_after: string;
  terminated: boolean;
  truncated: boolean;
  termination_reason: string | null;
}

// CLI step
interface CliStep {
  step_index: number;
  command: string;
  stdout: string;
  stderr: string;
  exit_code: number;
  objective_score: number;
  reward: number;
}

// Browser step
interface BrowserStep {
  step_index: number;
  action: { action_type: string; x?: number; y?: number; text?: string; key?: string; url?: string; delta_x?: number; delta_y?: number; reasoning?: string };
  screenshot_before: string;
  screenshot_after: string;
  url_before: string;
  url_after: string;
  objective_score: number;
  reward: number;
}

type TrajectoryStep = GeneralStep | CliStep | BrowserStep;

function isCliStep(s: TrajectoryStep): s is CliStep { return "command" in s; }
function isBrowserStep(s: TrajectoryStep): s is BrowserStep { return "screenshot_before" in s; }

function stepLabel(s: TrajectoryStep): string {
  if (isCliStep(s)) return s.command.slice(0, 40);
  if (isBrowserStep(s)) return s.action.action_type;
  return (s as GeneralStep).action.endpoint;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const STATUS_BADGE: Record<string, string> = {
  pending:   "bg-yellow-100 text-yellow-700",
  running:   "bg-blue-100 text-blue-700",
  completed: "bg-green-100 text-green-700",
  failed:    "bg-red-100 text-red-600",
  stopped:   "bg-gray-100 text-gray-500",
};

const TERM_BADGE: Record<string, string> = {
  success:   "bg-green-100 text-green-700",
  max_steps: "bg-yellow-100 text-yellow-700",
  diverged:  "bg-orange-100 text-orange-700",
  dead_end:  "bg-gray-100 text-gray-500",
  failed:    "bg-red-100 text-red-600",
};

function badge(map: Record<string, string>, key: string | null, fallback = "") {
  const label = key ?? fallback;
  const cls = (key && map[key]) ?? "bg-gray-100 text-gray-500";
  return <span className={`px-2 py-0.5 rounded text-xs font-medium ${cls}`}>{label}</span>;
}

function fmt(ts: string | null) {
  if (!ts) return "—";
  return new Date(ts).toLocaleString();
}

function scoreBar(score: number) {
  const pct = Math.round(score * 100);
  const color = score >= 0.8 ? "bg-green-500" : score >= 0.5 ? "bg-yellow-400" : "bg-red-400";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-200 rounded overflow-hidden">
        <div className={`h-full ${color} rounded`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-600 w-8 text-right">{pct}%</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// New Run Modal
// ---------------------------------------------------------------------------

function NewRunModal({
  envName,
  onClose,
  onCreated,
  defaultObjective,
  replayMode,
}: {
  envName: string;
  onClose: () => void;
  onCreated: () => void;
  defaultObjective?: string;
  replayMode?: boolean;
}) {
  const [objective, setObjective] = useState(defaultObjective ?? "");
  const [agentId, setAgentId] = useState("llm");
  const [numEpisodes, setNumEpisodes] = useState(5);
  const [maxSteps, setMaxSteps] = useState(50);
  const [divergenceThreshold, setDivergenceThreshold] = useState(0.2);
  const [deadEndPatience, setDeadEndPatience] = useState(5);
  const [successThreshold, setSuccessThreshold] = useState(0.9);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!objective.trim()) { setError("Objective is required"); return; }
    setSubmitting(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${envName}/agent-runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agent_id: agentId,
          objective: objective.trim(),
          num_episodes: numEpisodes,
          max_steps: maxSteps,
          divergence_threshold: divergenceThreshold,
          dead_end_patience: deadEndPatience,
          success_threshold: successThreshold,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      onCreated();
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create run");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">New Agent Run</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-xl leading-none">×</button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {replayMode && (
            <div className="flex items-start gap-2 bg-blue-50 border border-blue-200 rounded-lg px-3 py-2.5 text-sm text-blue-800">
              <span className="shrink-0">●</span>
              <span>Replay mode — the agent will follow the imported synthetic trajectory instead of using the LLM to pick actions.</span>
            </div>
          )}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Objective</label>
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm resize-none focus:ring-2 focus:ring-blue-500 focus:outline-none"
              rows={3}
              placeholder="e.g. Close all open tickets assigned to Alice"
              value={objective}
              onChange={e => setObjective(e.target.value)}
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Agent</label>
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                value={agentId}
                onChange={e => setAgentId(e.target.value)}
              >
                <option value="llm">LLM (Haiku)</option>
                <option value="llm:claude-sonnet-4-6">LLM (Sonnet)</option>
                <option value="random">Random</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Episodes</label>
              <input
                type="number" min={1} max={100}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                value={numEpisodes}
                onChange={e => setNumEpisodes(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Max steps</label>
              <input
                type="number" min={5} max={500}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                value={maxSteps}
                onChange={e => setMaxSteps(Number(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Div. threshold</label>
              <input
                type="number" min={0} max={1} step={0.05}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                value={divergenceThreshold}
                onChange={e => setDivergenceThreshold(Number(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Success ≥</label>
              <input
                type="number" min={0.5} max={1} step={0.05}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                value={successThreshold}
                onChange={e => setSuccessThreshold(Number(e.target.value))}
              />
            </div>
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          <div className="flex justify-end gap-3 pt-1">
            <button
              type="button" onClick={onClose}
              className="px-4 py-2 text-sm text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50"
            >Cancel</button>
            <button
              type="submit" disabled={submitting}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >{submitting ? "Launching…" : "Launch Run"}</button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Trajectory Drawer
// ---------------------------------------------------------------------------

function TrajectoryDrawer({
  envName,
  runId,
  episode,
  onClose,
}: {
  envName: string;
  runId: string;
  episode: AgentEpisode;
  onClose: () => void;
}) {
  const [steps, setSteps] = useState<TrajectoryStep[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<TrajectoryStep | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/sandbox/${envName}/agent-runs/${runId}/episodes/${episode.id}/trajectory`)
      .then(r => r.json())
      .then(data => { setSteps(data.steps ?? []); setLoading(false); })
      .catch(() => setLoading(false));
  }, [envName, runId, episode.id]);

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Backdrop */}
      <div className="flex-1 bg-black/40" onClick={onClose} />

      {/* Drawer */}
      <div className="w-full max-w-3xl bg-white shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b bg-gray-50">
          <div>
            <p className="text-xs text-gray-500 mb-0.5">Episode {episode.episode_index + 1}</p>
            <h2 className="font-semibold text-gray-900 text-sm font-mono">{episode.id}</h2>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right text-xs text-gray-500">
              <div>{episode.total_steps} steps · {episode.total_reward.toFixed(2)} total reward</div>
              <div>Final score: {(episode.final_objective_score * 100).toFixed(0)}%</div>
            </div>
            {badge(TERM_BADGE, episode.termination_reason, "unknown")}
            <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl leading-none">×</button>
          </div>
        </div>

        <div className="flex flex-1 overflow-hidden">
          {/* Step list */}
          <div className="w-56 border-r overflow-y-auto shrink-0">
            {loading ? (
              <div className="p-4 text-sm text-gray-400">Loading…</div>
            ) : steps.length === 0 ? (
              <div className="p-4 text-sm text-gray-400">No steps recorded</div>
            ) : steps.map(step => {
              const termReason = !isCliStep(step) && !isBrowserStep(step) ? (step as GeneralStep).termination_reason : null;
              return (
                <button
                  key={step.step_index}
                  onClick={() => setSelected(step)}
                  className={`w-full text-left px-4 py-3 border-b hover:bg-gray-50 transition-colors ${selected?.step_index === step.step_index ? "bg-blue-50 border-l-2 border-l-blue-500" : ""}`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-medium text-gray-700">Step {step.step_index + 1}</span>
                    {termReason && badge(TERM_BADGE, termReason)}
                  </div>
                  {scoreBar(step.objective_score)}
                  <p className="text-xs text-gray-400 mt-1 truncate font-mono">{stepLabel(step)}</p>
                </button>
              );
            })}
          </div>

          {/* Step detail */}
          <div className="flex-1 overflow-y-auto p-5">
            {selected ? (
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <span className="text-sm font-semibold text-gray-800">Step {selected.step_index + 1}</span>
                  <span className="text-xs text-gray-400">
                    score {(selected.objective_score * 100).toFixed(0)}% · reward {selected.reward.toFixed(3)}
                  </span>
                </div>

                {/* ── CLI step ── */}
                {isCliStep(selected) && (
                  <>
                    <div className="bg-gray-900 rounded-lg p-4">
                      <p className="text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wide">Command</p>
                      <pre className="text-sm text-green-400 font-mono whitespace-pre-wrap">$ {selected.command}</pre>
                    </div>
                    <div>
                      <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">
                        Output
                        {selected.exit_code !== 0 && (
                          <span className="ml-2 text-red-500 normal-case">exit {selected.exit_code}</span>
                        )}
                      </p>
                      <pre className="text-xs text-gray-700 whitespace-pre-wrap overflow-auto max-h-80 bg-gray-50 rounded p-3 border font-mono">
                        {selected.stdout || "(no output)"}
                      </pre>
                      {selected.stderr && (
                        <pre className="mt-2 text-xs text-red-700 whitespace-pre-wrap overflow-auto max-h-40 bg-red-50 rounded p-3 border font-mono">
                          {selected.stderr}
                        </pre>
                      )}
                    </div>
                  </>
                )}

                {/* ── Browser step ── */}
                {isBrowserStep(selected) && (
                  <>
                    <div className="bg-blue-50 rounded-lg p-4">
                      <p className="text-xs font-semibold text-blue-700 mb-2 uppercase tracking-wide">Action</p>
                      <p className="text-sm font-mono font-medium text-blue-900 mb-1">{selected.action.action_type}</p>
                      {selected.action.reasoning && (
                        <p className="text-xs text-blue-700 italic border-l-2 border-blue-300 pl-2 mb-2">
                          {selected.action.reasoning}
                        </p>
                      )}
                      <pre className="text-xs text-blue-800 whitespace-pre-wrap bg-white/60 rounded p-2">
                        {JSON.stringify(
                          Object.fromEntries(
                            Object.entries(selected.action).filter(([k]) => !["action_type","reasoning"].includes(k))
                          ),
                          null, 2
                        )}
                      </pre>
                      {(selected.url_before !== selected.url_after) && (
                        <p className="mt-2 text-xs text-gray-500">
                          <span className="font-medium">URL:</span> {selected.url_before} → {selected.url_after}
                        </p>
                      )}
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">Before</p>
                        <img
                          src={`data:image/png;base64,${selected.screenshot_before}`}
                          alt="before"
                          className="w-full rounded border border-gray-200"
                        />
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">After</p>
                        <img
                          src={`data:image/png;base64,${selected.screenshot_after}`}
                          alt="after"
                          className="w-full rounded border border-gray-200"
                        />
                      </div>
                    </div>
                  </>
                )}

                {/* ── General (HTTP) step ── */}
                {!isCliStep(selected) && !isBrowserStep(selected) && (() => {
                  const s = selected as GeneralStep;
                  return (
                    <>
                      <div className="bg-blue-50 rounded-lg p-4">
                        <p className="text-xs font-semibold text-blue-700 mb-2 uppercase tracking-wide">Action</p>
                        <p className="text-sm font-mono font-medium text-blue-900 mb-2">{s.action.endpoint}</p>
                        {s.action.reasoning && (
                          <p className="text-xs text-blue-700 italic mb-2 border-l-2 border-blue-300 pl-2">
                            {s.action.reasoning}
                          </p>
                        )}
                        <pre className="text-xs text-blue-800 whitespace-pre-wrap overflow-auto max-h-40 bg-white/60 rounded p-2">
                          {JSON.stringify(s.action.payload, null, 2)}
                        </pre>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">State Before</p>
                          <pre className="text-xs text-gray-700 whitespace-pre-wrap overflow-auto max-h-64 bg-gray-50 rounded p-3 border">
                            {JSON.stringify(s.state_before, null, 2)}
                          </pre>
                        </div>
                        <div>
                          <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">State After</p>
                          <pre className="text-xs text-gray-700 whitespace-pre-wrap overflow-auto max-h-64 bg-gray-50 rounded p-3 border">
                            {JSON.stringify(s.state_after, null, 2)}
                          </pre>
                        </div>
                      </div>
                      {s.state_hash_before !== s.state_hash_after && (
                        <p className="text-xs text-green-600 font-medium">✓ State changed this step</p>
                      )}
                      {s.state_hash_before === s.state_hash_after && (
                        <p className="text-xs text-gray-400">State unchanged this step</p>
                      )}
                    </>
                  );
                })()}
              </div>
            ) : (
              <div className="h-full flex items-center justify-center text-sm text-gray-400">
                Select a step to inspect
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Episode row
// ---------------------------------------------------------------------------

function EpisodeRow({
  envName,
  runId,
  ep,
  selected,
  onToggle,
}: {
  envName: string;
  runId: string;
  ep: AgentEpisode;
  selected: boolean;
  onToggle: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <tr
        className={`hover:bg-gray-50 ${ep.status === "completed" ? "cursor-pointer" : ""} ${selected ? "bg-blue-50" : ""}`}
        onClick={() => ep.status === "completed" && setOpen(true)}
      >
        <td className="px-3 py-3" onClick={(e) => e.stopPropagation()}>
          {ep.status === "completed" && (
            <input
              type="checkbox"
              checked={selected}
              onChange={() => onToggle(ep.id)}
              className="w-3.5 h-3.5 accent-blue-600 cursor-pointer"
            />
          )}
        </td>
        <td className="px-4 py-3 text-sm text-gray-700">{ep.episode_index + 1}</td>
        <td className="px-4 py-3">{badge(STATUS_BADGE, ep.status)}</td>
        <td className="px-4 py-3 text-sm text-gray-700">{ep.total_steps}</td>
        <td className="px-4 py-3 text-sm text-gray-700">{ep.total_reward.toFixed(3)}</td>
        <td className="px-4 py-3 w-32">{scoreBar(ep.final_objective_score)}</td>
        <td className="px-4 py-3">{badge(TERM_BADGE, ep.termination_reason, "—")}</td>
        <td className="px-4 py-3 text-xs text-gray-400">{fmt(ep.completed_at)}</td>
        <td className="px-4 py-3 text-xs text-blue-600">
          {ep.status === "completed" && "View →"}
        </td>
      </tr>
      {open && createPortal(
        <TrajectoryDrawer
          envName={envName}
          runId={runId}
          episode={ep}
          onClose={() => setOpen(false)}
        />,
        document.body
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Run card
// ---------------------------------------------------------------------------

function RunCard({
  run,
  envName,
  onRefresh,
  onSelectionChange,
}: {
  run: AgentRun;
  envName: string;
  onRefresh: () => void;
  onSelectionChange: (runId: string, episodeIds: string[]) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [episodes, setEpisodes] = useState<AgentEpisode[]>([]);
  const [loadingEp, setLoadingEp] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [selectedEps, setSelectedEps] = useState<Set<string>>(new Set());

  function toggleEp(id: string) {
    setSelectedEps(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      onSelectionChange(run.id, [...next]);
      return next;
    });
  }

  function toggleAll() {
    const completed = episodes.filter(e => e.status === "completed").map(e => e.id);
    const allSelected = completed.every(id => selectedEps.has(id));
    const next = allSelected ? new Set<string>() : new Set(completed);
    setSelectedEps(next);
    onSelectionChange(run.id, [...next]);
  }

  async function handleDelete(e: React.MouseEvent) {
    e.stopPropagation();
    if (!confirm(`Delete run ${run.id.slice(0, 8)} and all its trajectory files?`)) return;
    setDeleting(true);
    try {
      await fetch(`${API_BASE}/api/sandbox/${envName}/agent-runs/${run.id}`, { method: "DELETE" });
      onRefresh();
    } finally {
      setDeleting(false);
    }
  }

  const loadEpisodes = useCallback(async () => {
    setLoadingEp(true);
    const res = await fetch(`${API_BASE}/api/sandbox/${envName}/agent-runs/${run.id}/episodes`);
    const data = await res.json();
    setEpisodes(data);
    setLoadingEp(false);
  }, [envName, run.id]);

  useEffect(() => {
    if (expanded) loadEpisodes();
  }, [expanded, loadEpisodes]);

  // Auto-refresh while running
  useEffect(() => {
    if (run.status !== "running" && run.status !== "pending") return;
    const t = setInterval(() => { onRefresh(); if (expanded) loadEpisodes(); }, 4000);
    return () => clearInterval(t);
  }, [run.status, expanded, onRefresh, loadEpisodes]);

  const progress = run.num_episodes > 0 ? (run.episodes_completed / run.num_episodes) * 100 : 0;

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden mb-4 bg-white">
      {/* Header — div intentionally, not button, because it contains the delete button */}
      <div
        className="w-full text-left px-5 py-4 hover:bg-gray-50 transition-colors cursor-pointer"
        onClick={() => setExpanded(v => !v)}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              {badge(STATUS_BADGE, run.status)}
              <span className="text-xs text-gray-400 font-mono">{run.id.slice(0, 8)}</span>
              <span className="text-xs text-gray-400">· {run.agent_id}</span>
            </div>
            <p className="text-sm font-medium text-gray-900 leading-snug">{run.objective}</p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <div className="text-right text-xs text-gray-500">
              <div className="font-medium text-gray-700">{run.episodes_completed}/{run.num_episodes} episodes</div>
              <div>{fmt(run.created_at)}</div>
            </div>
            <button
              onClick={handleDelete}
              disabled={deleting}
              className="p-1.5 text-gray-300 hover:text-red-500 transition-colors disabled:opacity-40"
              title="Delete run"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        </div>

        {/* Progress bar */}
        {(run.status === "running" || run.status === "pending") && (
          <div className="mt-3 h-1.5 bg-gray-200 rounded overflow-hidden">
            <div
              className="h-full bg-blue-500 rounded transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}

        {run.error && (
          <p className="mt-2 text-xs text-red-600 font-mono truncate">{run.error}</p>
        )}
      </div>

      {/* Episode table */}
      {expanded && (
        <div className="border-t">
          <div className="flex items-center justify-between px-5 py-3 bg-gray-50 border-b">
            <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">Episodes</span>
            <div className="flex items-center gap-3">
              <a
                href={`${API_BASE}/api/sandbox/${envName}/agent-runs/${run.id}/export`}
                className="text-xs text-blue-600 hover:underline"
                onClick={e => e.stopPropagation()}
              >
                Download JSONL ↓
              </a>
              <button
                onClick={e => { e.stopPropagation(); loadEpisodes(); }}
                className="text-xs text-gray-500 hover:text-gray-700"
              >Refresh</button>
            </div>
          </div>
          {loadingEp ? (
            <div className="px-5 py-4 text-sm text-gray-400">Loading episodes…</div>
          ) : episodes.length === 0 ? (
            <div className="px-5 py-4 text-sm text-gray-400">No episodes yet</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b text-xs text-gray-500 bg-gray-50">
                    <th className="px-3 py-2">
                      <input
                        type="checkbox"
                        className="w-3.5 h-3.5 accent-blue-600 cursor-pointer"
                        checked={episodes.filter(e => e.status === "completed").length > 0 &&
                          episodes.filter(e => e.status === "completed").every(e => selectedEps.has(e.id))}
                        onChange={toggleAll}
                      />
                    </th>
                    <th className="px-4 py-2">#</th>
                    <th className="px-4 py-2">Status</th>
                    <th className="px-4 py-2">Steps</th>
                    <th className="px-4 py-2">Avg reward</th>
                    <th className="px-4 py-2">Final score</th>
                    <th className="px-4 py-2">Reason</th>
                    <th className="px-4 py-2">Completed</th>
                    <th className="px-4 py-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {episodes.map(ep => (
                    <EpisodeRow
                      key={ep.id}
                      envName={envName}
                      runId={run.id}
                      ep={ep}
                      selected={selectedEps.has(ep.id)}
                      onToggle={toggleEp}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Data Collection Panel
// ---------------------------------------------------------------------------

function DataCollectionPanel({
  envName,
  selection,  // { runId → [episodeId, ...] }
  onClear,
}: {
  envName: string;
  selection: Record<string, string[]>;
  onClear: () => void;
}) {
  const [prompt, setPrompt] = useState("");
  const [exporting, setExporting] = useState(false);
  const totalEps = Object.values(selection).reduce((s, ids) => s + ids.length, 0);

  async function handleExport() {
    if (!prompt.trim()) return;
    setExporting(true);
    try {
      const lines: string[] = [];
      // Metadata header line
      lines.push(JSON.stringify({
        type: "collection_metadata",
        prompt: prompt.trim(),
        env_name: envName,
        exported_at: new Date().toISOString(),
        episode_count: totalEps,
      }));

      // Fetch trajectories for every selected episode
      for (const [runId, epIds] of Object.entries(selection)) {
        for (const epId of epIds) {
          try {
            const res = await fetch(
              `${API_BASE}/api/sandbox/${envName}/agent-runs/${runId}/episodes/${epId}/trajectory`
            );
            if (!res.ok) continue;
            const data = await res.json();
            for (const step of data.steps ?? []) {
              lines.push(JSON.stringify({ type: "step", run_id: runId, episode_id: epId, ...step }));
            }
            if (data.summary) {
              lines.push(JSON.stringify({ type: "episode_summary", run_id: runId, episode_id: epId, ...data.summary }));
            }
          } catch { /* skip failed episodes */ }
        }
      }

      const blob = new Blob([lines.join("\n")], { type: "application/x-ndjson" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${envName}_collection_${Date.now()}.jsonl`;
      a.click();
      URL.revokeObjectURL(url);
      onClear();
    } finally {
      setExporting(false);
    }
  }

  if (totalEps === 0) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-40 border-t bg-white shadow-2xl">
      <div className="max-w-5xl mx-auto px-6 py-4">
        <div className="flex items-start gap-4">
          <div className="flex-1 space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-gray-900">Data Collection</span>
              <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs font-medium rounded-full">
                {totalEps} episode{totalEps !== 1 ? "s" : ""} selected
              </span>
              <button onClick={onClear} className="text-xs text-gray-400 hover:text-gray-600 ml-1">
                Clear
              </button>
            </div>
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm resize-none focus:ring-2 focus:ring-blue-500 focus:outline-none"
              rows={2}
              placeholder="Describe what this data should be used for (e.g. 'Train an agent to complete support tickets efficiently')"
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
            />
          </div>
          <div className="flex flex-col gap-2 shrink-0 pt-6">
            <button
              onClick={handleExport}
              disabled={exporting || !prompt.trim()}
              className="px-5 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 whitespace-nowrap"
            >
              {exporting ? "Exporting…" : "Export JSONL ↓"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Synthetic epoch card
// ---------------------------------------------------------------------------

interface ReplayStatus {
  active: boolean;
  objective?: string;
  num_episodes?: number;
  episodes?: { index: number; num_commands: number }[];
}

function SyntheticEpochCard({
  status,
  envName,
  onLaunch,
  launching,
}: {
  status: ReplayStatus;
  envName: string;
  onLaunch: (seedStart: number, numEpisodes: number) => void;
  launching: string | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const total = status.num_episodes ?? 0;

  return (
    <div className="border border-indigo-200 rounded-xl overflow-hidden">
      {/* Epoch header */}
      <div
        className="bg-indigo-50 px-5 py-4 flex items-center justify-between gap-4 cursor-pointer hover:bg-indigo-100/60 transition-colors"
        onClick={() => setExpanded((v) => !v)}
      >
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <span className="text-sm font-semibold text-indigo-900">Synthetic Epoch</span>
            <span className="px-1.5 py-0.5 bg-indigo-100 text-indigo-600 text-xs rounded font-medium">
              {total} episode{total !== 1 ? "s" : ""}
            </span>
          </div>
          <p className="text-xs text-indigo-600 font-mono truncate">{status.objective}</p>
        </div>
        <div className="flex items-center gap-3 shrink-0" onClick={(e) => e.stopPropagation()}>
          <button
            onClick={() => onLaunch(0, total)}
            disabled={launching !== null}
            className="px-3 py-1.5 text-xs font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
          >
            {launching === `0-${total}` ? "Launching…" : `Run all ${total} episodes →`}
          </button>
          <Link
            href={`/environments/${envName}/synthetic`}
            className="text-xs text-indigo-500 hover:text-indigo-700 underline"
          >
            Manage
          </Link>
          <span className="text-xs text-indigo-400">{expanded ? "▲" : "▼"}</span>
        </div>
      </div>

      {/* Episodes list */}
      {expanded && (
        <div className="divide-y divide-indigo-100">
          {status.episodes?.map((ep) => (
            <div key={ep.index} className="flex items-center justify-between px-5 py-3 hover:bg-indigo-50/40">
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-gray-700">Episode {ep.index + 1}</span>
                <span className="text-xs text-gray-400">{ep.num_commands} steps in trajectory</span>
              </div>
              <button
                onClick={() => onLaunch(ep.index, 1)}
                disabled={launching !== null}
                className="px-3 py-1.5 text-xs font-medium text-indigo-700 border border-indigo-300 rounded-lg hover:bg-indigo-50 disabled:opacity-50 transition-colors"
              >
                {launching === `${ep.index}-1` ? "Launching…" : "Run episode →"}
              </button>
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

export default function AgentRunsPage() {
  const params = useParams<{ env_name: string }>();
  const envName = params.env_name;

  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNew, setShowNew] = useState(false);
  const [newRunObjective, setNewRunObjective] = useState<string | undefined>(undefined);
  const [selection, setSelection] = useState<Record<string, string[]>>({});
  const [replayStatus, setReplayStatus] = useState<ReplayStatus | null>(null);

  const [launching, setLaunching] = useState<string | null>(null);

  const loadRuns = useCallback(async () => {
    try {
      const [runsRes, synthRes] = await Promise.all([
        fetch(`${API_BASE}/api/sandbox/${envName}/agent-runs`),
        fetch(`${API_BASE}/api/sandbox/${envName}/synthetic`, { cache: "no-store" }),
      ]);
      setRuns(await runsRes.json());
      if (synthRes.ok) setReplayStatus(await synthRes.json());
    } finally {
      setLoading(false);
    }
  }, [envName]);

  useEffect(() => { loadRuns(); }, [loadRuns]);

  async function launchReplayRun(seedStart: number, numEpisodes: number) {
    if (!replayStatus?.objective) return;
    const key = `${seedStart}-${numEpisodes}`;
    setLaunching(key);
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${envName}/agent-runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agent_id: "llm",
          objective: replayStatus.objective,
          num_episodes: numEpisodes,
          max_steps: 50,
          divergence_threshold: 0.2,
          dead_end_patience: 5,
          success_threshold: 0.9,
          seed_start: seedStart,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        alert(body.detail ?? `HTTP ${res.status}`);
        return;
      }
      await loadRuns();
    } finally {
      setLaunching(null);
    }
  }

  function openNewRun(prefillObjective?: string) {
    setNewRunObjective(prefillObjective);
    setShowNew(true);
  }

  function handleSelectionChange(runId: string, episodeIds: string[]) {
    setSelection(prev => ({ ...prev, [runId]: episodeIds }));
  }

  const totalSelected = Object.values(selection).reduce((s, ids) => s + ids.length, 0);

  return (
    <div className="min-h-screen bg-gray-50" style={{ paddingBottom: totalSelected > 0 ? "140px" : "0" }}>
      {/* Top bar */}
      <div className="bg-white border-b px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link
            href={`/environments/${envName}`}
            className="text-sm text-gray-500 hover:text-gray-700"
          >
            ← {envName}
          </Link>
          <span className="text-gray-300">/</span>
          <h1 className="text-sm font-semibold text-gray-900">Agent Runs</h1>
        </div>
        <button
          onClick={() => openNewRun()}
          className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
        >
          + New Run
        </button>
      </div>

      {/* Info banner */}
      <div className="bg-blue-50 border-b border-blue-100 px-6 py-3">
        <p className="text-sm text-blue-700">
          Agent runs record full step-level trajectories (state, action, reward, objective score) for policy training.
          Each episode stops automatically when the agent diverges, hits a dead end, succeeds, or reaches max steps.
        </p>
      </div>

      {/* Content */}
      <div className="max-w-5xl mx-auto px-6 py-6 space-y-6">

        {/* Synthetic epoch */}
        {replayStatus?.active && replayStatus.episodes && replayStatus.episodes.length > 0 && (
          <SyntheticEpochCard
            status={replayStatus}
            envName={envName}
            onLaunch={launchReplayRun}
            launching={launching}
          />
        )}

        {/* Runs list */}
        {loading ? (
          <div className="text-sm text-gray-400 py-10 text-center">Loading runs…</div>
        ) : runs.length === 0 ? (
          <div className="text-center py-16">
            <p className="text-gray-500 mb-4">No agent runs yet.</p>
            <button
              onClick={() => openNewRun()}
              className="px-5 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
            >
              Launch your first run
            </button>
          </div>
        ) : (
          runs.map(run => (
            <RunCard
              key={run.id}
              run={run}
              envName={envName}
              onRefresh={loadRuns}
              onSelectionChange={handleSelectionChange}
            />
          ))
        )}
      </div>

      {showNew && (
        <NewRunModal
          envName={envName}
          onClose={() => setShowNew(false)}
          onCreated={loadRuns}
          defaultObjective={newRunObjective}
          replayMode={replayStatus?.active && newRunObjective === replayStatus?.objective}
        />
      )}

      <DataCollectionPanel
        envName={envName}
        selection={selection}
        onClear={() => setSelection({})}
      />
    </div>
  );
}
