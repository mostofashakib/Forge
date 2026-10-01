"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";
import { TERM_BADGE, badge, scoreBar } from "./format";
import { AgentEpisode, GeneralStep, TrajectoryStep, isBrowserStep, isCliStep, stepLabel } from "./types";

export function TrajectoryDrawer({
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
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<TrajectoryStep | null>(null);

  useEffect(() => {
    // Aborting on change keeps a slow response from overwriting a newer episode.
    const controller = new AbortController();
    fetch(`${API_BASE}/api/sandbox/${envName}/agent-runs/${runId}/episodes/${episode.id}/trajectory`, {
      signal: controller.signal,
    })
      .then(r => {
        if (!r.ok) throw new Error(`Trajectory request failed (${r.status})`);
        return r.json();
      })
      .then(data => { setSteps(data.steps ?? []); setLoading(false); })
      .catch(cause => {
        if (controller.signal.aborted) return;
        setLoadError(cause instanceof Error ? cause.message : "Could not load the trajectory");
        setLoading(false);
      });
    return () => controller.abort();
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
            ) : loadError ? (
              <div role="alert" className="p-4 text-sm text-red-600">{loadError}</div>
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
