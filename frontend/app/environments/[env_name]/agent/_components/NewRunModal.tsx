"use client";

import { useState } from "react";
import { API_BASE } from "@/lib/api";
import PersonaRunToggle from "@/components/PersonaRunToggle";

export function NewRunModal({
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
              <label className="block text-sm font-medium text-gray-700 mb-1">Dead-end patience</label>
              <input
                type="number" min={1} max={100}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                value={deadEndPatience}
                onChange={e => setDeadEndPatience(Number(e.target.value))}
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

          <PersonaRunToggle envName={envName} />

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
