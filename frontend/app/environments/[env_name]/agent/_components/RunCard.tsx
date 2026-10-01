"use client";

import { useEffect, useState, useCallback } from "react";
import { API_BASE } from "@/lib/api";
import { EpisodeRow } from "./EpisodeRow";
import { STATUS_BADGE, badge, fmt } from "./format";
import { AgentEpisode, AgentRun } from "./types";

export function RunCard({
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
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${envName}/agent-runs/${run.id}/episodes`);
      if (res.ok) setEpisodes(await res.json());
    } finally {
      setLoadingEp(false);
    }
  }, [envName, run.id]);

  useEffect(() => {
    if (!expanded) return;
    const timer = window.setTimeout(() => void loadEpisodes(), 0);
    return () => window.clearTimeout(timer);
  }, [expanded, loadEpisodes]);

  // Refresh this card's episodes while it runs. The page refreshes the run list.
  useEffect(() => {
    if (!expanded || (run.status !== "running" && run.status !== "pending")) return;
    const t = setInterval(() => void loadEpisodes(), 4000);
    return () => clearInterval(t);
  }, [run.status, expanded, loadEpisodes]);

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
