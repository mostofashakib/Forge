"use client";

import { useState } from "react";
import { API_BASE } from "@/lib/api";

export function DataCollectionPanel({
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
  const [exportError, setExportError] = useState("");
  const totalEps = Object.values(selection).reduce((s, ids) => s + ids.length, 0);

  async function handleExport() {
    if (!prompt.trim()) return;
    setExporting(true);
    setExportError("");
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

      // Fetch every selected trajectory in parallel, then write them in selection order.
      const selected = Object.entries(selection).flatMap(([runId, epIds]) =>
        epIds.map(epId => ({ runId, epId })),
      );
      const trajectories = await Promise.all(selected.map(async ({ runId, epId }) => {
        const res = await fetch(
          `${API_BASE}/api/sandbox/${envName}/agent-runs/${runId}/episodes/${epId}/trajectory`
        );
        if (!res.ok) throw new Error(`Trajectory request failed (${res.status})`);
        return { runId, epId, data: await res.json() };
      }));
      for (const { runId, epId, data } of trajectories) {
        for (const step of data.steps ?? []) {
          lines.push(JSON.stringify({ type: "step", run_id: runId, episode_id: epId, ...step }));
        }
        if (data.summary) {
          lines.push(JSON.stringify({ type: "episode_summary", run_id: runId, episode_id: epId, ...data.summary }));
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
    } catch (cause) {
      setExportError(cause instanceof Error ? cause.message : "Could not export selected episodes");
    } finally {
      setExporting(false);
    }
  }

  if (totalEps === 0) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-40 border-t bg-white shadow-2xl">
      <div className="app-width py-4">
        <div className="flex items-start gap-4">
          <div className="flex-1 space-y-2">
            {exportError && <p role="alert" className="text-xs text-red-600">Export failed: {exportError}</p>}
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
