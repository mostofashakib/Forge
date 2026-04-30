"use client";
import { useState } from "react";

import { API_BASE as API } from "@/lib/api";

const FORMATS = [
  { id: "trajectories", label: "Trajectories (JSONL)" },
  { id: "rewards", label: "Rewards (JSONL)" },
  { id: "verifier_results", label: "Verifier Results (JSONL)" },
  { id: "sft_pairs", label: "SFT Pairs (JSONL)" },
  { id: "preference_pairs", label: "Preference Pairs (JSONL)" },
  { id: "grpo_rollouts", label: "GRPO Rollouts (Parquet)" },
];

const FORMAT_FILES: Record<string, string> = {
  trajectories: "trajectories.jsonl",
  rewards: "rewards.jsonl",
  verifier_results: "verifier_results.jsonl",
  sft_pairs: "sft_pairs.jsonl",
  preference_pairs: "preference_pairs.jsonl",
  grpo_rollouts: "grpo_rollouts.parquet",
};

interface ExportJob {
  id: string;
  status: string;
  output_path: string | null;
  error: string | null;
}

interface ExportPanelProps {
  envName: string;
}

export default function ExportPanel({ envName }: ExportPanelProps) {
  const [selectedFormats, setSelectedFormats] = useState<string[]>(["trajectories", "rewards"]);
  const [exportJob, setExportJob] = useState<ExportJob | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleFormat = (fmt: string) => {
    setSelectedFormats((prev) =>
      prev.includes(fmt) ? prev.filter((f) => f !== fmt) : [...prev, fmt]
    );
  };

  const handleExport = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API}/api/exports/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ env_name: envName, formats: selectedFormats }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const { export_job_id } = await resp.json();
      const jobResp = await fetch(`${API}/api/exports/${export_job_id}`);
      if (jobResp.ok) setExportJob(await jobResp.json());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <h3 className="text-base font-semibold">Export Training Data</h3>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      <div className="space-y-2">
        {FORMATS.map((fmt) => (
          <label key={fmt.id} className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={selectedFormats.includes(fmt.id)}
              onChange={() => toggleFormat(fmt.id)}
              className="rounded"
            />
            {fmt.label}
          </label>
        ))}
      </div>
      <button
        onClick={handleExport}
        disabled={loading || selectedFormats.length === 0}
        className="bg-green-600 text-white px-4 py-2 rounded text-sm font-medium hover:bg-green-700 disabled:opacity-50"
      >
        {loading ? "Exporting..." : "Export"}
      </button>

      {exportJob && exportJob.status === "completed" && (
        <div className="mt-4 space-y-1">
          <p className="text-sm font-medium text-gray-700">Download:</p>
          {selectedFormats.map((fmt) => (
            <a
              key={fmt}
              href={`${API}/api/exports/${exportJob.id}/download/${FORMAT_FILES[fmt]}`}
              className="block text-sm text-blue-600 hover:underline"
              download
            >
              {FORMAT_FILES[fmt]}
            </a>
          ))}
        </div>
      )}

      {exportJob && exportJob.status === "failed" && (
        <p className="text-sm text-red-600">Export failed: {exportJob.error}</p>
      )}
    </div>
  );
}
