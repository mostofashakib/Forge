"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

// ---------------------------------------------------------------------------
// Format definitions
// ---------------------------------------------------------------------------

type FormatId =
  | "sft_pairs"
  | "preference_pairs"
  | "grpo_rollouts"
  | "failure_dataset"
  | "trajectories"
  | "rewards"
  | "verifier_results";

interface FormatDef {
  id: FormatId;
  label: string;
  filename: string;
  tag: string;
  description: string;
  usedWith: string;
}

const TRAINING_FORMATS: FormatDef[] = [
  {
    id: "sft_pairs",
    label: "SFT Pairs",
    filename: "sft_pairs.jsonl",
    tag: "JSONL",
    description:
      "Instruction-following pairs from successful episodes. Each record has a messages array (user = task objective, assistant = command sequence). Only passing episodes are included.",
    usedWith: "TRL SFTTrainer · OpenAI fine-tuning · Axolotl",
  },
  {
    id: "preference_pairs",
    label: "Preference Pairs",
    filename: "preference_pairs.jsonl",
    tag: "JSONL",
    description:
      "Chosen / rejected trajectory pairs ranked by total reward. Episodes with the same task and similar seed are compared; the higher-reward trajectory becomes chosen. Full command sequences are included in both sides.",
    usedWith: "TRL DPOTrainer · LlamaFactory",
  },
  {
    id: "grpo_rollouts",
    label: "RL Trajectories",
    filename: "grpo_rollouts.parquet",
    tag: "Parquet",
    description:
      "Full episode rollouts with prompt, completion, total reward, and per-step reward list. Each row is one episode ready for policy gradient training.",
    usedWith: "TRL GRPOTrainer · veRL · OpenRLHF",
  },
  {
    id: "failure_dataset",
    label: "Failure Dataset",
    filename: "failure_dataset.jsonl",
    tag: "JSONL",
    description:
      "Complete step-by-step trajectories from episodes that did not pass, including per-step verifier diagnostics. Useful for studying failure modes and contrastive or adversarial training.",
    usedWith: "Failure analysis · Contrastive training · Red-teaming",
  },
];

const RAW_FORMATS: FormatDef[] = [
  {
    id: "trajectories",
    label: "Raw Trajectories",
    filename: "trajectories.jsonl",
    tag: "JSONL",
    description: "Full step-by-step trajectories for all completed episodes — actions, rewards, termination flags.",
    usedWith: "Custom pipelines",
  },
  {
    id: "rewards",
    label: "Rewards",
    filename: "rewards.jsonl",
    tag: "JSONL",
    description: "Per-episode reward summaries with step-level reward components.",
    usedWith: "Analysis · Custom pipelines",
  },
  {
    id: "verifier_results",
    label: "Verifier Results",
    filename: "verifier_results.jsonl",
    tag: "JSONL",
    description: "Detailed verifier output per step for all completed episodes.",
    usedWith: "Debugging · Custom reward models",
  },
];

const ALL_FORMATS = [...TRAINING_FORMATS, ...RAW_FORMATS];
const FORMAT_MAP = Object.fromEntries(ALL_FORMATS.map((f) => [f.id, f]));

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

interface ExportJob {
  id: string;
  status: "pending" | "running" | "completed" | "failed";
  error: string | null;
  formats: string[];
}

export default function ExportPage() {
  const params = useParams<{ env_name: string }>();
  const envName = params.env_name;

  const [selected, setSelected] = useState<Set<FormatId>>(
    new Set(["sft_pairs", "preference_pairs", "grpo_rollouts", "failure_dataset"])
  );
  const [job, setJob] = useState<ExportJob | null>(null);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState("");

  function toggle(id: FormatId) {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  async function runExport() {
    setExporting(true);
    setError("");
    setJob(null);
    try {
      const res = await fetch(`${API_BASE}/api/exports/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ env_name: envName, formats: Array.from(selected) }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);

      const jobRes = await fetch(`${API_BASE}/api/exports/${data.export_job_id}`);
      if (jobRes.ok) setJob(await jobRes.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setExporting(false);
    }
  }

  const completedFormats = job?.status === "completed" ? job.formats : [];

  return (
    <div className="space-y-6 max-w-2xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Export Dataset</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Package collected trajectories into training-ready formats.
          </p>
        </div>
        <Link
          href={`/environments/${envName}`}
          className="text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          ← {envName}
        </Link>
      </div>

      {/* Training formats */}
      <Section
        title="Training Formats"
        formats={TRAINING_FORMATS}
        selected={selected}
        toggle={toggle}
        completedFormats={completedFormats}
        jobId={job?.id}
      />

      {/* Raw data formats */}
      <Section
        title="Raw Data"
        formats={RAW_FORMATS}
        selected={selected}
        toggle={toggle}
        completedFormats={completedFormats}
        jobId={job?.id}
      />

      {/* Export button */}
      <div className="flex items-center gap-4 pt-1">
        <button
          onClick={runExport}
          disabled={exporting || selected.size === 0}
          className="px-5 py-2 text-sm font-medium bg-foreground text-background rounded-lg hover:opacity-90 disabled:opacity-50 transition-opacity"
        >
          {exporting
            ? "Exporting…"
            : `Export ${selected.size} format${selected.size !== 1 ? "s" : ""}`}
        </button>
        {error && <p className="text-xs text-red-600">{error}</p>}
        {job?.status === "failed" && (
          <p className="text-xs text-red-600">Export failed: {job.error}</p>
        )}
        {job?.status === "completed" && (
          <p className="text-xs text-green-600 font-medium">✓ Export complete — download links above</p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section component
// ---------------------------------------------------------------------------

function Section({
  title,
  formats,
  selected,
  toggle,
  completedFormats,
  jobId,
}: {
  title: string;
  formats: FormatDef[];
  selected: Set<FormatId>;
  toggle: (id: FormatId) => void;
  completedFormats: string[];
  jobId: string | undefined;
}) {
  return (
    <div>
      <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-widest mb-3">
        {title}
      </h2>
      <div className="space-y-2">
        {formats.map((fmt) => {
          const active = selected.has(fmt.id);
          const downloadReady = jobId && completedFormats.includes(fmt.id);
          return (
            <div
              key={fmt.id}
              className={`border-2 rounded-lg transition-colors ${
                active ? "border-foreground bg-muted/10" : "border-border"
              }`}
            >
              <button
                type="button"
                onClick={() => toggle(fmt.id)}
                className="w-full text-left px-4 py-3"
              >
                <div className="flex items-start gap-3">
                  <span
                    className={`mt-0.5 w-3.5 h-3.5 rounded border-2 flex items-center justify-center shrink-0 transition-colors ${
                      active ? "border-foreground bg-foreground" : "border-muted-foreground"
                    }`}
                  >
                    {active && (
                      <svg className="w-2 h-2 text-background" fill="currentColor" viewBox="0 0 8 8">
                        <path
                          d="M1.5 4L3.5 6L6.5 2"
                          stroke="currentColor"
                          strokeWidth="1.5"
                          fill="none"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    )}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-medium">{fmt.label}</span>
                      <span className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-mono">
                        {fmt.tag}
                      </span>
                      <span className="text-xs text-muted-foreground font-mono">{fmt.filename}</span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">{fmt.description}</p>
                    <p className="text-xs text-muted-foreground/60 mt-0.5">
                      Use with: {fmt.usedWith}
                    </p>
                  </div>
                </div>
              </button>
              {downloadReady && (
                <div className="border-t px-4 py-2 flex items-center justify-between">
                  <span className="text-xs text-green-600 font-medium">✓ Ready</span>
                  <a
                    href={`${API_BASE}/api/exports/${jobId}/download/${fmt.filename}`}
                    download={fmt.filename}
                    className="text-xs text-foreground underline underline-offset-2 hover:opacity-70"
                  >
                    Download {fmt.filename}
                  </a>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
