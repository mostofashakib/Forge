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
  tagColor: string;
  description: string;
  usedWith: string;
}

const TRAINING_FORMATS: FormatDef[] = [
  {
    id: "sft_pairs",
    label: "SFT Pairs",
    filename: "sft_pairs.jsonl",
    tag: "JSONL",
    tagColor: "bg-blue-50 text-blue-700",
    description:
      "Instruction-following pairs from successful episodes. Each record has a messages array (user = task objective, assistant = command sequence).",
    usedWith: "TRL SFTTrainer · OpenAI fine-tuning · Axolotl",
  },
  {
    id: "preference_pairs",
    label: "Preference Pairs",
    filename: "preference_pairs.jsonl",
    tag: "JSONL",
    tagColor: "bg-blue-50 text-blue-700",
    description:
      "Chosen / rejected trajectory pairs ranked by total reward. Full command sequences included on both sides.",
    usedWith: "TRL DPOTrainer · LlamaFactory",
  },
  {
    id: "grpo_rollouts",
    label: "RL Trajectories",
    filename: "grpo_rollouts.parquet",
    tag: "Parquet",
    tagColor: "bg-violet-50 text-violet-700",
    description:
      "Full episode rollouts with prompt, completion, total reward, and per-step reward list. Each row is one episode ready for policy gradient training.",
    usedWith: "TRL GRPOTrainer · veRL · OpenRLHF",
  },
  {
    id: "failure_dataset",
    label: "Failure Dataset",
    filename: "failure_dataset.jsonl",
    tag: "JSONL",
    tagColor: "bg-blue-50 text-blue-700",
    description:
      "Step-by-step trajectories from failed episodes with per-step verifier diagnostics. Useful for studying failure modes and contrastive training.",
    usedWith: "Failure analysis · Contrastive training · Red-teaming",
  },
];

const RAW_FORMATS: FormatDef[] = [
  {
    id: "trajectories",
    label: "Raw Trajectories",
    filename: "trajectories.jsonl",
    tag: "JSONL",
    tagColor: "bg-slate-100 text-slate-600",
    description: "Full step-by-step trajectories for all completed episodes — actions, rewards, termination flags.",
    usedWith: "Custom pipelines",
  },
  {
    id: "rewards",
    label: "Rewards",
    filename: "rewards.jsonl",
    tag: "JSONL",
    tagColor: "bg-slate-100 text-slate-600",
    description: "Per-episode reward summaries with step-level reward components.",
    usedWith: "Analysis · Custom pipelines",
  },
  {
    id: "verifier_results",
    label: "Verifier Results",
    filename: "verifier_results.jsonl",
    tag: "JSONL",
    tagColor: "bg-slate-100 text-slate-600",
    description: "Detailed verifier output per step for all completed episodes.",
    usedWith: "Debugging · Custom reward models",
  },
];

const ALL_FORMATS = [...TRAINING_FORMATS, ...RAW_FORMATS];

// ---------------------------------------------------------------------------
// ExportJob type
// ---------------------------------------------------------------------------

interface ExportJob {
  id: string;
  status: "pending" | "running" | "completed" | "failed";
  error: string | null;
  formats: string[];
}

// ---------------------------------------------------------------------------
// Format card
// ---------------------------------------------------------------------------

function FormatCard({
  fmt,
  active,
  onToggle,
  downloadReady,
  jobId,
}: {
  fmt: FormatDef;
  active: boolean;
  onToggle: () => void;
  downloadReady: boolean;
  jobId: string | undefined;
}) {
  return (
    <div
      className={`rounded-xl border-2 transition-all duration-150 overflow-hidden bg-card ${
        active
          ? "border-primary/50 shadow-sm shadow-primary/10"
          : "border-border/60 hover:border-border"
      }`}
    >
      <button type="button" onClick={onToggle} className="w-full text-left px-4 py-4">
        <div className="flex items-start gap-3">
          {/* Checkbox */}
          <span
            className={`mt-0.5 w-4 h-4 rounded border-2 flex items-center justify-center shrink-0 transition-colors ${
              active ? "border-primary bg-primary" : "border-muted-foreground/40"
            }`}
          >
            {active && (
              <svg className="w-2.5 h-2.5 text-white" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 8 8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M1.5 4L3.5 6L6.5 2" />
              </svg>
            )}
          </span>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-1.5">
              <span className="text-sm font-semibold">{fmt.label}</span>
              <span className={`text-xs px-1.5 py-0.5 rounded font-mono font-medium ${fmt.tagColor}`}>
                {fmt.tag}
              </span>
              <span className="text-xs text-muted-foreground/60 font-mono">{fmt.filename}</span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">{fmt.description}</p>
            <p className="text-xs text-muted-foreground/50 mt-1.5 flex items-center gap-1">
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="5" cy="5" r="4" />
                <path d="M5 3v2.5l1.5 1" />
              </svg>
              {fmt.usedWith}
            </p>
          </div>
        </div>
      </button>

      {downloadReady && jobId && (
        <div className="border-t border-border/40 bg-green-50/60 px-4 py-2.5 flex items-center justify-between">
          <span className="flex items-center gap-1.5 text-xs text-green-700 font-medium">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="6" cy="6" r="4.5" />
              <path d="M4 6.5l1.5 1.5 3-3" />
            </svg>
            Ready to download
          </span>
          <a
            href={`${API_BASE}/api/exports/${jobId}/download/${fmt.filename}`}
            download={fmt.filename}
            className="text-xs font-medium text-primary hover:underline underline-offset-2 flex items-center gap-1"
          >
            <svg width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5.5 1v6M3.5 5.5L5.5 7.5L7.5 5.5" />
              <path d="M1.5 9h8" />
            </svg>
            {fmt.filename}
          </a>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section wrapper
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
      <h2 className="section-label mb-3">{title}</h2>
      <div className="space-y-2.5">
        {formats.map((fmt) => (
          <FormatCard
            key={fmt.id}
            fmt={fmt}
            active={selected.has(fmt.id)}
            onToggle={() => toggle(fmt.id)}
            downloadReady={!!(jobId && completedFormats.includes(fmt.id))}
            jobId={jobId}
          />
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

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
  const selectedCount = selected.size;

  return (
    <div className="space-y-7 max-w-2xl">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Export Dataset</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Package collected trajectories into training-ready formats.
          </p>
        </div>
        <Link href={`/environments/${envName}`} className="text-sm text-muted-foreground hover:text-foreground transition-colors shrink-0">
          ← {envName}
        </Link>
      </div>

      {/* Quick-select bar */}
      <div className="flex items-center gap-2 flex-wrap">
        <button
          type="button"
          onClick={() => setSelected(new Set(ALL_FORMATS.map((f) => f.id)))}
          className="text-xs px-3 py-1.5 rounded-lg border border-border hover:border-primary/40 hover:bg-primary/5 transition-all"
        >
          Select all
        </button>
        <button
          type="button"
          onClick={() => setSelected(new Set(TRAINING_FORMATS.map((f) => f.id)))}
          className="text-xs px-3 py-1.5 rounded-lg border border-border hover:border-primary/40 hover:bg-primary/5 transition-all"
        >
          Training only
        </button>
        <button
          type="button"
          onClick={() => setSelected(new Set())}
          className="text-xs px-3 py-1.5 rounded-lg border border-border hover:border-red-300 hover:bg-red-50 hover:text-red-600 transition-all"
        >
          Clear
        </button>
      </div>

      <Section
        title="Training Formats"
        formats={TRAINING_FORMATS}
        selected={selected}
        toggle={toggle}
        completedFormats={completedFormats}
        jobId={job?.id}
      />

      <Section
        title="Raw Data"
        formats={RAW_FORMATS}
        selected={selected}
        toggle={toggle}
        completedFormats={completedFormats}
        jobId={job?.id}
      />

      {/* Export action */}
      <div className="flex items-center gap-4 pt-1">
        <button
          onClick={runExport}
          disabled={exporting || selectedCount === 0}
          className="flex items-center gap-2 px-5 py-2.5 text-sm font-semibold bg-foreground text-background rounded-lg hover:opacity-90 disabled:opacity-50 transition-all shadow-sm hover:shadow-md"
        >
          {exporting ? (
            <>
              <span className="w-3.5 h-3.5 rounded-full border-2 border-background/30 border-t-background animate-spin" />
              Exporting…
            </>
          ) : (
            <>
              <svg width="13" height="13" viewBox="0 0 11 11" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5.5 1v6M3.5 5.5L5.5 7.5L7.5 5.5" />
                <path d="M1.5 9h8" />
              </svg>
              Export {selectedCount} format{selectedCount !== 1 ? "s" : ""}
            </>
          )}
        </button>

        {error && (
          <p className="text-xs text-red-600 bg-red-50 px-3 py-1.5 rounded-lg border border-red-200">{error}</p>
        )}
        {job?.status === "failed" && (
          <p className="text-xs text-red-600">Export failed: {job.error}</p>
        )}
        {job?.status === "completed" && (
          <span className="flex items-center gap-1.5 text-xs text-green-700 font-medium bg-green-50 px-3 py-1.5 rounded-lg border border-green-200">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="6" cy="6" r="4.5" />
              <path d="M4 6.5l1.5 1.5 3-3" />
            </svg>
            Export complete — download links above
          </span>
        )}
      </div>
    </div>
  );
}
