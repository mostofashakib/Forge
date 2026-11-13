/** Calibration: how often a target model solves each task in a saved batch. */

export const MAX_ATTEMPTS = 5;
const PROVIDERS = ["ollama", "vllm", "openai", "anthropic"];

export type Band = "too_easy" | "frontier" | "unsolved";

export const BAND_LABELS: Record<Band, { label: string; desc: string }> = {
  too_easy: { label: "Too easy", desc: "Solved on every attempt" },
  frontier: { label: "Frontier", desc: "Solved sometimes" },
  unsolved: { label: "Unsolved", desc: "Never solved" },
};

export type CalibrationRun = {
  id: string;
  batch_id: string;
  target_model: string;
  attempts_per_task: number;
  status: "queued" | "running" | "complete" | "failed";
  error: string | null;
  created_at: string | null;
  completed_at: string | null;
};

export type FailureSummary = {
  attempts: number;
  failed: number;
  terminations: Record<string, number>;
  failed_checks: Record<string, number>;
  text: string;
};

export type TaskResult = {
  task_id: string;
  title: string;
  category: string;
  difficulty: number;
  edge_case: string | null;
  passed: number;
  attempts: number;
  solve_rate: number;
  band: Band;
  failure_summary: FailureSummary | null;
};

export type BandCounts = Record<Band, number>;

export type CalibrationDetail = CalibrationRun & {
  tasks: TaskResult[];
  summary: {
    tasks: number;
    bands: BandCounts;
    by_category: Record<string, BandCounts>;
    by_difficulty: Record<string, BandCounts>;
    histogram: Record<string, number>;
  };
};

export type AttemptView = {
  attempt_index: number;
  passed: boolean;
  termination: "submitted" | "budget" | "policy_error" | "seed_error";
  error: string;
  steps: { step: { tool: string; args: Record<string, unknown> }; ok: boolean; output: string; error: string }[];
  labels: { failed_checks: string[]; step_errors: number; off_golden_tools: string[] };
};

/** Why this text cannot name a target model, or null when it can. Mirrors the API's check. */
export function targetModelProblem(text: string): string | null {
  const value = text.trim();
  const colon = value.indexOf(":");
  const provider = colon === -1 ? "" : value.slice(0, colon);
  const model = colon === -1 ? "" : value.slice(colon + 1);
  if (!PROVIDERS.includes(provider) || !model) {
    return `Use <provider>:<model>, with provider one of ${PROVIDERS.join(", ")}.`;
  }
  return null;
}

/** Histogram rows from 0% to 100%, each with its share of the tasks. */
export function histogramRows(histogram: Record<string, number>): { label: string; count: number; share: number }[] {
  const total = Object.values(histogram).reduce((sum, n) => sum + n, 0);
  return Object.entries(histogram)
    .map(([rate, count]) => ({ rate: Number(rate), count }))
    .sort((a, b) => a.rate - b.rate)
    .map(({ rate, count }) => ({ label: `${Math.round(rate * 100)}%`, count, share: total ? count / total : 0 }));
}

export function isActive(status: CalibrationRun["status"]): boolean {
  return status === "queued" || status === "running";
}
