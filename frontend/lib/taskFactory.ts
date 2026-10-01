import { API_BASE } from "@/lib/api";

export type TargetEnvironment = {
  name: string;
  env_type: string;
  family: "state" | "cli" | "browser";
  ready: boolean;
  reason: string | null;
};

export type ModelView = { provider: string | null; model: string | null; family: string | null };

export type Settings = {
  writer: ModelView;
  task_validator: ModelView & { configured: boolean; error: string | null };
  providers: { name: string; needs_key: boolean; key_set: boolean }[];
  ollama: { models: OllamaModel[]; error: string | null };
};

export type OllamaModel = { name: string; family: string; parameters: string | null; cloud: boolean };

export type BatchSummary = {
  id: string;
  env_name: string;
  version: number | null;
  status: "queued" | "running" | "complete" | "short" | "failed";
  requested: number;
  delivered: number;
  shortfall: number | null;
  pass_k: number;
  writer_model: string;
  validator_model: string;
  error: string | null;
  created_at: string | null;
  completed_at: string | null;
};

export type GoldenStep = { tool: string; args: Record<string, unknown> };

export type Check = {
  kind: string;
  description?: string;
  collection?: string | null;
  match?: Record<string, unknown> | null;
  expect?: Record<string, unknown> | null;
  path?: string | null;
  op?: string | null;
  value?: unknown;
  tools?: string[] | null;
  ordered?: boolean;
  command?: string | null;
  selector?: string | null;
  prop?: string | null;
  attr?: string | null;
};

export type TaskSeed = {
  records: Record<string, Record<string, unknown>[]>;
  setup: string[];
  pages: { path: string; html: string }[];
  start_path: string | null;
};

export type ReviewVerdict = {
  realistic: boolean;
  realistic_reason: string;
  fair: boolean;
  fair_reason: string;
  sensible: boolean;
  sensible_reason: string;
};

export type SyntheticTask = {
  id: string;
  category: string;
  difficulty: number;
  title: string;
  objective: string;
  seed: TaskSeed;
  golden: GoldenStep[];
  checks: Check[];
  reflection_points: { step: number; kind: string; note: string }[];
  step_budget: number;
  round: number;
  review: ReviewVerdict;
  fingerprint: string;
};

export type TaskRejection = {
  slot: number;
  category: string;
  difficulty: number;
  round: number;
  stage: "writer" | "static" | "pass_k" | "review";
  reason: string;
  draft: { title: string } | null;
};

export type Taxonomy = {
  categories: { name: string; description: string; exercises: string[]; difficulties: number[] }[];
  difficulty_rubric: Record<string, string>;
};

export type BatchDetail = BatchSummary & {
  taxonomy: Taxonomy | null;
  tasks: SyntheticTask[];
  rejections: TaskRejection[];
};

/** Fetch JSON from the Forge API, turning a non-2xx answer into an Error with the API's detail. */
export async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      // Keep the status-only message when the body is not JSON.
    }
    throw new Error(detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export const STAGE_LABELS: Record<TaskRejection["stage"], string> = {
  writer: "Writer",
  static: "Static check",
  pass_k: "Golden pass^k",
  review: "LLM review",
};
