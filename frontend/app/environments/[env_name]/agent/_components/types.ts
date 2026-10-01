export interface AgentRun {
  id: string;
  env_name: string;
  agent_id: string;
  objective: string;
  num_episodes: number;
  max_steps: number;
  dead_end_patience: number;
  success_threshold: number;
  status: string;
  episodes_completed: number;
  error: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface AgentEpisode {
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
export interface GeneralStep {
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
export interface CliStep {
  step_index: number;
  command: string;
  stdout: string;
  stderr: string;
  exit_code: number;
  objective_score: number;
  reward: number;
}

// Browser step
export interface BrowserStep {
  step_index: number;
  action: { action_type: string; x?: number; y?: number; text?: string; key?: string; url?: string; delta_x?: number; delta_y?: number; reasoning?: string };
  screenshot_before: string;
  screenshot_after: string;
  url_before: string;
  url_after: string;
  objective_score: number;
  reward: number;
}

export type TrajectoryStep = GeneralStep | CliStep | BrowserStep;

export function isCliStep(s: TrajectoryStep): s is CliStep { return "command" in s; }
export function isBrowserStep(s: TrajectoryStep): s is BrowserStep { return "screenshot_before" in s; }

export function stepLabel(s: TrajectoryStep): string {
  if (isCliStep(s)) return s.command.slice(0, 40);
  if (isBrowserStep(s)) return s.action.action_type;
  return (s as GeneralStep).action.endpoint;
}
