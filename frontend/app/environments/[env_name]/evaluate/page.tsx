"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type EvalType = "policy" | "reward";

interface PolicyViolation {
  episode_id: string;
  step_index: number;
  command: string;
  rule_violated: string;
  severity: "high" | "medium" | "low";
}

interface RewardReevaluation {
  episode_id: string;
  new_score: number;
  original_reward: number | null;
  delta: number;
  reasoning: string;
  key_factors: string[];
}

interface PolicyResult {
  eval_type: "policy";
  episodes_evaluated: number;
  violations: PolicyViolation[];
  summary: string;
}

interface RewardResult {
  eval_type: "reward";
  episodes_evaluated: number;
  reevaluations: RewardReevaluation[];
  summary: string;
}

type EvalResult = PolicyResult | RewardResult;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SEVERITY_COLORS: Record<string, string> = {
  high:   "bg-red-100 text-red-700",
  medium: "bg-orange-100 text-orange-700",
  low:    "bg-gray-100 text-gray-500",
};

function deltaColor(d: number) {
  if (d > 0.05) return "text-green-600";
  if (d < -0.05) return "text-red-600";
  return "text-muted-foreground";
}

// ---------------------------------------------------------------------------
// Result panels
// ---------------------------------------------------------------------------

function PolicyResultPanel({ result }: { result: PolicyResult }) {
  return (
    <div className="space-y-4">
      <div className="border rounded-lg p-4 bg-muted/20">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-widest mb-1">
          Summary · {result.episodes_evaluated} episode{result.episodes_evaluated !== 1 ? "s" : ""} evaluated
        </p>
        <p className="text-sm">{result.summary}</p>
      </div>
      {result.violations.length === 0 ? (
        <div className="border rounded-lg p-8 text-center text-sm text-muted-foreground">
          No violations found.
        </div>
      ) : (
        <div className="border rounded-lg overflow-hidden">
          <div className="px-4 py-2.5 border-b bg-muted/40 text-xs text-muted-foreground">
            {result.violations.length} violation{result.violations.length !== 1 ? "s" : ""} found
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/20 text-xs text-muted-foreground">
                <th className="px-4 py-2 text-left">Episode</th>
                <th className="px-4 py-2 text-left">Step</th>
                <th className="px-4 py-2 text-left">Command</th>
                <th className="px-4 py-2 text-left">Rule violated</th>
                <th className="px-4 py-2 text-left">Severity</th>
              </tr>
            </thead>
            <tbody>
              {result.violations.map((v, i) => (
                <tr key={i} className="border-b last:border-0 hover:bg-muted/10">
                  <td className="px-4 py-3 font-mono text-xs text-muted-foreground">{v.episode_id}</td>
                  <td className="px-4 py-3 text-xs tabular-nums">{v.step_index}</td>
                  <td className="px-4 py-3 font-mono text-xs max-w-xs truncate" title={v.command}>{v.command || "—"}</td>
                  <td className="px-4 py-3 text-xs">{v.rule_violated}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${SEVERITY_COLORS[v.severity] ?? SEVERITY_COLORS.low}`}>
                      {v.severity}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function RewardResultPanel({ result }: { result: RewardResult }) {
  return (
    <div className="space-y-4">
      <div className="border rounded-lg p-4 bg-muted/20">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-widest mb-1">
          Summary · {result.episodes_evaluated} episode{result.episodes_evaluated !== 1 ? "s" : ""} evaluated
        </p>
        <p className="text-sm">{result.summary}</p>
      </div>
      {result.reevaluations.length === 0 ? (
        <div className="border rounded-lg p-8 text-center text-sm text-muted-foreground">
          No re-evaluations returned.
        </div>
      ) : (
        <div className="space-y-3">
          {result.reevaluations.map((r, i) => (
            <div key={i} className="border rounded-lg p-4">
              <div className="flex items-center justify-between gap-4 mb-2">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs text-muted-foreground">{r.episode_id}</span>
                  <span className="text-sm font-semibold">{r.new_score.toFixed(3)}</span>
                  {r.original_reward !== null && (
                    <span className={`text-xs font-medium ${deltaColor(r.delta)}`}>
                      {r.delta >= 0 ? "+" : ""}{r.delta.toFixed(3)} vs {r.original_reward.toFixed(3)}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <div className="w-24 h-1.5 bg-muted rounded overflow-hidden">
                    <div
                      className={`h-full rounded ${r.new_score >= 0.8 ? "bg-green-500" : r.new_score >= 0.5 ? "bg-yellow-400" : "bg-red-400"}`}
                      style={{ width: `${r.new_score * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-8 text-right">{Math.round(r.new_score * 100)}%</span>
                </div>
              </div>
              <p className="text-xs text-muted-foreground mb-2">{r.reasoning}</p>
              {r.key_factors.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {r.key_factors.map((f, j) => (
                    <span key={j} className="px-2 py-0.5 bg-muted rounded text-xs text-muted-foreground">{f}</span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function EvaluatePage() {
  const params = useParams<{ env_name: string }>();
  const envName = params.env_name;

  const [policyReqs, setPolicyReqs] = useState("");
  const [rewardReqs, setRewardReqs] = useState("");
  const [loadingReqs, setLoadingReqs] = useState(true);

  const [policyResult, setPolicyResult] = useState<EvalResult | null>(null);
  const [rewardResult, setRewardResult] = useState<EvalResult | null>(null);
  const [runningPolicy, setRunningPolicy] = useState(false);
  const [runningReward, setRunningReward] = useState(false);
  const [policyError, setPolicyError] = useState("");
  const [rewardError, setRewardError] = useState("");

  const loadReqs = useCallback(async () => {
    const data = await fetch(`${API_BASE}/api/sandbox/${envName}/evaluate`)
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
    setPolicyReqs(data?.policy_requirements ?? "");
    setRewardReqs(data?.reward_requirements ?? "");
    setLoadingReqs(false);
  }, [envName]);

  useEffect(() => { loadReqs(); }, [loadReqs]);

  async function runEval(type: EvalType) {
    const setRunning = type === "policy" ? setRunningPolicy : setRunningReward;
    const setError = type === "policy" ? setPolicyError : setRewardError;
    const setResult = type === "policy" ? setPolicyResult : setRewardResult;

    setRunning(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/sandbox/${envName}/evaluate/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ eval_type: type }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setResult(data as EvalResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation failed");
    } finally {
      setRunning(false);
    }
  }

  const hasPolicy = policyReqs.trim().length > 0;
  const hasReward = rewardReqs.trim().length > 0;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Evaluate</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Run the LLM against recent agent trajectories using your saved policy and reward requirements.
          </p>
        </div>
        <Link
          href={`/environments/${envName}`}
          className="text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          ← {envName}
        </Link>
      </div>

      {loadingReqs ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

          {/* Policy panel */}
          <div className="border rounded-lg p-5 space-y-4">
            <div className="flex items-start justify-between gap-2">
              <div>
                <h2 className="text-sm font-semibold">Policy Evaluation</h2>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Checks trajectories against your policy rules.
                </p>
              </div>
              <Link
                href={`/environments/${envName}/policy`}
                className="text-xs text-primary hover:underline whitespace-nowrap"
              >
                Edit rules →
              </Link>
            </div>

            {hasPolicy ? (
              <p className="text-xs text-muted-foreground border rounded p-2.5 bg-muted/20 line-clamp-3 font-mono">
                {policyReqs}
              </p>
            ) : (
              <p className="text-xs text-muted-foreground italic">
                No policy requirements set.{" "}
                <Link href={`/environments/${envName}/policy`} className="text-primary hover:underline">
                  Add rules →
                </Link>
              </p>
            )}

            <button
              onClick={() => runEval("policy")}
              disabled={runningPolicy || !hasPolicy}
              className="w-full py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {runningPolicy ? "Running…" : "Run Policy Evaluation"}
            </button>
            {policyError && <p className="text-xs text-red-600">{policyError}</p>}
            {policyResult && policyResult.eval_type === "policy" && (
              <PolicyResultPanel result={policyResult} />
            )}
          </div>

          {/* Reward panel */}
          <div className="border rounded-lg p-5 space-y-4">
            <div className="flex items-start justify-between gap-2">
              <div>
                <h2 className="text-sm font-semibold">Reward Evaluation</h2>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Re-scores trajectories against your reward criteria.
                </p>
              </div>
              <Link
                href={`/environments/${envName}/reward`}
                className="text-xs text-primary hover:underline whitespace-nowrap"
              >
                Edit criteria →
              </Link>
            </div>

            {hasReward ? (
              <p className="text-xs text-muted-foreground border rounded p-2.5 bg-muted/20 line-clamp-3 font-mono">
                {rewardReqs}
              </p>
            ) : (
              <p className="text-xs text-muted-foreground italic">
                No reward requirements set.{" "}
                <Link href={`/environments/${envName}/reward`} className="text-primary hover:underline">
                  Add criteria →
                </Link>
              </p>
            )}

            <button
              onClick={() => runEval("reward")}
              disabled={runningReward || !hasReward}
              className="w-full py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {runningReward ? "Running…" : "Run Reward Evaluation"}
            </button>
            {rewardError && <p className="text-xs text-red-600">{rewardError}</p>}
            {rewardResult && rewardResult.eval_type === "reward" && (
              <RewardResultPanel result={rewardResult} />
            )}
          </div>

        </div>
      )}
    </div>
  );
}
