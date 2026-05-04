"""Evaluate endpoint — re-apply policy and reward requirements to past trajectories.

Reads policy_requirements / reward_requirements stored on the SandboxEnvironment,
lets the user update them, and runs an evaluation over a sample of recent
completed agent episodes using the configured scoring method (LLM or ML).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

_ENVS_ROOT = Path("generated_envs")
_VALID_SCORING_METHODS = ("llm", "embeddings", "rouge", "bleu")


def _reward_config_path(env_name: str) -> Path:
    return _ENVS_ROOT / env_name / "reward_config.json"


def _load_scoring_methods(env_name: str) -> list[str]:
    path = _reward_config_path(env_name)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Support both old single-string format and new list format.
            if "scoring_methods" in data:
                return data["scoring_methods"] or ["llm"]
            if "scoring_method" in data:
                return [data["scoring_method"]]
        except Exception:
            pass
    return ["llm"]


def _save_scoring_methods(env_name: str, methods: list[str]) -> None:
    path = _reward_config_path(env_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scoring_methods": methods}), encoding="utf-8")

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import AgentEpisode, AgentRun, SandboxEnvironment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sandbox", tags=["evaluate"])

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class UpdateRequirementsRequest(BaseModel):
    policy_requirements: str | None = None
    reward_requirements: str | None = None
    scoring_methods: list[str] | None = None


class EvaluationRunRequest(BaseModel):
    eval_type: Literal["policy", "reward"]
    # Optionally override stored requirements for a dry-run without saving.
    requirements: str | None = None
    # Override the stored scoring methods for a dry-run.
    scoring_methods: list[str] | None = None


# LLM output schemas (internal, Pydantic-validated)

class _PolicyViolation(BaseModel):
    episode_id: str
    step_index: int
    command: str = ""
    rule_violated: str
    severity: Literal["high", "medium", "low"] = "medium"


class _PolicyEvalResult(BaseModel):
    violations: list[_PolicyViolation] = Field(default_factory=list)
    summary: str


class _RewardReevaluation(BaseModel):
    episode_id: str
    new_score: float = Field(ge=0.0, le=1.0)
    delta: float = 0.0
    reasoning: str
    key_factors: list[str] = Field(default_factory=list)


class _RewardEvalResult(BaseModel):
    reevaluations: list[_RewardReevaluation] = Field(default_factory=list)
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_EPISODES = 5
_MAX_STEPS_PER_EP = 15


def _load_trajectory_summary(ep: AgentEpisode) -> list[dict]:
    """Return a compact step list (up to _MAX_STEPS_PER_EP) from the JSONL."""
    if not ep.jsonl_path or not Path(ep.jsonl_path).exists():
        return []
    try:
        lines = Path(ep.jsonl_path).read_text(encoding="utf-8").strip().splitlines()
        steps = []
        for line in lines:
            rec = json.loads(line)
            if rec.get("type") == "episode_summary":
                continue
            steps.append(rec)
        return steps[-_MAX_STEPS_PER_EP:]
    except Exception:
        return []


def _build_trajectory_text(episodes: list[tuple[AgentEpisode, list[dict]]]) -> str:
    """Format episode summaries for the LLM prompt."""
    parts: list[str] = []
    for ep, steps in episodes:
        header = (
            f"Episode {ep.id[:8]} "
            f"(steps={ep.total_steps}, reward={ep.total_reward:.3f}, "
            f"termination={ep.termination_reason or 'unknown'})"
        )
        step_lines: list[str] = []
        for s in steps:
            cmd = s.get("command", s.get("action", {}).get("endpoint", "?"))
            if isinstance(cmd, dict):
                cmd = json.dumps(cmd)
            exit_code = s.get("exit_code", "?")
            idx = s.get("step_index", "?")
            step_lines.append(f"  [{idx}] $ {str(cmd)[:100]}  (exit={exit_code})")
        parts.append(header + "\n" + "\n".join(step_lines))
    return "\n\n".join(parts)


_POLICY_SYSTEM = """\
You are a policy-compliance auditor for AI agent episodes.

Given policy requirements (natural-language rules) and a sample of agent trajectory steps,
identify concrete violations.

For each violation produce:
  - episode_id: first 8 chars of the episode UUID
  - step_index: integer index of the offending step
  - command: the command or action that violated the rule (max 120 chars)
  - rule_violated: short phrase naming the rule that was broken
  - severity: "high", "medium", or "low"

Be conservative — only flag clear, unambiguous violations.
Return an object { violations: [...], summary: "..." } via the extract tool.
"""

_REWARD_SYSTEM = """\
You are a reward-function engineer evaluating AI agent episodes against updated
reward requirements.

Given reward requirements (natural-language criteria) and episode summaries, produce a
re-evaluation score for each episode between 0.0 (terrible) and 1.0 (perfect).

For each episode produce:
  - episode_id: first 8 chars of the episode UUID
  - new_score: float 0.0–1.0 under the new requirements
  - delta: new_score minus the original reward (can be negative)
  - reasoning: one sentence explaining the score
  - key_factors: list of 1–3 bullet phrases (the dominant factors in this score)

Return an object { reevaluations: [...], summary: "..." } via the extract tool.
"""


def _run_policy_eval(
    requirements: str,
    episodes: list[tuple[AgentEpisode, list[dict]]],
) -> _PolicyEvalResult:
    from forge.extraction.llm_client import get_client as _get_client
    client = _get_client(max_tokens=1024)
    user = (
        f"Policy requirements:\n{requirements}\n\n"
        f"Agent trajectories:\n{_build_trajectory_text(episodes)}"
    )
    try:
        return client.extract(system=_POLICY_SYSTEM, user=user, schema=_PolicyEvalResult)
    except Exception as exc:
        logger.warning("[evaluate] policy LLM failed: %s", exc)
        return _PolicyEvalResult(violations=[], summary=str(exc))


def _run_reward_eval(
    requirements: str,
    episodes: list[tuple[AgentEpisode, list[dict]]],
) -> _RewardEvalResult:
    from forge.extraction.llm_client import get_client as _get_client
    client = _get_client(max_tokens=1024)
    user = (
        f"Reward requirements:\n{requirements}\n\n"
        f"Agent trajectories:\n{_build_trajectory_text(episodes)}"
    )
    try:
        return client.extract(system=_REWARD_SYSTEM, user=user, schema=_RewardEvalResult)
    except Exception as exc:
        logger.warning("[evaluate] reward LLM failed: %s", exc)
        return _RewardEvalResult(reevaluations=[], summary=str(exc))


def _traj_to_text(steps: list[dict]) -> str:
    lines = []
    for s in steps:
        cmd = s.get("command", s.get("action", {}).get("endpoint", ""))
        if isinstance(cmd, dict):
            import json as _j
            cmd = _j.dumps(cmd)
        lines.append(str(cmd)[:120])
    return "\n".join(lines)


def _run_reward_eval_ml(
    requirements: str,
    episodes: list[tuple[AgentEpisode, list[dict]]],
    method: str,
) -> _RewardEvalResult:
    """Score each episode against reward requirements using a single ML scorer."""
    from forge.envgen.ml_reward import build_scorer
    scorer = build_scorer(method)
    if scorer is None:
        return _RewardEvalResult(reevaluations=[], summary=f"Unknown scoring method: {method!r}")

    reevals: list[_RewardReevaluation] = []
    for ep, steps in episodes:
        candidate = _traj_to_text(steps)
        try:
            raw = scorer.score(requirements, candidate)
        except Exception as exc:
            logger.warning("[evaluate] ML score failed for ep %s: %s", ep.id[:8], exc)
            raw = 0.0
        delta = round(raw - (ep.total_reward or 0.0), 4)
        reevals.append(_RewardReevaluation(
            episode_id=ep.id[:8],
            new_score=round(raw, 4),
            delta=delta,
            reasoning=f"Similarity score ({method}) between requirements and trajectory.",
            key_factors=[f"{method} similarity = {raw:.3f}"],
        ))

    avg = sum(r.new_score for r in reevals) / len(reevals) if reevals else 0.0
    return _RewardEvalResult(
        reevaluations=reevals,
        summary=f"ML re-evaluation ({method}) across {len(reevals)} episodes. Avg score: {avg:.3f}.",
    )


def _run_reward_eval_multi(
    requirements: str,
    episodes: list[tuple[AgentEpisode, list[dict]]],
    methods: list[str],
) -> tuple[_RewardEvalResult, dict[str, list[float]]]:
    """Run all selected methods and return averaged scores plus per-method breakdown."""
    per_method: dict[str, list[float]] = {}

    for method in methods:
        if method == "llm":
            result = _run_reward_eval(requirements, episodes)
            per_method["llm"] = [r.new_score for r in result.reevaluations]
        else:
            result = _run_reward_eval_ml(requirements, episodes, method)
            per_method[method] = [r.new_score for r in result.reevaluations]

    n_eps = len(episodes)
    merged: list[_RewardReevaluation] = []
    for i, (ep, _) in enumerate(episodes):
        scores_for_ep = [per_method[m][i] for m in methods if i < len(per_method.get(m, []))]
        avg_score = sum(scores_for_ep) / len(scores_for_ep) if scores_for_ep else 0.0
        delta = round(avg_score - (ep.total_reward or 0.0), 4)
        factors = [f"{m}={per_method[m][i]:.3f}" for m in methods if i < len(per_method.get(m, []))]
        merged.append(_RewardReevaluation(
            episode_id=ep.id[:8],
            new_score=round(avg_score, 4),
            delta=delta,
            reasoning=f"Averaged across {len(methods)} scoring method(s).",
            key_factors=factors,
        ))

    overall_avg = sum(r.new_score for r in merged) / len(merged) if merged else 0.0
    summary = (
        f"Multi-method re-evaluation ({', '.join(methods)}) across {n_eps} episodes. "
        f"Avg score: {overall_avg:.3f}."
    )
    return _RewardEvalResult(reevaluations=merged, summary=summary), per_method


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/{env_name}/evaluate")
def get_evaluate(env_name: str, db: Session = Depends(get_db)):
    sb = db.get(SandboxEnvironment, env_name)
    if sb is None:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    return {
        "policy_requirements": sb.policy_requirements or "",
        "reward_requirements": sb.reward_requirements or "",
        "scoring_methods": _load_scoring_methods(env_name),
    }


@router.put("/{env_name}/evaluate", status_code=200)
def update_evaluate(
    env_name: str,
    body: UpdateRequirementsRequest,
    db: Session = Depends(get_db),
):
    sb = db.get(SandboxEnvironment, env_name)
    if sb is None:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    if body.policy_requirements is not None:
        sb.policy_requirements = body.policy_requirements or None
    if body.reward_requirements is not None:
        sb.reward_requirements = body.reward_requirements or None
    if body.scoring_methods is not None:
        invalid = [m for m in body.scoring_methods if m not in _VALID_SCORING_METHODS]
        if invalid:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid methods: {invalid}. Choose from: {list(_VALID_SCORING_METHODS)}",
            )
        if not body.scoring_methods:
            raise HTTPException(status_code=422, detail="At least one scoring method required.")
        _save_scoring_methods(env_name, body.scoring_methods)
    db.commit()
    return {"status": "saved"}


@router.post("/{env_name}/evaluate/run")
def run_evaluate(
    env_name: str,
    body: EvaluationRunRequest,
    db: Session = Depends(get_db),
):
    sb = db.get(SandboxEnvironment, env_name)
    if sb is None:
        raise HTTPException(status_code=404, detail="Sandbox not found")

    # Resolve requirements text (request body overrides stored value for dry-runs).
    if body.eval_type == "policy":
        requirements = body.requirements or sb.policy_requirements or ""
    else:
        requirements = body.requirements or sb.reward_requirements or ""

    if not requirements.strip():
        raise HTTPException(
            status_code=422,
            detail=f"No {body.eval_type} requirements configured. Save requirements first.",
        )

    # Load a sample of recent completed agent episodes.
    runs = (
        db.query(AgentRun)
        .filter(AgentRun.env_name == env_name)
        .order_by(AgentRun.created_at.desc())
        .limit(10)
        .all()
    )
    run_ids = [r.id for r in runs]

    eps = (
        db.query(AgentEpisode)
        .filter(AgentEpisode.run_id.in_(run_ids), AgentEpisode.status == "completed")
        .order_by(AgentEpisode.completed_at.desc())
        .limit(_MAX_EPISODES)
        .all()
    )

    if not eps:
        raise HTTPException(
            status_code=422,
            detail="No completed episodes found for this environment. Run agents first.",
        )

    episode_data = [(ep, _load_trajectory_summary(ep)) for ep in eps]

    if body.eval_type == "policy":
        result = _run_policy_eval(requirements, episode_data)
        return {
            "eval_type": "policy",
            "episodes_evaluated": len(eps),
            "violations": [v.model_dump() for v in result.violations],
            "summary": result.summary,
        }
    else:
        effective_methods = body.scoring_methods or _load_scoring_methods(env_name)
        result, per_method_scores = _run_reward_eval_multi(requirements, episode_data, effective_methods)
        return {
            "eval_type": "reward",
            "scoring_methods": effective_methods,
            "episodes_evaluated": len(eps),
            "reevaluations": [
                {**r.model_dump(), "original_reward": next(
                    (ep.total_reward for ep in eps if ep.id[:8] == r.episode_id),
                    None,
                )}
                for r in result.reevaluations
            ],
            "per_method_scores": per_method_scores,
            "summary": result.summary,
        }
