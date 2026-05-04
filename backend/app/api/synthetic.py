"""Synthetic trajectory generation and import.

POST /api/sandbox/{env_name}/synthetic/generate
  — LLM generates realistic CLI command sequences from a research goal.
    Supports difficulty scaling (1–5) and structured edge case injection.

POST /api/sandbox/{env_name}/synthetic/suggest-goals
  — LLM proposes novel research goals based on the environment context.

POST /api/sandbox/{env_name}/synthetic/import
  — Saves the command sequences as a replay manifest on disk.

GET  /api/sandbox/{env_name}/synthetic
  — Returns replay manifest status.

DELETE /api/sandbox/{env_name}/synthetic
  — Removes the replay manifest; agent runs revert to LLM-driven mode.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import SandboxEnvironment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sandbox", tags=["synthetic"])


def _replay_path(env_name: str) -> Path:
    envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
    return envs_root / env_name / "synthetic_replay.json"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

EDGE_CASE_TYPES = (
    "boundary_conditions",
    "permission_errors",
    "missing_deps",
    "conflicting_state",
    "recovery",
)

class GenerateRequest(BaseModel):
    research_goal: str
    num_episodes: int = Field(default=3, ge=1, le=10)
    quality: Literal["optimal", "diverse", "failure_cases"] = "diverse"
    # 1 = trivial single-step tasks, 5 = expert multi-component system tasks
    difficulty: int = Field(default=3, ge=1, le=5)
    # Edge case types to weave into trajectories (empty = no injection)
    edge_cases: list[str] = Field(default_factory=list)


class SuggestGoalsRequest(BaseModel):
    # Optional — what the agent already has, so the LLM avoids duplicates
    existing_goals: list[str] = Field(default_factory=list)
    # Align suggestions to the target difficulty
    difficulty: int = Field(default=3, ge=1, le=5)
    num_suggestions: int = Field(default=5, ge=2, le=8)


class ImportRequest(BaseModel):
    research_goal: str
    objective: str
    episodes: list[list[str]]
    difficulty: int = Field(default=3, ge=1, le=5)
    edge_cases: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM schemas
# ---------------------------------------------------------------------------

class _SyntheticStep(BaseModel):
    command: str = Field(description="Shell command the agent should execute.")


class _SyntheticEpisode(BaseModel):
    steps: list[_SyntheticStep] = Field(
        min_length=2,
        max_length=20,
        description="Sequence of shell commands for this episode.",
    )


class _GoalSuggestionsOut(BaseModel):
    goals: list[str] = Field(
        min_length=2,
        description="List of novel research goal strings, each a complete sentence.",
    )


class _ObjectiveOut(BaseModel):
    objective: str


# ---------------------------------------------------------------------------
# Difficulty level descriptors
# ---------------------------------------------------------------------------

_DIFFICULTY_HINTS = {
    1: (
        "TRIVIAL difficulty. Task should require 2–4 commands. Use only the most common "
        "shell utilities (ls, cat, echo, mkdir). No configuration, no multi-step setup. "
        "Clear, unambiguous success criteria."
    ),
    2: (
        "BEGINNER difficulty. Task should require 4–7 commands. Standard tools allowed "
        "(python3, pip, apt, git basics). Minimal error handling. Single-service or "
        "single-file goal."
    ),
    3: (
        "INTERMEDIATE difficulty. Task should require 7–12 commands. Requires installing "
        "dependencies, editing config files, and verifying results. May involve two "
        "components that must work together."
    ),
    4: (
        "ADVANCED difficulty. Task should require 12–16 commands. Involves system "
        "configuration, multi-step troubleshooting, multiple interacting services. "
        "Agents should need to debug at least one failure along the way."
    ),
    5: (
        "EXPERT difficulty. Task should require 15–20 commands. Complex system-level "
        "tasks: compiling from source, configuring daemons, managing multiple processes, "
        "diagnosing and recovering from cascading failures. High ambiguity."
    ),
}

_DIFFICULTY_LABELS = {1: "Trivial", 2: "Beginner", 3: "Intermediate", 4: "Advanced", 5: "Expert"}

# ---------------------------------------------------------------------------
# Edge case descriptors
# ---------------------------------------------------------------------------

_EDGE_CASE_DESCRIPTIONS = {
    "boundary_conditions": (
        "Include at least one step where the agent encounters a boundary condition: "
        "an empty file, a zero-length input, a maximum-length string, or an output "
        "that is exactly at an expected threshold."
    ),
    "permission_errors": (
        "Include at least one step where the agent tries an operation and gets a "
        "permission denied error, then must adapt (use sudo, change ownership, or "
        "pick a different path)."
    ),
    "missing_deps": (
        "Include at least one step where a command or package is not yet available "
        "and the agent must first install or configure the dependency before continuing."
    ),
    "conflicting_state": (
        "Include at least one step where the environment is already in a partially "
        "set-up state (a file already exists, a service is already running, a port "
        "is already bound) and the agent must detect and resolve the conflict."
    ),
    "recovery": (
        "Include at least one point mid-trajectory where a command fails (non-zero "
        "exit), and the agent must diagnose the failure and apply a corrective action "
        "before proceeding."
    ),
}

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_QUALITY_HINTS = {
    "optimal":       "Generate a near-perfect sequence where the agent succeeds efficiently.",
    "diverse":       "Generate a realistic sequence with both exploratory and direct commands.",
    "failure_cases": "Generate a sequence where the agent takes several wrong turns and struggles.",
}

_GENERATE_SYSTEM = """\
You generate synthetic CLI agent trajectory sequences for reinforcement-learning research.

Each episode is a realistic sequence of bash commands an AI agent might run inside an
Ubuntu container to accomplish a stated objective. Generate only the commands — do NOT
include expected outputs, scores, or rewards. Focus on making the command sequence
believable and purposeful.

Return EXACTLY one episode as a JSON object matching the schema. Call the extract tool.
"""

_GOAL_SYSTEM = """\
You propose novel research goals for training CLI agent trajectories in a given
reinforcement-learning environment. Each goal should be a single sentence that describes
a concrete, verifiable task an agent can accomplish in a Linux terminal.

Goals must be:
  - Distinct from any existing goals listed
  - Matched to the stated difficulty level
  - Appropriate for the environment's domain and policy constraints
  - Testable by running assertions in the container after completion

Return a JSON object {"goals": ["...", "...", ...]} via the extract tool.
"""

_OBJECTIVE_SYSTEM = """\
You distill a researcher's high-level research goal into a crisp one-sentence agent
objective suitable for a CLI episode runner (e.g. "Create a Flask web server that
listens on port 8080 and returns 'hello' at the root path.").

Call the extract tool with {"objective": "<one sentence>"}.
"""


def _derive_objective(research_goal: str) -> str:
    from forge.extraction.llm_client import AnthropicClient
    client = AnthropicClient(model="claude-haiku-4-5-20251001", max_tokens=256)
    try:
        result = client.extract(
            system=_OBJECTIVE_SYSTEM,
            user=f"Research goal: {research_goal}",
            schema=_ObjectiveOut,
        )
        return result.objective
    except Exception:
        return research_goal


def _build_episode_user_prompt(
    objective: str,
    env_type: str,
    episode_index: int,
    quality: str,
    difficulty: int,
    edge_cases: list[str],
) -> str:
    lines = [
        f"Research objective: {objective}",
        f"Environment type: {env_type}",
        f"Episode index: {episode_index}",
        f"Quality hint: {_QUALITY_HINTS.get(quality, _QUALITY_HINTS['diverse'])}",
        f"Difficulty: {_DIFFICULTY_HINTS[difficulty]}",
    ]
    if edge_cases:
        lines.append("\nEDGE CASE REQUIREMENTS — you MUST incorporate all of the following:")
        for ec in edge_cases:
            desc = _EDGE_CASE_DESCRIPTIONS.get(ec)
            if desc:
                lines.append(f"  • {ec.replace('_', ' ').title()}: {desc}")
    lines.append("\nGenerate one realistic command sequence for this objective.")
    return "\n".join(lines)


def _generate_one_episode(
    objective: str,
    quality: str,
    episode_index: int,
    env_type: str,
    difficulty: int = 3,
    edge_cases: list[str] | None = None,
) -> _SyntheticEpisode | None:
    from forge.extraction.llm_client import AnthropicClient
    client = AnthropicClient(model="claude-haiku-4-5-20251001", max_tokens=1024)
    user = _build_episode_user_prompt(
        objective, env_type, episode_index, quality, difficulty, edge_cases or []
    )
    try:
        return client.extract(system=_GENERATE_SYSTEM, user=user, schema=_SyntheticEpisode)
    except Exception as exc:
        logger.warning("[synthetic] generation failed for episode %d: %s", episode_index, exc)
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/{env_name}/synthetic/suggest-goals")
def suggest_goals(
    env_name: str,
    body: SuggestGoalsRequest,
    db: Session = Depends(get_db),
):
    """Propose novel research goals for this environment."""
    sb = db.get(SandboxEnvironment, env_name)
    if sb is None:
        raise HTTPException(status_code=404, detail="Sandbox not found")

    from forge.extraction.llm_client import AnthropicClient
    client = AnthropicClient(model="claude-haiku-4-5-20251001", max_tokens=512)

    difficulty_label = _DIFFICULTY_LABELS[body.difficulty]
    user_parts = [
        f"Environment name: {env_name}",
        f"Environment type: {sb.env_type or 'cli'}",
        f"Target difficulty: {difficulty_label} ({body.difficulty}/5) — {_DIFFICULTY_HINTS[body.difficulty]}",
        f"Number of goals to suggest: {body.num_suggestions}",
    ]
    if sb.policy_requirements:
        user_parts.append(f"Policy constraints: {sb.policy_requirements[:300]}")
    if sb.reward_requirements:
        user_parts.append(f"Reward criteria: {sb.reward_requirements[:300]}")
    if body.existing_goals:
        user_parts.append(
            "Existing goals (do NOT suggest these or close variants):\n"
            + "\n".join(f"  - {g}" for g in body.existing_goals)
        )

    try:
        result = client.extract(
            system=_GOAL_SYSTEM,
            user="\n".join(user_parts),
            schema=_GoalSuggestionsOut,
        )
        return {"goals": result.goals, "difficulty": body.difficulty}
    except Exception as exc:
        logger.warning("[synthetic] goal suggestion failed: %s", exc)
        raise HTTPException(status_code=500, detail="LLM failed to suggest goals.")


@router.post("/{env_name}/synthetic/generate")
def generate_synthetic(
    env_name: str,
    body: GenerateRequest,
    db: Session = Depends(get_db),
):
    sb = db.get(SandboxEnvironment, env_name)
    if sb is None:
        raise HTTPException(status_code=404, detail="Sandbox not found")

    # Validate edge case types.
    invalid_ec = [ec for ec in body.edge_cases if ec not in EDGE_CASE_TYPES]
    if invalid_ec:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown edge case types: {invalid_ec}. Valid: {list(EDGE_CASE_TYPES)}",
        )

    objective = _derive_objective(body.research_goal)
    env_type = sb.env_type or "cli"

    episodes: list[list[str]] = []
    for i in range(body.num_episodes):
        ep = _generate_one_episode(
            objective, body.quality, i, env_type,
            difficulty=body.difficulty, edge_cases=body.edge_cases,
        )
        if ep is not None:
            episodes.append([s.command for s in ep.steps])

    if not episodes:
        raise HTTPException(status_code=500, detail="LLM failed to generate any episodes.")

    return {
        "objective": objective,
        "episodes": episodes,
        "difficulty": body.difficulty,
        "difficulty_label": _DIFFICULTY_LABELS[body.difficulty],
        "edge_cases": body.edge_cases,
    }


@router.post("/{env_name}/synthetic/import")
def import_synthetic(
    env_name: str,
    body: ImportRequest,
    db: Session = Depends(get_db),
):
    sb = db.get(SandboxEnvironment, env_name)
    if sb is None:
        raise HTTPException(status_code=404, detail="Sandbox not found")

    if not body.episodes:
        raise HTTPException(status_code=422, detail="No episodes to import.")

    manifest = {
        "objective": body.objective,
        "research_goal": body.research_goal,
        "episodes": body.episodes,
        "difficulty": body.difficulty,
        "difficulty_label": _DIFFICULTY_LABELS.get(body.difficulty, "Intermediate"),
        "edge_cases": body.edge_cases,
    }
    path = _replay_path(env_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("[synthetic] saved replay manifest for %s: %d episodes", env_name, len(body.episodes))
    return {"episodes_imported": len(body.episodes), "objective": body.objective}


@router.get("/{env_name}/synthetic")
def get_synthetic(env_name: str, db: Session = Depends(get_db)):
    sb = db.get(SandboxEnvironment, env_name)
    if sb is None:
        raise HTTPException(status_code=404, detail="Sandbox not found")

    path = _replay_path(env_name)
    if not path.exists():
        return {"active": False}

    manifest = json.loads(path.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes", [])
    return {
        "active": True,
        "objective": manifest.get("objective"),
        "num_episodes": len(episodes),
        "difficulty": manifest.get("difficulty", 3),
        "difficulty_label": manifest.get("difficulty_label", "Intermediate"),
        "edge_cases": manifest.get("edge_cases", []),
        "episodes": [{"index": i, "num_commands": len(ep)} for i, ep in enumerate(episodes)],
    }


@router.delete("/{env_name}/synthetic", status_code=204)
def clear_synthetic(env_name: str, db: Session = Depends(get_db)):
    sb = db.get(SandboxEnvironment, env_name)
    if sb is None:
        raise HTTPException(status_code=404, detail="Sandbox not found")

    path = _replay_path(env_name)
    if path.exists():
        path.unlink()
    logger.info("[synthetic] cleared replay manifest for %s", env_name)
