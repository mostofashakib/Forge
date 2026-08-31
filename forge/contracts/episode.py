"""Shared machinery for every environment-type episode runner.

CLI, browser, and container runners all need the same things: an episode
config with early-stop thresholds, a result that serializes steps + summary
to JSONL, and the success / dead-end / divergence termination logic. They
live here once so a new environment type only implements what is unique to
it (how to act and how to observe).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from forge.contracts.types import AgentAdapter
from forge.contracts.rollout import RolloutRecord


@dataclass(kw_only=True)
class BaseEpisodeConfig:
    objective: str
    max_steps: int = 30
    # Stop if objective score stays below this for `consecutive_below_threshold` steps
    divergence_threshold: float = 0.2
    consecutive_below_threshold: int = 3
    # Stop if progress marker (state hash or rounded score) is identical this many steps
    dead_end_patience: int = 5
    # Stop early with "success" if score reaches this
    success_threshold: float = 0.9


@dataclass(kw_only=True)
class BaseEpisodeResult:
    steps: list = field(default_factory=list)
    total_reward: float = 0.0
    final_objective_score: float = 0.0
    termination_reason: str = "unknown"
    # Verdicts a model issued during this episode. Observed, not inferred:
    # it is the count of scorer calls, so a step recorded without being
    # scored does not inflate it.
    llm_verdicts: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def _step_to_dict(self, step) -> dict:
        """One step as a JSON-serializable dict; override when steps aren't dicts."""
        return step

    def _step_dicts(self) -> list[dict]:
        """Steps as JSON-serializable dicts."""
        return [self._step_to_dict(step) for step in self.steps]

    def summary(self) -> dict:
        return {
            "type": "episode_summary",
            "total_steps": len(self.steps),
            "total_reward": self.total_reward,
            "final_objective_score": self.final_objective_score,
            "termination_reason": self.termination_reason,
            "llm_verdicts": self.llm_verdicts,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def to_jsonl(self) -> str:
        lines = [json.dumps(step) for step in self._step_dicts()]
        lines.append(json.dumps(self.summary()))
        return "\n".join(lines)

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl(), encoding="utf-8")

    def to_rollout_record(
        self,
        *,
        env_name: str = "",
        task_name: str = "",
        seed: int | None = None,
        prompt: str | None = None,
    ) -> RolloutRecord:
        """Convert any controller result to the collector/trainer contract."""
        step_dicts = self._step_dicts()
        rewards = [float(step.get("reward", 0.0)) for step in step_dicts]
        actions = [step.get("action") for step in step_dicts if step.get("action")]
        completion = "\n".join(json.dumps(action, sort_keys=True) for action in actions)
        passed = self.termination_reason == "success"
        if passed:
            outcome = "success"
        elif any("error" in step for step in step_dicts):
            outcome = "edge_case"
        elif self.total_reward > 0:
            outcome = "partial_success"
        else:
            outcome = "failure"
        last = step_dicts[-1] if step_dicts else {}
        return RolloutRecord(
            episode_id=str(getattr(self, "episode_id", "") or "episode"),
            env_name=env_name,
            task_name=task_name,
            prompt=prompt or f"Task: {task_name}\nEnvironment: {env_name}",
            completion=completion,
            seed=seed,
            total_reward=self.total_reward,
            per_step_rewards=rewards,
            passed=passed,
            outcome=outcome,
            steps=len(step_dicts),
            terminated=bool(last.get("terminated", passed)),
            truncated=bool(last.get("truncated", False)),
            invalid_actions=sum(1 for step in step_dicts if "error" in step),
        )


class TrajectoryWriter:
    """Appends step records to a JSONL file as they happen.

    Writing each step immediately (and flushing) means a run that crashes or is
    killed mid-episode still leaves a durable, replayable partial trace — unlike
    ``write_jsonl``, which persists the whole trajectory only once the episode
    finishes. The episode summary is appended on ``close()`` (including when the
    episode exits via an exception), so the file always ends with a summary line.
    """

    def __init__(self, path: Path, result: BaseEpisodeResult) -> None:
        self._result = result
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")
        self._closed = False

    def record(self, step) -> None:
        self._fh.write(json.dumps(self._result._step_to_dict(step)) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._fh.write(json.dumps(self._result.summary()) + "\n")
            self._fh.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class EpisodeController(ABC):
    """Drives the multi-turn loop and decides when to stop.

    Deliberately not a member of the Environment facade: a controller drives an
    environment from outside, and the same environment may be run by a trainer,
    a benchmark harness, or a parallel rollout worker.
    """

    @abstractmethod
    def run_episode(
        self,
        agent: AgentAdapter,
        *,
        episode_id: str | None = None,
        seed: int | None = None,
        jsonl_path: Path | None = None,
    ) -> BaseEpisodeResult:
        """Run one episode to termination and return its result.

        `seed` is accepted by every controller for a uniform call signature,
        even where the family has no seeding path and ignores it.
        """
