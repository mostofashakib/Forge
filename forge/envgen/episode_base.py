"""Shared machinery for every environment-type episode runner.

CLI, browser, and container runners all need the same things: an episode
config with early-stop thresholds, a result that serializes steps + summary
to JSONL, and the success / dead-end / divergence termination logic. They
live here once so a new environment type only implements what is unique to
it (how to act and how to observe).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


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


class TerminationMonitor:
    """Early-stop decisions shared by every runner.

    Call observe() once per step with the objective score and an optional
    progress marker (a state hash for stateful envs; defaults to the rounded
    score). Returns "success", "dead_end", "diverged", or None to continue.
    """

    def __init__(self, config: BaseEpisodeConfig) -> None:
        self._cfg = config
        self._markers: list[object] = []
        self._below_threshold_count = 0

    def observe(self, score: float, marker: object = None) -> str | None:
        self._markers.append(marker if marker is not None else round(score, 2))

        if score >= self._cfg.success_threshold:
            return "success"

        if len(self._markers) >= self._cfg.dead_end_patience:
            recent = self._markers[-self._cfg.dead_end_patience:]
            if len(set(recent)) == 1:
                return "dead_end"

        if score < self._cfg.divergence_threshold:
            self._below_threshold_count += 1
        else:
            self._below_threshold_count = 0
        if self._below_threshold_count >= self._cfg.consecutive_below_threshold:
            return "diverged"
        return None
