"""Shared machinery for every environment-type episode runner.

CLI, browser, and container runners all need the same things: an episode
config with early-stop thresholds, a result that serializes steps + summary
to JSONL, and the success / dead-end / divergence termination logic. They
live here once so a new environment type only implements what is unique to
it (how to act and how to observe).
"""
from __future__ import annotations

# Moved to forge/contracts/episode.py. Re-exported here so existing imports
# keep working; prefer importing from forge.contracts.
from forge.contracts.episode import (  # noqa: F401
    BaseEpisodeConfig,
    BaseEpisodeResult,
    TrajectoryWriter,
)


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
