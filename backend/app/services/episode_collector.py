from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backend.app.models import AuditLog, Episode, EpisodeStep
from forge.runtime.snapshot import StepSnapshot


class EpisodeDataCollector:
    """Persists facts emitted by an episode run to the database and JSONL."""

    def __init__(
        self,
        episode_id: str,
        db_session,
        jsonl_path: Path | None = None,
    ) -> None:
        self._episode_id = episode_id
        self._db = db_session
        self._jsonl_path = jsonl_path

    def record_step(self, snapshot: StepSnapshot) -> None:
        self._db.add(EpisodeStep(
            episode_id=self._episode_id,
            step_index=snapshot.step_index,
            action=json.dumps(snapshot.action),
            reward=snapshot.reward,
            verifier_results=json.dumps(snapshot.verifier_results),
            diff=json.dumps(snapshot.diff),
            events=json.dumps(snapshot.events),
            state_hash_before=snapshot.state_hash_before,
            state_hash_after=snapshot.state_hash_after,
            terminated=snapshot.terminated,
            truncated=snapshot.truncated,
        ))
        self._db.commit()
        if self._jsonl_path is not None:
            with self._jsonl_path.open("a") as output:
                output.write(snapshot.model_dump_json() + "\n")

    def complete_episode(
        self, total_reward: float, passed: bool, total_steps: int
    ) -> None:
        episode = self._db.get(Episode, self._episode_id)
        if episode is None:
            return
        episode.status = "completed"
        episode.total_reward = total_reward
        episode.passed = passed
        episode.total_steps = total_steps
        episode.completed_at = datetime.now(timezone.utc)
        self._db.commit()

    def record_policy_violation(
        self,
        step_index: int,
        action_type: str,
        violations: list,
    ) -> None:
        for violation in violations:
            self._db.add(AuditLog(
                episode_id=self._episode_id,
                step_index=step_index,
                actor="agent",
                action_type=action_type,
                rule_id=violation.rule_id,
                violation=violation.description,
                severity=violation.severity,
                created_at=datetime.now(timezone.utc),
            ))
        self._db.commit()
