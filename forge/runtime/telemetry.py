# forge/runtime/telemetry.py
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from backend.app.models import Episode, EpisodeStep
from forge.runtime.snapshot import StepSnapshot


class TelemetryClient:
    def __init__(
        self,
        episode_id: str,
        db_session,
        jsonl_path: "Path | None" = None,
    ) -> None:
        self._episode_id = episode_id
        self._db = db_session
        self._jsonl_path = jsonl_path

    def record_step(self, snapshot: StepSnapshot) -> None:
        step = EpisodeStep(
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
        )
        self._db.add(step)
        self._db.commit()
        if self._jsonl_path is not None:
            with open(self._jsonl_path, "a") as f:
                f.write(snapshot.model_dump_json() + "\n")

    def complete_episode(
        self, total_reward: float, passed: bool, total_steps: int
    ) -> None:
        ep = self._db.get(Episode, self._episode_id)
        if ep is None:
            return
        ep.status = "completed"
        ep.total_reward = total_reward
        ep.passed = passed
        ep.total_steps = total_steps
        ep.completed_at = datetime.now(timezone.utc)
        self._db.commit()

    def record_policy_violation(
        self,
        step_index: int,
        action_type: str,
        violations: list,
    ) -> None:
        from backend.app.models import AuditLog
        from datetime import datetime, timezone
        for v in violations:
            log = AuditLog(
                episode_id=self._episode_id,
                step_index=step_index,
                actor="agent",
                action_type=action_type,
                rule_id=v.rule_id,
                violation=v.description,
                severity=v.severity,
                created_at=datetime.now(timezone.utc),
            )
            self._db.add(log)
        self._db.commit()
