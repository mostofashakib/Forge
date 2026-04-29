from __future__ import annotations
from dataclasses import dataclass, field
from forge.runtime.snapshot import StepSnapshot


@dataclass
class Trajectory:
    episode_id: str
    steps: list[StepSnapshot]

    @property
    def events(self) -> list[dict]:
        return [event for step in self.steps for event in step.events]

    @property
    def has_policy_violation(self) -> bool:
        return any(e.get("type") == "policy_violation" for e in self.events)

    @property
    def step_count(self) -> int:
        return len(self.steps)


class TrajectoryStore:
    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self._steps: list[StepSnapshot] = []

    def record(self, snapshot: StepSnapshot) -> None:
        self._steps.append(snapshot)

    def to_trajectory(self) -> Trajectory:
        return Trajectory(episode_id=self.episode_id, steps=list(self._steps))

    def to_jsonl(self) -> str:
        return "\n".join(step.model_dump_json() for step in self._steps)
