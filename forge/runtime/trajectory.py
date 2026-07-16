from __future__ import annotations
from dataclasses import dataclass
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

    def to_trajectory_with_events(self, current_events: list[dict]) -> Trajectory:
        """Return a trajectory that also includes *current_events* from the in-flight step.

        This is used so verifiers can see events emitted by the current action
        before the step snapshot has been formally recorded.
        """
        traj = self.to_trajectory()
        if current_events:
            traj = _TrajectoryWithExtraEvents(traj, current_events)
        return traj

    def to_jsonl(self) -> str:
        return "\n".join(step.model_dump_json() for step in self._steps)


class _TrajectoryWithExtraEvents:
    """Lightweight wrapper that appends extra events to a trajectory's event list."""

    def __init__(self, base: Trajectory, extra_events: list[dict]) -> None:
        self.episode_id = base.episode_id
        self.steps = base.steps
        self._extra_events = extra_events

    @property
    def events(self) -> list[dict]:
        return self.steps.__class__([e for step in self.steps for e in step.events]) + self._extra_events

    @property
    def has_policy_violation(self) -> bool:
        return any(e.get("type") == "policy_violation" for e in self.events)

    @property
    def step_count(self) -> int:
        return len(self.steps)
