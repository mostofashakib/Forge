"""What a task adds to the task-tracker world.

The world itself is a task tracker and nothing else: projects, milestones,
tasks, dependencies, assignments, an audit log and a clock. It has no idea
which task it is hosting.

A scenario is the task-shaped half, and it is *data*. It says which things can
happen in this world (`events`), what work items appear when they do
(`latent_tasks`), and what the agent has to do for them to happen (`rules`).
The evaluator in `tracker` reads those rules generically, so adding a second
Harbor task means writing a second `Scenario`, not editing the world.

An empty `Scenario()` is legal and gives a plain, static task tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from task_sim.models import (
    LatentDependency,
    LatentTask,
    ScenarioEvent,
    ScenarioRule,
)


@dataclass(frozen=True, slots=True)
class Scenario:
    """What a task adds to the tracker world. Data, never behaviour.

    An empty `Scenario()` is legal and gives a plain, static task tracker.
    """

    events: tuple[ScenarioEvent, ...] = ()
    rules: tuple[ScenarioRule, ...] = ()
    latent_tasks: tuple[LatentTask, ...] = ()
    latent_dependencies: tuple[LatentDependency, ...] = ()
    #: Ids of events that carry no rows of their own. Recording that the agent
    #: got somewhere is often the point, so this is not an edge case.
    observation_only: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        declared = {event.event_id for event in self.events}
        for task in self.latent_tasks:
            if task.event_id not in declared:
                raise ValueError(
                    f"latent task {task.task_id} belongs to undeclared event {task.event_id}"
                )
        for dependency in self.latent_dependencies:
            if dependency.event_id not in declared:
                raise ValueError(
                    f"latent dependency {dependency.dep_id} belongs to undeclared "
                    f"event {dependency.event_id}"
                )
        for rule in self.rules:
            unknown = (
                {rule.event_id} | set(rule.requires_activated) | set(rule.requires_pending)
            ) - declared
            if unknown:
                raise ValueError(
                    f"rule {rule.rule_id} references unknown events: {sorted(unknown)}"
                )
        for event_id in self.observation_only:
            if event_id not in declared:
                raise ValueError(f"observation_only names undeclared event {event_id}")

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self.events)
