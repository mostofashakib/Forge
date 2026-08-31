# tests/contracts/test_environment_facade.py
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from forge.contracts import (
    Action,
    ActionResult,
    Environment,
    ExecutionBackend,
    InitialStateProvider,
    Observation,
    ObservationEncoder,
    RewardBreakdown,
    RewardComponent,
    Rubric,
    StateManager,
    StepOutcome,
    Task,
    TaskSource,
    Termination,
    TerminationPolicy,
    ToolProvider,
    ToolSpec,
)


class _Tasks(TaskSource):
    def tasks(self) -> Sequence[Task]:
        return [Task(id="t1", objective="close the ticket")]

    def get(self, task_id: str) -> Task:
        return self.tasks()[0]


class _Initial(InitialStateProvider):
    def reset(self, ctx, *, seed: int | None, options: Mapping[str, object]) -> dict:
        return {}


class _Obs(ObservationEncoder):
    def encode(self, state: dict, ctx) -> Observation:
        return Observation(payload=state)


class _Backend(ExecutionBackend):
    def execute(self, action: Action, state: dict, ctx) -> ActionResult:
        return ActionResult(state=state)


class _State(StateManager):
    def get(self) -> dict:
        return {}

    def apply(self, state: dict) -> None:
        return None

    def hash(self) -> str:
        return "sha256:0"


class _Rubric(Rubric):
    def score(self, state, trajectory, verifier_results, task) -> RewardBreakdown:
        return RewardBreakdown(
            total_reward=0.0, components=[RewardComponent(name="none", value=0.0)]
        )


class _Termination(TerminationPolicy):
    def check(self, outcome: StepOutcome) -> Termination | None:
        return None


class _Headless(Environment):
    """An environment with no tools, prompt, or transport — a CLI-shaped one."""

    task_source = _Tasks()
    initial_state = _Initial()
    observations = _Obs()
    backend = _Backend()
    state = _State()
    rubric = _Rubric()
    termination = _Termination()


def test_an_environment_supplying_the_seven_required_members_instantiates():
    env = _Headless()
    assert env.task_source.get("t1").objective == "close the ticket"
    assert env.state.hash() == "sha256:0"


def test_optional_members_default_to_none():
    # False-positive guard: a shell environment has no tool schema and no wire.
    # It must not be forced to stub them.
    env = _Headless()
    assert env.prompt is None
    assert env.tools is None
    assert env.transport is None


def test_an_environment_missing_a_required_member_cannot_be_instantiated():
    # Negative: omitting a required concern fails at instantiation.
    class Incomplete(Environment):
        task_source = _Tasks()
        initial_state = _Initial()
        observations = _Obs()
        backend = _Backend()
        state = _State()
        rubric = _Rubric()
        # termination omitted

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_an_environment_may_supply_the_optional_members():
    class WithTools(_Headless):
        @property
        def tools(self) -> ToolProvider:
            class _Static(ToolProvider):
                def tools(self) -> Sequence[ToolSpec]:
                    return [ToolSpec(name="close_ticket")]

            return _Static()

    assert [t.name for t in WithTools().tools.tools()] == ["close_ticket"]


def test_episode_controller_is_not_part_of_the_facade():
    # A controller drives an environment from outside; folding it in would
    # imply every environment owns its own loop.
    assert not hasattr(_Headless(), "episode_controller")
