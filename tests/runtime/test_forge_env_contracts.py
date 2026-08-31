# tests/runtime/test_forge_env_contracts.py
from __future__ import annotations

from collections.abc import Mapping

from forge.contracts import (
    Action,
    Environment,
    InitialStateProvider,
    Observation,
    ObservationEncoder,
    ToolSpec,
)
from forge.contracts.backend import TransitionHandler
from forge.runtime.env import ForgeEnv
from forge.runtime.reward import RewardEngine
from forge.runtime.snapshot import EnvironmentSpec
from forge.runtime.transition import TransitionEngine, TransitionResult
from forge.runtime.verifier import VerifierEngine


class _Initial(InitialStateProvider):
    def reset(self, ctx, *, seed: int | None, options: Mapping[str, object]) -> dict:
        return {"tickets": [], "seed": seed}


class _Close(TransitionHandler):
    def apply(self, state: dict, action: Action, ctx) -> TransitionResult:
        return TransitionResult(state={**state, "closed": True}, events=[])


def _env(**kwargs) -> ForgeEnv:
    engine = TransitionEngine()
    engine.register("close_ticket", _Close())
    return ForgeEnv(
        env_spec=EnvironmentSpec(name="t", domain="support", max_steps=5),
        initial_state_provider=_Initial(),
        transition_engine=engine,
        verifier_engine=VerifierEngine(),
        reward_engine=RewardEngine(),
        **kwargs,
    )


def test_reset_threads_the_seed_to_the_provider():
    obs, info = _env().reset(seed=11)
    assert obs["seed"] == 11
    assert info["seed"] == 11


def test_an_unseeded_reset_still_produces_a_seed():
    # False-positive guard: gym requires a usable seed even when none is given,
    # so the provider must receive the derived one rather than None.
    obs, info = _env().reset()
    assert obs["seed"] == info["seed"]
    assert isinstance(info["seed"], int)


def test_step_applies_the_registered_handler():
    env = _env()
    env.reset(seed=1)
    obs, _reward, _term, _trunc, _info = env.step({"type": "close_ticket"})
    assert obs["closed"] is True


def test_initial_state_factory_remains_importable_as_an_alias():
    from forge.runtime.env import InitialStateFactory

    assert InitialStateFactory is InitialStateProvider


def test_forge_env_implements_the_composed_environment_facade():
    env = _env()

    assert isinstance(env, Environment)
    assert env.initial_state is env._initial_state
    assert env.backend.action_types == {"close_ticket"}
    assert env.rubric is env._reward_engine
    assert env.state.get() == {}
    assert env.task_source.tasks() == ()
    assert env.tools.tools() == ()
    assert env.termination is not None


class _RedactedObservation(ObservationEncoder):
    def encode(self, state: dict, ctx) -> Observation:
        return Observation(payload={"visible": state.get("closed", False)})


def test_all_observations_flow_through_the_injected_encoder():
    env = _env(observation_encoder=_RedactedObservation())

    initial, _ = env.reset(seed=1)
    changed, *_ = env.step({"type": "close_ticket"})

    assert initial == {"visible": False}
    assert changed == {"visible": True}


def test_default_task_and_tools_are_exposed_through_contracts():
    engine = TransitionEngine()
    engine.register("close_ticket", _Close())
    env = ForgeEnv(
        env_spec=EnvironmentSpec(
            name="support",
            domain="support",
            default_task={"id": "close", "objective": "Close the ticket"},
        ),
        initial_state_provider=_Initial(),
        transition_engine=engine,
        verifier_engine=VerifierEngine(),
        reward_engine=RewardEngine(),
        tool_specs=[ToolSpec(name="close_ticket", description="Close it")],
    )

    assert env.task_source.get("close").objective == "Close the ticket"
    assert env.tools.tools()[0].description == "Close it"
