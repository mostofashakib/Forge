# tests/runtime/test_forge_env_contracts.py
from __future__ import annotations

from collections.abc import Mapping

from forge.contracts import Action, InitialStateProvider
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


def _env() -> ForgeEnv:
    engine = TransitionEngine()
    engine.register("close_ticket", _Close())
    return ForgeEnv(
        env_spec=EnvironmentSpec(name="t", domain="support", max_steps=5),
        initial_state_provider=_Initial(),
        transition_engine=engine,
        verifier_engine=VerifierEngine(),
        reward_engine=RewardEngine(),
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
