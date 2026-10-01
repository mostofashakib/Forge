"""A checkpoint restores everything the next steps depend on: state, RNG, clock, counters and personas."""
from __future__ import annotations

import copy
import random

from forge.runtime.env import ForgeEnv
from forge.runtime.reliability import SnapshotManager
from forge.runtime.reward import RewardEngine
from forge.runtime.snapshot import EnvironmentSpec
from forge.runtime.transition import FunctionTransitionHandler, TransitionEngine, TransitionResult
from forge.runtime.verifier import VerifierEngine


class _State:
    def reset(self, ctx, *, seed, options):
        return {"draws": [], "times": []}


def _draw(state, action, ctx):
    # Depends on the context RNG, the clock and the ID generator, so a
    # restore that misses any of them changes what the next step produces.
    new_state = copy.deepcopy(state)
    new_state["draws"].append([ctx.rng.random(), ctx.id_generator.next("d")])
    ctx.clock.advance(5)
    new_state["times"].append(ctx.clock.now().isoformat())
    return TransitionResult(state=new_state, events=[])


def _env() -> ForgeEnv:
    engine = TransitionEngine()
    engine.register("draw", FunctionTransitionHandler(_draw))
    return ForgeEnv(
        env_spec=EnvironmentSpec(name="ckpt", domain="test", max_steps=20),
        initial_state_provider=_State(),
        transition_engine=engine,
        verifier_engine=VerifierEngine(),
        reward_engine=RewardEngine(),
        deterministic=True,
    )


def _steps(env: ForgeEnv, count: int) -> list:
    outcomes = []
    for _ in range(count):
        obs, reward, terminated, truncated, _info = env.step({"action_type": "draw", "params": {}})
        outcomes.append((obs, reward, terminated, truncated, float(env.np_random.random())))
    return outcomes


def test_restoring_a_checkpoint_replays_the_same_future():
    env = _env()
    env.reset(seed=3)
    _steps(env, 2)
    checkpoint = env.checkpoint()

    first = _steps(env, 3)
    env.restore(checkpoint)
    second = _steps(env, 3)

    assert first == second
    assert len(env.current_trajectory().steps) == 5


def test_a_checkpoint_is_unaffected_by_later_steps_and_restores_twice():
    env = _env()
    env.reset(seed=3)
    checkpoint = env.checkpoint()

    _steps(env, 2)
    env.restore(checkpoint)
    assert env.state.get() == {"draws": [], "times": []}
    _steps(env, 2)
    env.restore(checkpoint)
    assert env.state.get() == {"draws": [], "times": []}
    assert env.current_trajectory().steps == []


def test_persona_engine_checkpoint_restores_rng_schedule_and_transcript():
    from forge.contracts.persona import PersonaPopulation
    from forge.personas.engine import PersonaEngine

    engine = PersonaEngine(PersonaPopulation())
    engine.reset(seed=1)
    checkpoint = engine.checkpoint()
    expected = random.Random()
    expected.setstate(checkpoint.rng_state)

    engine._rng.random()
    engine._schedule_state.action_count["x"] = 3
    engine.restore(checkpoint)

    assert engine._rng.random() == expected.random()
    assert engine._schedule_state.action_count == {}
    assert engine.transcript == []


def test_in_process_snapshot_holds_the_environment_checkpoint():
    env = _env()
    env.reset(seed=3)
    _steps(env, 1)
    manager = SnapshotManager(env_type="in_process", snapshot_interval=1)

    record = manager.capture_snapshot(step=1, state_hash="h", in_process_env=env)
    future = _steps(env, 2)
    env.restore(record.data)

    assert _steps(env, 2) == future


def test_checkpoint_before_reset_is_rejected():
    import pytest

    from forge.runtime.errors import ResetRequiredError

    with pytest.raises(ResetRequiredError):
        _env().checkpoint()


def test_a_checkpoint_does_not_pin_the_future_of_another_seed():
    # False-positive guard: replay only matches because the RNG was restored,
    # not because every episode draws the same numbers.
    env = _env()
    env.reset(seed=3)
    checkpoint = env.checkpoint()
    restored_future = (env.restore(checkpoint), _steps(env, 2))[1]

    env.reset(seed=4)
    assert _steps(env, 2) != restored_future
