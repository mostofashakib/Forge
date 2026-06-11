# tests/runtime/test_determinism.py
import copy
import itertools
import pytest
from forge.runtime.determinism import DeterminismError, run_determinism_check
from forge.runtime.env import ForgeEnv
from forge.runtime.reward import RewardEngine
from forge.runtime.snapshot import EnvironmentSpec
from forge.runtime.transition import TransitionEngine, TransitionResult
from forge.runtime.verifier import VerifierEngine


class SeededStateFactory:
    def create(self, ctx, options):
        return {"counter": {"c_0": {"id": "c_0", "value": ctx.rng.randint(0, 1000)}}}


_nondeterministic_source = itertools.count()


class NonDeterministicStateFactory:
    def create(self, ctx, options):
        return {"counter": {"c_0": {"id": "c_0", "value": next(_nondeterministic_source)}}}


def increment_transition(state, action, ctx):
    new_state = copy.deepcopy(state)
    new_state["counter"]["c_0"]["value"] += 1
    return TransitionResult(state=new_state, events=[{"type": "incremented", "entity_id": "c_0"}])


def build_env(factory, max_steps: int = 10) -> ForgeEnv:
    spec = EnvironmentSpec(name="test_env", domain="test", max_steps=max_steps)
    te = TransitionEngine()
    te.register("increment", increment_transition)
    return ForgeEnv(
        env_spec=spec,
        initial_state_factory=factory,
        transition_engine=te,
        verifier_engine=VerifierEngine(),
        reward_engine=RewardEngine(),
    )


def test_deterministic_env_passes_check():
    env = build_env(SeededStateFactory())
    report = run_determinism_check(env, seed=42, num_steps=5)
    assert report.passed is True
    assert report.observation_hash
    assert report.seed == 42


def test_nondeterministic_env_raises():
    env = build_env(NonDeterministicStateFactory())
    with pytest.raises(DeterminismError) as exc_info:
        run_determinism_check(env, seed=42, num_steps=5)
    assert exc_info.value.first_hash != exc_info.value.second_hash


def test_replay_uses_recorded_actions():
    env = build_env(SeededStateFactory())
    report = run_determinism_check(env, seed=7, num_steps=3)
    assert len(report.actions) == 3
    assert all(a["type"] == "increment" for a in report.actions)


def test_check_stops_at_truncation():
    env = build_env(SeededStateFactory(), max_steps=2)
    report = run_determinism_check(env, seed=1, num_steps=10)
    assert len(report.actions) == 2
    assert report.passed is True


def test_explicit_action_sequence_is_replayed():
    env = build_env(SeededStateFactory())
    actions = [{"type": "increment"}, {"type": "increment"}]
    report = run_determinism_check(env, seed=3, num_steps=10, actions=actions)
    assert report.actions == actions
    assert report.passed is True


def test_gmail_env_passes_determinism_check():
    from examples.gmail_env.gym_wrapper import build_gmail_env

    env = build_gmail_env()
    report = run_determinism_check(env, seed=42, num_steps=5)
    assert report.passed is True


def test_nondeterministic_reward_fails_check():
    from forge.runtime.reward import RewardBreakdown, RewardComponent

    flaky = itertools.count()

    def flaky_reward(state, trajectory, verifier_results, task):
        value = float(next(flaky))
        return RewardBreakdown(
            total_reward=value, components=[RewardComponent(name="flaky", value=value)]
        )

    spec_env = build_env(SeededStateFactory())
    spec_env._reward_engine.set_default(flaky_reward)
    with pytest.raises(DeterminismError):
        run_determinism_check(spec_env, seed=42, num_steps=3)


def test_same_seed_same_trajectory_produces_same_score():
    actions = [{"type": "increment"}] * 4

    env = build_env(SeededStateFactory())
    report1 = run_determinism_check(env, seed=42, num_steps=4, actions=actions)
    report2 = run_determinism_check(env, seed=42, num_steps=4, actions=actions)
    assert report1.total_reward == report2.total_reward
    assert report1.observation_hash == report2.observation_hash
