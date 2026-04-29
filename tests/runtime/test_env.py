# tests/runtime/test_env.py
import copy
import pytest
from forge.runtime.context import RuntimeContext
from forge.runtime.env import ForgeEnv
from forge.runtime.snapshot import EnvironmentSpec
from forge.runtime.transition import TransitionEngine, TransitionResult
from forge.runtime.verifier import VerifierEngine
from forge.runtime.reward import RewardEngine
from forge.runtime.verification import CheckResult, VerificationResult


class FixedStateFactory:
    def create(self, ctx: RuntimeContext, options: dict) -> dict:
        ctx.actor_id = "u_0000"
        return {"counter": {"c_0": {"id": "c_0", "value": 0}}}


def increment_transition(state, action, ctx):
    new_state = copy.deepcopy(state)
    new_state["counter"]["c_0"]["value"] += 1
    return TransitionResult(state=new_state, events=[{"type": "incremented", "entity_id": "c_0"}])


def check_counter_verifier(state, trajectory, task):
    passed = state["counter"]["c_0"]["value"] >= task["inputs"]["target"]
    return VerificationResult.from_checks(
        "check_counter",
        [CheckResult(name="counter_reached", passed=passed, score=1.0 if passed else 0.0)],
    )


def build_env(max_steps: int = 10) -> ForgeEnv:
    spec = EnvironmentSpec(name="test_env", domain="test", max_steps=max_steps)
    te = TransitionEngine()
    te.register("increment", increment_transition)
    ve = VerifierEngine()
    ve.register("check_counter", check_counter_verifier)
    re = RewardEngine()
    return ForgeEnv(
        env_spec=spec,
        initial_state_factory=FixedStateFactory(),
        transition_engine=te,
        verifier_engine=ve,
        reward_engine=re,
    )


def test_reset_returns_observation_and_info():
    env = build_env()
    obs, info = env.reset(seed=1)
    assert isinstance(obs, dict)
    assert "episode_id" in info
    assert "seed" in info


def test_step_returns_five_tuple():
    env = build_env()
    env.reset(seed=1)
    task = {"name": "reach_3", "verifier_id": "check_counter", "inputs": {"target": 3}}
    result = env.step({"type": "increment", "task": task})
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, dict)
    assert isinstance(reward, float)


def test_task_completion_terminates_episode():
    env = build_env()
    task = {"name": "reach_1", "verifier_id": "check_counter", "inputs": {"target": 1}}
    env.reset(seed=1, options={"task": task})
    _, _, terminated, _, _ = env.step({"type": "increment"})
    assert terminated is True


def test_invalid_action_does_not_mutate_state():
    env = build_env()
    env.reset(seed=1)
    obs_before, _ = env.reset(seed=1)
    obs_after, reward, _, _, info = env.step({"type": "nonexistent_action"})
    assert obs_after == obs_before
    assert reward == 0.0
    assert "error" in info


def test_step_before_reset_raises():
    env = build_env()
    with pytest.raises(RuntimeError, match="reset()"):
        env.step({"type": "increment"})


def test_truncation_at_max_steps():
    env = build_env(max_steps=2)
    task = {"name": "impossible", "verifier_id": "check_counter", "inputs": {"target": 999}}
    env.reset(seed=1, options={"task": task})
    env.step({"type": "increment"})
    _, _, _, truncated, _ = env.step({"type": "increment"})
    assert truncated is True


def test_same_seed_produces_same_initial_observation():
    env = build_env()
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert obs1 == obs2
