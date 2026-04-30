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


def test_invalid_action_count_tracked():
    env = build_env()
    env.reset(seed=1)
    env.step({"type": "nonexistent_action"})
    env.step({"type": "another_nonexistent"})
    assert env._invalid_action_count == 2


def test_invalid_action_count_resets_on_reset():
    env = build_env()
    env.reset(seed=1)
    env.step({"type": "nonexistent_action"})
    assert env._invalid_action_count == 1
    env.reset(seed=2)
    assert env._invalid_action_count == 0


def test_invalid_action_count_in_task_dict_passed_to_reward():
    received_task = {}

    def capturing_reward(state, trajectory, verifier_results, task=None):
        received_task.update(task or {})
        from forge.runtime.reward import RewardBreakdown, RewardComponent
        return RewardBreakdown(total_reward=0.0, components=[])

    env = build_env()
    env._reward_engine.set_default(capturing_reward)
    env.reset(seed=1)
    env.step({"type": "nonexistent_action"})  # invalid — increments count
    env.step({"type": "increment"})           # valid step — reward is called
    assert received_task.get("invalid_action_count") == 1


# --- Appended for M5 telemetry tests ---

class MockTelemetry:
    def __init__(self):
        self.steps = []
        self.completions = []

    def record_step(self, snapshot):
        self.steps.append(snapshot)

    def complete_episode(self, total_reward, passed, total_steps):
        self.completions.append({"total_reward": total_reward, "passed": passed, "total_steps": total_steps})


def build_env_with_telemetry(telemetry, max_steps: int = 10) -> ForgeEnv:
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
        telemetry=telemetry,
    )


def test_forgeenv_telemetry_defaults_to_none():
    env = build_env()
    assert env._telemetry is None


def test_forgeenv_action_types_returns_registered_types():
    env = build_env()
    assert "increment" in env.action_types


def test_forgeenv_telemetry_records_step_on_valid_action():
    telemetry = MockTelemetry()
    env = build_env_with_telemetry(telemetry)
    env.reset(seed=1)
    env.step({"type": "increment"})
    assert len(telemetry.steps) == 1
    assert telemetry.steps[0].step_index == 0
    assert telemetry.steps[0].action == {"type": "increment"}


def test_forgeenv_telemetry_records_step_on_invalid_action():
    telemetry = MockTelemetry()
    env = build_env_with_telemetry(telemetry)
    env.reset(seed=1)
    env.step({"type": "nonexistent_action"})
    assert len(telemetry.steps) == 1


def test_forgeenv_telemetry_calls_complete_episode_on_truncation():
    telemetry = MockTelemetry()
    env = build_env_with_telemetry(telemetry, max_steps=1)
    env.reset(seed=1)
    _, reward, terminated, truncated, _ = env.step({"type": "increment"})
    assert truncated is True
    assert len(telemetry.completions) == 1
    assert telemetry.completions[0]["passed"] is False
    assert telemetry.completions[0]["total_steps"] == 1


def test_forgeenv_telemetry_none_does_not_raise():
    env = build_env()  # no telemetry
    env.reset(seed=1)
    env.step({"type": "increment"})  # must not raise


# --- M7: PolicyEngine + ObservationFilter integration ---
from forge.runtime.policy_engine import PolicyEngine
from forge.runtime.observation_filter import ObservationFilter, RBACConfig, RolePermissions
from forge.extraction.schemas import PolicyRule


def _build_env_m7(policy_engine=None, observation_filter=None, max_steps=10):
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
        policy_engine=policy_engine,
        observation_filter=observation_filter,
    )


def test_policy_violation_returns_unchanged_state():
    rule = PolicyRule(
        id="no_increment",
        condition="True",
        forbidden_actions=["increment"],
        description="high: always forbidden",
    )
    env = _build_env_m7(policy_engine=PolicyEngine([rule]))
    obs, _ = env.reset(seed=42)
    state_before = dict(obs)
    obs2, reward, terminated, truncated, info = env.step({"type": "increment"})
    assert obs2 == state_before
    assert reward == 0.0
    assert terminated is False
    assert "policy_violations" in info
    assert len(info["policy_violations"]) == 1
    assert info["policy_violations"][0]["rule_id"] == "no_increment"


def test_no_policy_engine_behaves_normally():
    env = _build_env_m7()  # no policy_engine
    env.reset(seed=42)
    obs2, reward, terminated, truncated, info = env.step({"type": "increment"})
    assert "policy_violations" not in info


def test_observation_filter_applied_to_reset():
    config = RBACConfig(roles={"agent": RolePermissions(cannot_see=["counter"])})
    obs_filter = ObservationFilter(rbac_config=config, role="agent")
    env = _build_env_m7(observation_filter=obs_filter)
    obs, _ = env.reset(seed=42)
    assert "counter" not in obs


def test_observation_filter_applied_to_step():
    config = RBACConfig(roles={"agent": RolePermissions(cannot_see=["counter"])})
    obs_filter = ObservationFilter(rbac_config=config, role="agent")
    env = _build_env_m7(observation_filter=obs_filter)
    env.reset(seed=42)
    obs2, _, _, _, _ = env.step({"type": "increment"})
    assert "counter" not in obs2


def test_policy_violation_increments_step_count():
    rule = PolicyRule(
        id="no_increment",
        condition="True",
        forbidden_actions=["increment"],
        description="high: forbidden",
    )
    env = _build_env_m7(policy_engine=PolicyEngine([rule]))
    env.reset(seed=42)
    assert env._step_count == 0
    env.step({"type": "increment"})
    assert env._step_count == 1
