from forge.customization.hooks import (
    override_transition, verifier, reward, observation_transform, policy_rule,
    clear_registry, get_registry,
)


def test_override_transition_registers_function():
    clear_registry()
    @override_transition("my_action")
    def my_fn(state, action, ctx):
        return None
    assert "my_action" in get_registry()["transitions"]
    assert get_registry()["transitions"]["my_action"] is my_fn


def test_verifier_registers_function():
    clear_registry()
    @verifier("my_task")
    def my_verifier(state, traj, task):
        return None
    assert "my_task" in get_registry()["verifiers"]


def test_reward_registers_function():
    clear_registry()
    @reward("my_reward")
    def my_reward_fn(state, traj, vr, task=None):
        return None
    assert "my_reward" in get_registry()["rewards"]


def test_observation_transform_registers_function():
    clear_registry()
    @observation_transform("agent_view")
    def my_transform(state, actor):
        return state
    assert "agent_view" in get_registry()["observation_transforms"]


def test_policy_rule_registers_function():
    clear_registry()
    @policy_rule("no_refund_first")
    def my_rule(state, action, ctx):
        return True
    assert "no_refund_first" in get_registry()["policy_rules"]


def test_clear_registry_empties_all():
    @override_transition("x")
    def fn(s, a, c):
        pass
    clear_registry()
    assert all(len(v) == 0 for v in get_registry().values())


def test_decorators_return_original_function():
    clear_registry()
    @override_transition("z")
    def my_fn(s, a, c):
        return "hello"
    assert my_fn(None, None, None) == "hello"


def test_get_registry_returns_copy():
    clear_registry()
    @override_transition("w")
    def fn(s, a, c):
        pass
    reg = get_registry()
    reg["transitions"].clear()
    # Original registry unaffected
    assert "w" in get_registry()["transitions"]
