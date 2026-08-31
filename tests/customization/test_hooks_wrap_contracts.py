"""The decorator API stays plain-function; the registry stores contracts."""
from __future__ import annotations

import pytest

from forge.contracts import Rubric, Verifier
from forge.contracts.backend import TransitionHandler
from forge.customization.hooks import (
    clear_registry,
    get_registry,
    observation_transform,
    override_transition,
    policy_rule,
    reward,
    verifier,
)
from forge.runtime.transition import TransitionEngine, TransitionResult


def test_a_decorated_transition_is_stored_as_a_contract_instance():
    clear_registry()

    @override_transition("close_ticket")
    def _close(state, action, ctx):
        return TransitionResult(state={"closed": True}, events=[])

    stored = get_registry()["transitions"]["close_ticket"]
    assert isinstance(stored, TransitionHandler)


def test_a_decorated_transition_registers_without_a_type_error():
    # This is the integration the tightened registry would otherwise break.
    clear_registry()

    @override_transition("close_ticket")
    def _close(state, action, ctx):
        return TransitionResult(state={"closed": True}, events=[])

    engine = TransitionEngine()
    engine.register("close_ticket", get_registry()["transitions"]["close_ticket"])
    assert engine.apply({}, {"type": "close_ticket"}, None).state == {"closed": True}


def test_the_decorator_returns_the_original_function():
    # False-positive guard: decorating must not replace the author's function,
    # or the module's own callers would break.
    clear_registry()

    def _close(state, action, ctx):
        return TransitionResult(state={}, events=[])

    assert override_transition("close_ticket")(_close) is _close


def test_decorated_verifiers_and_rewards_are_also_wrapped():
    clear_registry()

    @verifier("task_a")
    def _v(state, trajectory, task):
        return None

    @reward("task_a")
    def _r(state, trajectory, verifier_results, task):
        return None

    assert isinstance(get_registry()["verifiers"]["task_a"], Verifier)
    assert isinstance(get_registry()["rewards"]["task_a"], Rubric)


def test_hooks_without_a_contract_are_left_as_plain_functions():
    # Negative: only the three contract-backed hooks wrap. Wrapping the others
    # would break their callers, which invoke them directly.
    clear_registry()

    @observation_transform("redact")
    def _obs(obs):
        return obs

    @policy_rule("no_send")
    def _rule(state, action):
        return True

    assert get_registry()["observation_transforms"]["redact"] is _obs
    assert get_registry()["policy_rules"]["no_send"] is _rule


def test_a_wrapped_transition_still_receives_a_plain_action_dict():
    # False-positive guard: the adapter converts the typed Action back to the
    # dict the author's function was written against.
    clear_registry()
    seen: dict = {}

    @override_transition("close_ticket")
    def _close(state, action, ctx):
        seen.update(action)
        return TransitionResult(state=state, events=[])

    engine = TransitionEngine()
    engine.register("close_ticket", get_registry()["transitions"]["close_ticket"])
    engine.apply({}, {"type": "close_ticket", "ticket_id": "t_1"}, None)
    assert seen == {"type": "close_ticket", "ticket_id": "t_1"}


def test_a_wrong_arity_hook_raises_at_decoration_not_mid_episode():
    # The user-visible payoff: an author who mis-declares a hook learns at
    # import time, not three steps into a rollout.
    clear_registry()

    with pytest.raises(TypeError, match="FunctionTransitionHandler"):

        @override_transition("close_ticket")
        def _close(state, action):  # missing ctx
            return TransitionResult(state={}, events=[])

    assert "close_ticket" not in get_registry()["transitions"]


def test_wrong_arity_verifier_and_reward_hooks_also_raise_at_decoration():
    clear_registry()

    with pytest.raises(TypeError, match="FunctionVerifier"):

        @verifier("task_a")
        def _v(state):
            return None

    with pytest.raises(TypeError, match="FunctionRubric"):

        @reward("task_a")
        def _r(state, trajectory, verifier_results, task, extra):
            return None


def test_a_correctly_declared_hook_is_still_accepted():
    # False-positive guard: the arity check must not reject the shape the
    # documented hooks API asks authors to write.
    clear_registry()

    @override_transition("close_ticket")
    def _close(state, action, ctx):
        return TransitionResult(state={"closed": True}, events=[])

    @verifier("task_a")
    def _v(state, trajectory, task):
        return None

    @reward("task_a")
    def _r(state, trajectory, verifier_results, task=None):
        return None

    registry = get_registry()
    assert isinstance(registry["transitions"]["close_ticket"], TransitionHandler)
    assert isinstance(registry["verifiers"]["task_a"], Verifier)
    assert isinstance(registry["rewards"]["task_a"], Rubric)
