"""Registries must reject non-conforming handlers at registration time."""
from __future__ import annotations

import functools

import pytest

from forge.contracts import Action, RewardBreakdown, RewardComponent, Rubric, Verifier
from forge.contracts.backend import TransitionHandler
from forge.runtime.reward import FunctionRubric, RewardEngine
from forge.runtime.transition import (
    FunctionTransitionHandler,
    TransitionEngine,
    TransitionResult,
)
from forge.runtime.verifier import FunctionVerifier, VerifierEngine


class _Close(TransitionHandler):
    def apply(self, state: dict, action: Action, ctx) -> TransitionResult:
        return TransitionResult(state={**state, "closed": True}, events=[])


def test_a_conforming_handler_registers_and_applies():
    engine = TransitionEngine()
    engine.register("close_ticket", _Close())
    result = engine.apply({}, {"type": "close_ticket"}, None)
    assert result.state == {"closed": True}


def test_registering_a_bare_function_raises_at_registration():
    # This is the bug the contracts exist to prevent: a wrong-arity handler
    # used to be accepted here and blow up mid-episode instead.
    engine = TransitionEngine()
    with pytest.raises(TypeError, match="TransitionHandler"):
        engine.register("close_ticket", lambda state, action: state)


def test_a_plain_function_can_be_adapted_explicitly():
    # False-positive guard: wrapping stays available, so the customization
    # hooks API does not become less ergonomic.
    engine = TransitionEngine()
    engine.register(
        "close_ticket",
        FunctionTransitionHandler(
            lambda state, action, ctx: TransitionResult(state={"ok": True}, events=[])
        ),
    )
    assert engine.apply({}, {"type": "close_ticket"}, None).state == {"ok": True}


def test_verifier_engine_rejects_a_bare_function():
    with pytest.raises(TypeError, match="Verifier"):
        VerifierEngine().register("v1", lambda s, t, task: None)


def test_reward_engine_rejects_a_bare_function():
    with pytest.raises(TypeError, match="Rubric"):
        RewardEngine().register("t1", lambda *args: None)


def test_reward_engine_default_still_scores_from_verifier_results():
    # Behavior preservation: the documented fallback is unchanged.
    engine = RewardEngine()

    class _Passed:
        passed = True

    breakdown = engine.compute({}, None, [_Passed()], None)
    assert breakdown.total_reward == 1.0
    assert breakdown.components[0].name == "task_success"


def test_reward_engine_default_scores_zero_with_no_passing_verifier():
    # Negative: nothing passing must score zero, not a vacuous one.
    assert RewardEngine().compute({}, None, [], None).total_reward == 0.0


def test_reward_engine_set_default_rejects_a_bare_function():
    with pytest.raises(TypeError, match="Rubric"):
        RewardEngine().set_default(lambda *args: None)


def test_adapters_satisfy_their_contracts():
    # The Function* adapters are the sanctioned escape hatch: they must be
    # genuine contract instances, not merely callables that look right.
    assert isinstance(FunctionTransitionHandler(lambda s, a, c: None), TransitionHandler)
    assert isinstance(FunctionVerifier(lambda s, t, task: None), Verifier)
    assert isinstance(FunctionRubric(lambda *args: None), Rubric)


def test_a_wrapped_rubric_overrides_the_default_for_its_task():
    engine = RewardEngine()
    engine.register(
        "t1",
        FunctionRubric(
            lambda state, trajectory, verifier_results, task: RewardBreakdown(
                total_reward=0.5,
                components=[RewardComponent(name="custom", value=0.5)],
            )
        ),
    )

    class _Passed:
        passed = True

    scored = engine.compute({}, None, [_Passed()], {"name": "t1"})
    assert scored.total_reward == 0.5
    assert scored.components[0].name == "custom"
    # Negative: a task with no registered rubric still falls back to the default.
    assert engine.compute({}, None, [_Passed()], {"name": "other"}).total_reward == 1.0


# ---------------------------------------------------------------------------
# Arity: isinstance proves kind, not shape. The adapters are where a plain
# callable enters the typed world, so a wrong-signature function must be
# rejected there — at build time, not mid-rollout.
# ---------------------------------------------------------------------------

def test_a_wrong_arity_transition_function_is_rejected_when_wrapped():
    with pytest.raises(TypeError, match="FunctionTransitionHandler"):
        FunctionTransitionHandler(lambda state, action: state)


def test_a_wrong_arity_verifier_function_is_rejected_when_wrapped():
    with pytest.raises(TypeError, match="FunctionVerifier"):
        FunctionVerifier(lambda state: None)


def test_a_wrong_arity_rubric_function_is_rejected_when_wrapped():
    with pytest.raises(TypeError, match="FunctionRubric"):
        FunctionRubric(lambda state, trajectory, verifier_results: None)


def test_a_correct_arity_function_is_accepted_by_every_adapter():
    # False-positive guard: the arity check must not reject the ordinary shape.
    assert FunctionTransitionHandler(lambda state, action, ctx: None)
    assert FunctionVerifier(lambda state, trajectory, task: None)
    assert FunctionRubric(lambda state, trajectory, verifier_results, task: None)


def test_varargs_and_defaults_are_accepted():
    # False-positive guard: *args and trailing defaults are legitimate ways to
    # write a handler; a raw parameter count would wrongly reject both.
    assert FunctionTransitionHandler(lambda *args: None)
    assert FunctionVerifier(lambda state, trajectory, task=None: None)
    assert FunctionRubric(lambda state, trajectory, verifier_results, task=None: None)


def test_a_partial_supplying_leading_arguments_is_accepted():
    # False-positive guard: functools.partial is how authors bind config into a
    # handler. The bound arguments are gone from the signature, so what remains
    # must be what is checked.
    def _handler(config, state, action, ctx):
        return TransitionResult(state={"cfg": config}, events=[])

    handler = FunctionTransitionHandler(functools.partial(_handler, "cfg_a"))
    assert handler.apply({}, Action(type="close_ticket"), None).state == {"cfg": "cfg_a"}


def test_a_bound_method_is_accepted():
    # False-positive guard: `self` is already bound, so the visible arity is 3.
    class _Handlers:
        def close(self, state, action, ctx):
            return TransitionResult(state={"closed": True}, events=[])

    handler = FunctionTransitionHandler(_Handlers().close)
    assert handler.apply({}, Action(type="close_ticket"), None).state == {"closed": True}


def test_the_arity_error_names_the_expected_signature():
    with pytest.raises(TypeError) as exc_info:
        FunctionRubric(lambda state: None)
    message = str(exc_info.value)
    assert "state, trajectory, verifier_results, task" in message


def test_a_falsy_rubric_is_still_the_rubric_that_runs():
    # A Rubric is a user-supplied object and may define __bool__ — resolving it
    # by truthiness would silently discard it and score the default instead.
    class _FalsyRubric(Rubric):
        def __bool__(self) -> bool:
            return False

        def score(self, state, trajectory, verifier_results, task):
            return RewardBreakdown(
                total_reward=0.25,
                components=[RewardComponent(name="falsy", value=0.25)],
            )

    class _Passed:
        passed = True

    default_engine = RewardEngine()
    default_engine.set_default(_FalsyRubric())
    scored = default_engine.compute({}, None, [_Passed()], None)
    assert scored.total_reward == 0.25
    assert scored.components[0].name == "falsy"

    task_engine = RewardEngine()
    task_engine.register("t1", _FalsyRubric())
    assert task_engine.compute({}, None, [_Passed()], {"name": "t1"}).total_reward == 0.25
