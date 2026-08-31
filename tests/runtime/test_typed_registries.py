"""Registries must reject non-conforming handlers at registration time."""
from __future__ import annotations

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
