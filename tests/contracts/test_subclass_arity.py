"""Subclass method arity is validated at class-definition time.

`@abstractmethod` only checks that a method name exists, never its signature.
A generated `class CloseTicketHandler(TransitionHandler): def apply(self, state)`
satisfies that check, registers cleanly through the strict `isinstance` checks
in the engines, and then fails mid-rollout when the engine calls it with three
arguments. `__init_subclass__` on the three ABCs that engines register moves
that failure to class-definition time (i.e. import time for generated
packages), which is earlier and stricter than registration.
"""
from __future__ import annotations

import pytest

from forge.contracts.backend import TransitionHandler
from forge.contracts.reward import Rubric, Verifier


# --- Negatives: wrong arity is rejected at class-definition time -----------


def test_transition_handler_wrong_arity_apply_raises_at_class_definition():
    with pytest.raises(TypeError) as exc_info:

        class BadHandler(TransitionHandler):
            def apply(self, state):
                return state

    message = str(exc_info.value)
    assert "BadHandler" in message
    assert "apply" in message
    assert "state, action, ctx" in message


def test_verifier_wrong_arity_verify_raises_at_class_definition():
    with pytest.raises(TypeError) as exc_info:

        class BadVerifier(Verifier):
            def verify(self, state):
                return None

    message = str(exc_info.value)
    assert "BadVerifier" in message
    assert "verify" in message
    assert "state, trajectory, task" in message


def test_rubric_wrong_arity_score_raises_at_class_definition():
    with pytest.raises(TypeError) as exc_info:

        class BadRubric(Rubric):
            def score(self, state, trajectory):
                return None

    message = str(exc_info.value)
    assert "BadRubric" in message
    assert "score" in message
    assert "state, trajectory, verifier_results, task" in message


def test_wrong_arity_error_is_raised_before_instantiation():
    # The whole point is that this fails at `class` time, not at `()` time.
    # If defining the class itself didn't raise, this test's body would never
    # run the `pytest.raises` context at all and would fail differently.
    with pytest.raises(TypeError):

        class BadHandler(TransitionHandler):
            def apply(self, state, action):  # missing ctx
                return state


# --- False-positive guards --------------------------------------------------


def test_correctly_shaped_transition_handler_subclass_defines_cleanly():
    class GoodHandler(TransitionHandler):
        def apply(self, state, action, ctx):
            return state

    assert GoodHandler is not None


def test_correctly_shaped_verifier_subclass_defines_cleanly():
    class GoodVerifier(Verifier):
        def verify(self, state, trajectory, task):
            return None

    assert GoodVerifier is not None


def test_correctly_shaped_rubric_subclass_defines_cleanly():
    class GoodRubric(Rubric):
        def score(self, state, trajectory, verifier_results, task):
            return None

    assert GoodRubric is not None


def test_subclass_using_star_args_defines_without_error():
    class VarargsHandler(TransitionHandler):
        def apply(self, *args):
            return args

    assert VarargsHandler is not None


def test_subclass_adding_keyword_only_param_with_default_defines_without_error():
    class ExtraKwHandler(TransitionHandler):
        def apply(self, state, action, ctx, *, retries=0):
            return state

    assert ExtraKwHandler is not None


def test_intermediate_subclass_not_defining_method_still_defines_and_stays_abstract():
    class IntermediateHandler(TransitionHandler):
        """An abstract intermediate that doesn't implement `apply` itself."""

    # Defining it must not raise.
    assert IntermediateHandler is not None
    # It must still be abstract: instantiating it fails because `apply` is
    # still missing, not because the arity check misfired on it.
    with pytest.raises(TypeError, match="abstract"):
        IntermediateHandler()

    # A further subclass that finally defines `apply` correctly must still
    # be checked and must define cleanly.
    class ConcreteHandler(IntermediateHandler):
        def apply(self, state, action, ctx):
            return state

    assert ConcreteHandler is not None


def test_intermediate_verifier_subclass_not_defining_method_stays_abstract():
    class IntermediateVerifier(Verifier):
        """Doesn't implement `verify` itself."""

    assert IntermediateVerifier is not None
    with pytest.raises(TypeError, match="abstract"):
        IntermediateVerifier()


def test_intermediate_rubric_subclass_not_defining_method_stays_abstract():
    class IntermediateRubric(Rubric):
        """Doesn't implement `score` itself."""

    assert IntermediateRubric is not None
    with pytest.raises(TypeError, match="abstract"):
        IntermediateRubric()


# --- Real in-tree implementations must still define cleanly ----------------


def test_real_in_tree_implementations_still_define_and_are_usable():
    # If the check were too strict, importing any of these would raise.
    from forge.runtime.reward import FunctionRubric, TaskSuccessRubric
    from forge.runtime.transition import FunctionTransitionHandler
    from forge.runtime.verifier import FunctionVerifier

    handler = FunctionTransitionHandler(lambda state, action, ctx: state)
    assert callable(handler.apply)

    verifier = FunctionVerifier(lambda state, trajectory, task: None)
    assert callable(verifier.verify)

    rubric = FunctionRubric(lambda state, trajectory, verifier_results, task: None)
    assert callable(rubric.score)

    assert callable(TaskSuccessRubric().score)
