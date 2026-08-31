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

import functools

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


# --- Decorator forms and MRO resolution (fix round 1) ----------------------
#
# Round 1 review found three gaps, all in how the method was located and
# introspected rather than in the bind-based check itself:
#   1. `inspect.signature()` raises TypeError on a raw `classmethod`
#      descriptor, which the old code's fallback silently treated as
#      "not introspectable, accept" — reopening the exact defect this task
#      closes for `@classmethod`-shaped subclasses.
#   2. The old code always assumed an implicit `self`, so a correct
#      `@staticmethod` (no implicit first argument) was wrongly rejected.
#   3. The old code read only `cls.__dict__`, so a concrete method arriving
#      from a mixin base (not `cls` itself) bypassed the check entirely,
#      even though `@abstractmethod` considers it satisfied and the class
#      instantiates.


def test_transition_handler_wrong_arity_classmethod_apply_raises_at_class_definition():
    with pytest.raises(TypeError) as exc_info:

        class BadClassmethodHandler(TransitionHandler):
            @classmethod
            def apply(cls, state):
                return state

    message = str(exc_info.value)
    assert "BadClassmethodHandler" in message
    assert "apply" in message
    assert "state, action, ctx" in message


def test_transition_handler_wrong_arity_staticmethod_apply_raises_at_class_definition():
    with pytest.raises(TypeError) as exc_info:

        class BadStaticmethodHandler(TransitionHandler):
            @staticmethod
            def apply(state):
                return state

    message = str(exc_info.value)
    assert "BadStaticmethodHandler" in message
    assert "apply" in message
    assert "state, action, ctx" in message


def test_wrong_arity_method_from_mixin_base_raises_at_class_definition():
    class _BadMixin:
        def apply(self, state):
            return state

    with pytest.raises(TypeError) as exc_info:

        class BadMixinHandler(_BadMixin, TransitionHandler):
            """Supplies no `apply` of its own; inherits the mixin's."""

    message = str(exc_info.value)
    assert "BadMixinHandler" in message
    assert "apply" in message
    assert "state, action, ctx" in message


def test_correct_staticmethod_apply_defines_without_error():
    class GoodStaticmethodHandler(TransitionHandler):
        @staticmethod
        def apply(state, action, ctx):
            return state

    assert GoodStaticmethodHandler is not None
    # And it behaves the way the engine actually calls it: instance.apply(...).
    assert GoodStaticmethodHandler().apply({"n": 1}, None, None) == {"n": 1}


def test_correct_classmethod_apply_defines_without_error():
    class GoodClassmethodHandler(TransitionHandler):
        @classmethod
        def apply(cls, state, action, ctx):
            return state

    assert GoodClassmethodHandler is not None
    assert GoodClassmethodHandler().apply({"n": 1}, None, None) == {"n": 1}


def test_correct_method_from_mixin_base_defines_without_error():
    class _GoodMixin:
        def apply(self, state, action, ctx):
            return state

    class GoodMixinHandler(_GoodMixin, TransitionHandler):
        """Supplies no `apply` of its own; inherits the mixin's correct one."""

    assert GoodMixinHandler is not None
    assert GoodMixinHandler().apply({"n": 1}, None, None) == {"n": 1}


def test_subclass_inheriting_correct_concrete_apply_from_parent_defines_without_error():
    # Not a mixin: a plain single-inheritance chain where the child adds
    # nothing. `apply` is resolved through `getattr`, so this must not be
    # treated any differently from the mixin case above.
    class BaseHandler(TransitionHandler):
        def apply(self, state, action, ctx):
            return state

    class ChildHandler(BaseHandler):
        """Defines nothing new; inherits the correct concrete `apply`."""

    assert ChildHandler is not None
    assert ChildHandler().apply({"n": 1}, None, None) == {"n": 1}


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


# --- Non-function descriptor forms (fix round 2) ----------------------------
#
# Round 2 review found three more gaps, all in the same "not introspectable,
# accept" fallback that round 1 narrowed for classmethod/staticmethod:
#   1. A wrong-arity `functools.partialmethod` was silently accepted — the
#      raw descriptor also raises TypeError from `inspect.signature()`, so
#      it fell into the accept-by-default fallback.
#   2. A `property` shadowing a contract method defined cleanly and failed
#      only at call time — same fallback, same silent acceptance.
#   3. Over-rejection: a correctly-shaped callable *instance* assigned as
#      the class attribute was rejected, because the old code assumed every
#      non-static/classmethod attribute has an implicit `self` the way a
#      plain function does.


def _apply_impl_missing_extra(self, state, action, ctx, extra):
    """No default for `extra` — a partialmethod that doesn't bind it is
    wrong-arity for TransitionHandler.apply(state, action, ctx)."""
    return state


def _apply_impl_with_extra_bound(self, state, action, ctx, extra):
    return state


def test_wrong_arity_partialmethod_apply_raises_at_class_definition():
    with pytest.raises(TypeError) as exc_info:

        class BadPartialMethodHandler(TransitionHandler):
            apply = functools.partialmethod(_apply_impl_missing_extra)

    message = str(exc_info.value)
    assert "BadPartialMethodHandler" in message
    assert "apply" in message


def test_property_shadowing_apply_raises_at_class_definition():
    with pytest.raises(TypeError) as exc_info:

        class BadPropertyHandler(TransitionHandler):
            @property
            def apply(self):
                return lambda state, action, ctx: state

    message = str(exc_info.value)
    assert "BadPropertyHandler" in message
    assert "apply" in message
    assert "property" in message


def test_correctly_shaped_callable_instance_apply_is_accepted():
    # The false-positive guard: a plain object with `__call__` assigned as
    # the class attribute is not reached through the function descriptor
    # protocol, so it has no implicit `self` the way a method does.
    class _ApplyCallable:
        def __call__(self, state, action, ctx):
            return state

    class GoodCallableInstanceHandler(TransitionHandler):
        apply = _ApplyCallable()

    assert GoodCallableInstanceHandler is not None
    # And it behaves the way the engine actually calls it.
    assert GoodCallableInstanceHandler().apply({"n": 1}, None, None) == {"n": 1}


def test_wrong_arity_callable_instance_apply_raises_at_class_definition():
    # The corresponding negative: a callable instance still gets checked,
    # not waved through just because it isn't a plain function.
    class _WrongArityApplyCallable:
        def __call__(self, state):
            return state

    with pytest.raises(TypeError) as exc_info:

        class BadCallableInstanceHandler(TransitionHandler):
            apply = _WrongArityApplyCallable()

    message = str(exc_info.value)
    assert "BadCallableInstanceHandler" in message
    assert "apply" in message


def test_correctly_shaped_partialmethod_apply_defines_and_behaves():
    class GoodPartialMethodHandler(TransitionHandler):
        apply = functools.partialmethod(_apply_impl_with_extra_bound, extra="pinned")

    assert GoodPartialMethodHandler is not None
    assert GoodPartialMethodHandler().apply({"n": 1}, None, None) == {"n": 1}
