"""LayeredVerifier satisfies the Verifier contract by inheritance.

It used to satisfy it only through `EnvBuilder._as_verifier`, which wraps any
plain callable in `FunctionVerifier` when `isinstance(fn, Verifier)` is False.
That worked, but it meant the one verifier the composer actually builds was not
a `Verifier`, and anything reaching for `.verify()` on it failed.

`__call__` is kept and still has real callers — `forge/benchmark/_eval.py`
invokes verifiers as callables — so these tests pin both entry points, and pin
that they agree.
"""
from __future__ import annotations

import pytest

from forge.contracts import Verifier
from forge.runtime.env_builder import _as_verifier
from forge.runtime.layered_verifier import LayeredVerifier
from forge.runtime.verifier import FunctionVerifier


class _Trajectory:
    def __init__(self, events=None, steps=None) -> None:
        self.events = events or []
        self.steps = steps or []


def _verifier() -> LayeredVerifier:
    v = LayeredVerifier("reply_task")
    v.add_state_check("replied", lambda state, trajectory, task: state.get("replied", False))
    return v


def test_a_layered_verifier_is_a_verifier():
    assert isinstance(_verifier(), Verifier)


def test_verify_reaches_the_same_verdict_as_calling_it():
    # The two entry points must not drift: `__call__` is kept for the callers
    # that already use it, and it delegates rather than duplicating.
    verifier = _verifier()
    state, trajectory, task = {"replied": True}, _Trajectory(), {"name": "t"}

    assert verifier.verify(state, trajectory, task) == verifier(state, trajectory, task)


def test_verify_reports_a_failure():
    # Negative case: a passing-only test would not notice a verify() that
    # always returned a passing result.
    result = _verifier().verify({"replied": False}, _Trajectory(), {"name": "t"})

    assert result.passed is False
    assert result.explanation == "failed layers: state"


def test_verify_reports_a_pass():
    result = _verifier().verify({"replied": True}, _Trajectory(), {"name": "t"})

    assert result.passed is True
    assert result.verifier_id == "reply_task"


def test_the_builder_registers_it_directly_rather_than_wrapping_it():
    # This is what inheriting actually buys. Before, `_as_verifier` fell
    # through to `FunctionVerifier(fn)` because the isinstance check failed,
    # so the object the engine held was a wrapper, not the verifier itself.
    verifier = _verifier()

    registered = _as_verifier(verifier)

    assert registered is verifier
    assert not isinstance(registered, FunctionVerifier)


def test_a_wrong_arity_subclass_is_rejected_at_definition():
    # Inheriting brings the contract's arity guard with it: a subclass that
    # redefines verify() with the wrong shape now fails at import rather than
    # mid-episode.
    with pytest.raises(TypeError, match="verify"):

        class Broken(LayeredVerifier):
            def verify(self, state):  # missing trajectory and task
                return None


def test_a_correctly_shaped_subclass_still_defines():
    # False-positive guard: the arity check must not reject a legitimate
    # override, or the class becomes unextendable.
    class Narrowed(LayeredVerifier):
        def verify(self, state, trajectory, task):
            return super().verify(state, trajectory, task)

    result = Narrowed("x").verify({}, _Trajectory(), {"name": "t"})
    assert result.verifier_id == "x"
