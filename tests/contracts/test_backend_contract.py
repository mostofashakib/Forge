from __future__ import annotations

import pytest

from forge.contracts import Action, ActionResult, ExecutionBackend, TransitionHandler


class _Echo(ExecutionBackend):
    def execute(self, action: Action, state: dict, ctx) -> ActionResult:
        return ActionResult(
            state={**state, "last": action.type},
            events=[{"type": f"{action.type}_done"}],
        )


def test_a_backend_returns_new_state_and_events():
    result = _Echo().execute(Action(type="close"), {"n": 1}, None)
    assert result.state == {"n": 1, "last": "close"}
    assert result.events == [{"type": "close_done"}]
    assert result.error is None


def test_close_is_concrete_so_stateless_backends_need_not_define_it():
    # False-positive guard: an in-process backend holds no connection to close.
    _Echo().close()


def test_a_backend_missing_execute_cannot_be_instantiated():
    class Incomplete(ExecutionBackend):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_a_transition_handler_missing_apply_cannot_be_instantiated():
    class Incomplete(TransitionHandler):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
