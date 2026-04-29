# tests/runtime/test_transition.py
import copy
import pytest
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError
from forge.runtime.transition import TransitionEngine, TransitionResult


def make_noop_transition(state, action, ctx):
    return TransitionResult(state=copy.deepcopy(state), events=[])


def make_mutation_transition(state, action, ctx):
    new_state = copy.deepcopy(state)
    new_state["counter"] = new_state.get("counter", 0) + 1
    return TransitionResult(
        state=new_state,
        events=[{"type": "counter_incremented", "entity_id": "counter"}],
    )


def test_registered_action_dispatches_correctly():
    engine = TransitionEngine()
    engine.register("increment", make_mutation_transition)
    ctx = RuntimeContext(seed=0)
    result = engine.apply({"counter": 0}, {"type": "increment"}, ctx)
    assert result.state["counter"] == 1
    assert result.events[0]["type"] == "counter_incremented"


def test_unknown_action_raises_invalid_action_error():
    engine = TransitionEngine()
    ctx = RuntimeContext(seed=0)
    with pytest.raises(InvalidActionError) as exc_info:
        engine.apply({}, {"type": "nonexistent"}, ctx)
    assert exc_info.value.code == "UNKNOWN_ACTION_TYPE"


def test_action_types_returns_registered_types():
    engine = TransitionEngine()
    engine.register("a", make_noop_transition)
    engine.register("b", make_noop_transition)
    assert engine.action_types == {"a", "b"}


def test_transition_does_not_mutate_original_state():
    engine = TransitionEngine()
    engine.register("increment", make_mutation_transition)
    ctx = RuntimeContext(seed=0)
    original = {"counter": 0}
    engine.apply(original, {"type": "increment"}, ctx)
    assert original["counter"] == 0
