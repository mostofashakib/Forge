from __future__ import annotations

import pytest

from forge.contracts import StateManager


class _Conforming(StateManager):
    def __init__(self) -> None:
        self._state: dict = {}

    def get(self) -> dict:
        return dict(self._state)

    def apply(self, state: dict) -> None:
        self._state = dict(state)

    def hash(self) -> str:
        return f"sha256:{len(self._state)}"


def test_a_conforming_state_manager_instantiates():
    manager = _Conforming()
    manager.apply({"a": 1})
    assert manager.get() == {"a": 1}


def test_a_state_manager_missing_a_method_cannot_be_instantiated():
    # Negative: the failure must land at instantiation, not at first call.
    class Incomplete(StateManager):
        def get(self) -> dict:
            return {}

        def apply(self, state: dict) -> None:
            return None

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_snapshot_slots_are_optional_and_fail_loudly():
    # False-positive guard: only the container family supports slots, so
    # snapshot/restore are concrete and must raise rather than silently no-op.
    manager = _Conforming()
    with pytest.raises(NotImplementedError):
        manager.snapshot("slot_a")
    with pytest.raises(NotImplementedError):
        manager.restore("slot_a")
