# tests/runtime/test_state_manager.py
from __future__ import annotations

import httpx
import pytest

from forge.contracts import StateManager
from forge.runtime.http_state import HttpStateManager
from forge.runtime.state import InProcessStateManager, StateStore


def test_in_process_manager_satisfies_the_contract():
    assert isinstance(InProcessStateManager({}), StateManager)


def test_state_store_remains_importable_as_an_alias():
    # False-positive guard: renaming must not break existing imports.
    assert StateStore is InProcessStateManager


def test_get_returns_a_copy_not_a_live_reference():
    # Negative: a caller mutating the returned dict must not corrupt state.
    manager = InProcessStateManager({"tickets": []})
    manager.get()["tickets"].append("leaked")
    assert manager.get() == {"tickets": []}


def test_hash_is_stable_across_key_order():
    a = InProcessStateManager({"x": 1, "y": 2})
    b = InProcessStateManager({"y": 2, "x": 1})
    assert a.hash() == b.hash()


def test_in_process_manager_rejects_named_slots():
    # Only the container family supports slots.
    with pytest.raises(NotImplementedError):
        InProcessStateManager({}).snapshot("s1")


def test_http_manager_reads_state_over_the_wire():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/forge/state"
        return httpx.Response(200, json={"tickets": [{"id": "t1"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    manager = HttpStateManager("http://env", client=client)
    assert manager.get() == {"tickets": [{"id": "t1"}]}


def test_http_manager_supports_named_slots():
    # False-positive guard: the container family does support slots, so the
    # base class's refusal must be overridden here.
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    manager = HttpStateManager("http://env", client=client)
    manager.snapshot("s1")
    manager.restore("s1")
    assert calls == ["/forge/snapshot", "/forge/restore/s1"]
