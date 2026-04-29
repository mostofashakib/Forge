# tests/runtime/test_state.py
import copy
from forge.runtime.state import StateStore


SAMPLE_STATE = {
    "emails": {
        "e_0000": {"id": "e_0000", "labels": ["inbox"], "archived": False}
    },
    "users": {
        "u_0000": {"id": "u_0000", "email": "agent@example.com"}
    },
}


def test_get_returns_deep_copy():
    store = StateStore(SAMPLE_STATE)
    s1 = store.get()
    s1["emails"]["e_0000"]["archived"] = True
    s2 = store.get()
    assert s2["emails"]["e_0000"]["archived"] is False


def test_apply_updates_state():
    store = StateStore(SAMPLE_STATE)
    new_state = copy.deepcopy(SAMPLE_STATE)
    new_state["emails"]["e_0000"]["archived"] = True
    store.apply(new_state)
    assert store.get()["emails"]["e_0000"]["archived"] is True


def test_hash_is_stable_for_same_state():
    store = StateStore(SAMPLE_STATE)
    assert store.hash() == store.hash()


def test_hash_changes_after_mutation():
    store = StateStore(SAMPLE_STATE)
    h1 = store.hash()
    new_state = copy.deepcopy(SAMPLE_STATE)
    new_state["emails"]["e_0000"]["archived"] = True
    store.apply(new_state)
    assert store.hash() != h1


def test_hash_starts_with_sha256_prefix():
    store = StateStore(SAMPLE_STATE)
    assert store.hash().startswith("sha256:")


def test_same_state_always_produces_same_hash():
    s1 = StateStore(SAMPLE_STATE)
    s2 = StateStore(SAMPLE_STATE)
    assert s1.hash() == s2.hash()
