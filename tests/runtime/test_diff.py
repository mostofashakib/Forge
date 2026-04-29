# tests/runtime/test_diff.py
from forge.runtime.diff import compute_diff


def test_no_change_returns_empty_diff():
    state = {"emails": {"e_0": {"id": "e_0", "archived": False}}}
    diff = compute_diff(state, state)
    assert diff == {"added": {}, "changed": {}, "removed": {}}


def test_added_entity_appears_in_added():
    before = {"emails": {"e_0": {"id": "e_0"}}}
    after = {"emails": {"e_0": {"id": "e_0"}, "e_1": {"id": "e_1"}}}
    diff = compute_diff(before, after)
    assert "emails.e_1" in diff["added"]
    assert diff["added"]["emails.e_1"] == {"id": "e_1"}


def test_removed_entity_appears_in_removed():
    before = {"emails": {"e_0": {"id": "e_0"}, "e_1": {"id": "e_1"}}}
    after = {"emails": {"e_0": {"id": "e_0"}}}
    diff = compute_diff(before, after)
    assert "emails.e_1" in diff["removed"]


def test_changed_field_appears_in_changed_with_before_and_after():
    before = {"emails": {"e_0": {"id": "e_0", "labels": ["inbox"]}}}
    after = {"emails": {"e_0": {"id": "e_0", "labels": ["inbox", "urgent"]}}}
    diff = compute_diff(before, after)
    assert "emails.e_0.labels" in diff["changed"]
    assert diff["changed"]["emails.e_0.labels"]["before"] == ["inbox"]
    assert diff["changed"]["emails.e_0.labels"]["after"] == ["inbox", "urgent"]


def test_multiple_changes_all_captured():
    before = {
        "emails": {"e_0": {"id": "e_0", "archived": False, "labels": ["inbox"]}},
        "threads": {"t_0": {"id": "t_0", "escalated": False}},
    }
    after = {
        "emails": {"e_0": {"id": "e_0", "archived": True, "labels": ["inbox"]}},
        "threads": {"t_0": {"id": "t_0", "escalated": True}},
    }
    diff = compute_diff(before, after)
    assert "emails.e_0.archived" in diff["changed"]
    assert "threads.t_0.escalated" in diff["changed"]


def test_unchanged_fields_not_in_changed():
    before = {"emails": {"e_0": {"id": "e_0", "archived": False, "labels": ["inbox"]}}}
    after = {"emails": {"e_0": {"id": "e_0", "archived": True, "labels": ["inbox"]}}}
    diff = compute_diff(before, after)
    assert "emails.e_0.id" not in diff["changed"]
    assert "emails.e_0.labels" not in diff["changed"]
