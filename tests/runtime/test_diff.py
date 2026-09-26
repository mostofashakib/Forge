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


def _diff_json_under_hash_seed(hash_seed: str) -> str:
    """Run compute_diff in a fresh interpreter with a fixed string-hash salt."""
    import os
    import subprocess
    import sys

    script = (
        "import json\n"
        "from forge.runtime.diff import compute_diff\n"
        "names = [f'entity_{i}' for i in range(40)]\n"
        "before = {'tickets': {n: {'status': 'open', 'owner': 'a'} for n in names[:20]},\n"
        "          'notes': {n: {'text': 'x'} for n in names[:10]}}\n"
        "after = {'tickets': {n: {'status': 'closed', 'owner': 'b'} for n in names[10:30]},\n"
        "         'labels': {n: {'name': n} for n in names[:10]}}\n"
        "print(json.dumps(compute_diff(before, after)))\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    return subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True,
    ).stdout


def test_diff_key_order_does_not_depend_on_string_hash_salt():
    # Trajectories and exports serialize the diff as-is, so its key order must
    # be identical across processes, not just within one.
    outputs = {_diff_json_under_hash_seed(seed) for seed in ("1", "2", "3", "4")}
    assert len(outputs) == 1
