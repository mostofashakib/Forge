"""Seeding and checking the state of in-process and app environments.

Both are pure functions over plain state dicts, so the in-process and
container runners share one meaning for "seed" and "check".
"""
from __future__ import annotations

import pytest

from forge.taskfactory.schemas import Check
from forge.taskfactory.state_checks import apply_seed, evaluate_state_check, resolve_path

LIST_STATE = {
    "emails": [
        {"id": "e001", "folder": "inbox", "subject": "Invoice 42", "is_read": False},
        {"id": "e002", "folder": "inbox", "subject": "Lunch", "is_read": True},
    ],
    "inbox_unread": 1,
    "meta": {"owner": {"name": "Sam"}},
}
DICT_STATE = {
    "tickets": {
        "t1": {"id": "t1", "status": "open", "priority": "low"},
        "t2": {"id": "t2", "status": "closed", "priority": "high"},
    }
}


# ---------------------------------------------------------------------------
# apply_seed
# ---------------------------------------------------------------------------

def test_a_seed_row_with_a_new_id_is_appended_to_a_list_collection():
    seeded = apply_seed(LIST_STATE, {"emails": [{"id": "e003", "folder": "inbox", "subject": "Refund"}]})

    assert [e["id"] for e in seeded["emails"]] == ["e001", "e002", "e003"]


def test_a_seed_row_with_an_existing_id_merges_into_that_row():
    seeded = apply_seed(LIST_STATE, {"emails": [{"id": "e002", "is_read": False}]})

    row = next(e for e in seeded["emails"] if e["id"] == "e002")
    assert row == {"id": "e002", "folder": "inbox", "subject": "Lunch", "is_read": False}
    assert len(seeded["emails"]) == 2


def test_a_seed_row_is_keyed_by_id_in_a_dict_collection():
    seeded = apply_seed(DICT_STATE, {"tickets": [{"id": "t3", "status": "open"}, {"id": "t1", "priority": "high"}]})

    assert seeded["tickets"]["t3"] == {"id": "t3", "status": "open"}
    assert seeded["tickets"]["t1"]["priority"] == "high"
    assert seeded["tickets"]["t1"]["status"] == "open"


def test_seeding_never_mutates_the_base_state():
    apply_seed(LIST_STATE, {"emails": [{"id": "e001", "is_read": True}]})

    assert LIST_STATE["emails"][0]["is_read"] is False


def test_a_seed_into_a_missing_collection_is_rejected():
    with pytest.raises(ValueError, match="calendar"):
        apply_seed(LIST_STATE, {"calendar": [{"id": "c1"}]})


def test_a_dict_collection_row_without_an_id_is_rejected():
    with pytest.raises(ValueError, match="id"):
        apply_seed(DICT_STATE, {"tickets": [{"status": "open"}]})


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------

def test_paths_walk_nested_keys_and_list_indexes():
    assert resolve_path(LIST_STATE, "meta.owner.name") == "Sam"
    assert resolve_path(LIST_STATE, "emails.1.subject") == "Lunch"


def test_a_path_that_does_not_exist_raises_key_error():
    with pytest.raises(KeyError):
        resolve_path(LIST_STATE, "meta.owner.email")


# ---------------------------------------------------------------------------
# evaluate_state_check
# ---------------------------------------------------------------------------

def _check(**fields) -> Check:
    return Check(**fields)


def test_record_exists_passes_when_a_matching_row_has_the_expected_fields():
    check = _check(kind="record_exists", collection="emails", match={"id": "e001"}, expect={"folder": "inbox"})

    assert evaluate_state_check(check, LIST_STATE, []).passed


def test_record_exists_fails_when_the_expected_field_differs():
    check = _check(kind="record_exists", collection="emails", match={"id": "e001"}, expect={"folder": "archive"})

    outcome = evaluate_state_check(check, LIST_STATE, [])
    assert not outcome.passed
    assert "e001" in outcome.detail or "archive" in outcome.detail


def test_record_absent_fails_when_a_row_matches():
    check = _check(kind="record_absent", collection="tickets", match={"status": "closed"})

    assert not evaluate_state_check(check, DICT_STATE, []).passed


def test_record_count_compares_the_number_of_matching_rows():
    check = _check(kind="record_count", collection="emails", match={"folder": "inbox"}, op=">=", value=2)

    assert evaluate_state_check(check, LIST_STATE, []).passed
    assert not evaluate_state_check(check.model_copy(update={"value": 3}), LIST_STATE, []).passed


def test_value_check_supports_contains():
    check = _check(kind="value", path="emails.0.subject", op="contains", value="Invoice")

    assert evaluate_state_check(check, LIST_STATE, []).passed


def test_a_value_check_on_a_missing_path_fails_instead_of_raising():
    check = _check(kind="value", path="nope.deeper", op="==", value=1)

    outcome = evaluate_state_check(check, LIST_STATE, [])
    assert not outcome.passed
    assert "nope" in outcome.detail


def test_actions_called_in_order_requires_that_order():
    check = _check(kind="actions_called", tools=["read", "archive"], ordered=True)

    assert evaluate_state_check(check, {}, ["read", "label", "archive"]).passed
    assert not evaluate_state_check(check, {}, ["archive", "read"]).passed


def test_actions_called_unordered_needs_every_tool():
    check = _check(kind="actions_called", tools=["read", "archive"])

    assert evaluate_state_check(check, {}, ["archive", "read"]).passed
    assert not evaluate_state_check(check, {}, ["read"]).passed


def test_actions_not_called_fails_when_a_forbidden_tool_ran():
    check = _check(kind="actions_not_called", tools=["delete"])

    assert evaluate_state_check(check, {}, ["read"]).passed
    assert not evaluate_state_check(check, {}, ["read", "delete"]).passed
