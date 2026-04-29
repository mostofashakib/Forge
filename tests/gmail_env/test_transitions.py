# tests/gmail_env/test_transitions.py
import pytest
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError
from examples.gmail_env.initial_state import GmailInitialStateFactory
from examples.gmail_env.transitions.reply_email import apply_reply_email
from examples.gmail_env.transitions.send_email import apply_send_email
from examples.gmail_env.transitions.archive_email import apply_archive_email


def make_state(seed: int = 0) -> tuple[dict, RuntimeContext]:
    ctx = RuntimeContext(seed=seed)
    state = GmailInitialStateFactory().create(ctx, {})
    return state, ctx


def get_first_thread_id(state: dict) -> str:
    return next(iter(state["threads"]))


def get_first_email_id(state: dict) -> str:
    return next(iter(state["emails"]))


# --- reply_email ---

def test_reply_email_adds_new_email_to_thread():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_reply_email(state, {"type": "reply_email", "thread_id": thread_id, "body": "Hello"}, ctx)
    thread_email_ids = result.state["threads"][thread_id]["email_ids"]
    assert len(thread_email_ids) == 2


def test_reply_email_new_email_has_sent_label():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_reply_email(state, {"type": "reply_email", "thread_id": thread_id, "body": "Hi"}, ctx)
    thread = result.state["threads"][thread_id]
    new_email_id = thread["email_ids"][-1]
    assert "sent" in result.state["emails"][new_email_id]["labels"]


def test_reply_email_emits_email_replied_event():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_reply_email(state, {"type": "reply_email", "thread_id": thread_id, "body": "Hi"}, ctx)
    assert any(e["type"] == "email_replied" for e in result.events)


def test_reply_email_raises_for_unknown_thread():
    state, ctx = make_state()
    with pytest.raises(InvalidActionError):
        apply_reply_email(state, {"type": "reply_email", "thread_id": "bad_id", "body": "Hi"}, ctx)


def test_reply_email_does_not_mutate_original_state():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    original_count = len(state["threads"][thread_id]["email_ids"])
    apply_reply_email(state, {"type": "reply_email", "thread_id": thread_id, "body": "Hi"}, ctx)
    assert len(state["threads"][thread_id]["email_ids"]) == original_count


# --- send_email ---

def test_send_email_creates_new_thread_and_email():
    state, ctx = make_state()
    result = apply_send_email(
        state,
        {"type": "send_email", "to": "other@example.com", "subject": "Hello", "body": "Hi there"},
        ctx,
    )
    assert len(result.state["threads"]) == 2
    assert len(result.state["emails"]) == 2


def test_send_email_emits_email_sent_event():
    state, ctx = make_state()
    result = apply_send_email(
        state,
        {"type": "send_email", "to": "x@x.com", "subject": "S", "body": "B"},
        ctx,
    )
    assert any(e["type"] == "email_sent" for e in result.events)


# --- archive_email ---

def test_archive_email_sets_archived_true():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_archive_email(state, {"type": "archive_email", "email_id": email_id}, ctx)
    assert result.state["emails"][email_id]["archived"] is True


def test_archive_email_emits_email_archived_event():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_archive_email(state, {"type": "archive_email", "email_id": email_id}, ctx)
    assert any(e["type"] == "email_archived" for e in result.events)


def test_archive_email_raises_for_unknown_email():
    state, ctx = make_state()
    with pytest.raises(InvalidActionError):
        apply_archive_email(state, {"type": "archive_email", "email_id": "bad_id"}, ctx)


from examples.gmail_env.transitions.apply_label import apply_apply_label
from examples.gmail_env.transitions.mark_read import apply_mark_read
from examples.gmail_env.transitions.escalate_thread import apply_escalate_thread


# --- apply_label ---

def test_apply_label_adds_label_to_email():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_apply_label(state, {"type": "apply_label", "email_id": email_id, "label": "urgent"}, ctx)
    assert "urgent" in result.state["emails"][email_id]["labels"]


def test_apply_label_is_idempotent():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result1 = apply_apply_label(state, {"type": "apply_label", "email_id": email_id, "label": "urgent"}, ctx)
    result2 = apply_apply_label(result1.state, {"type": "apply_label", "email_id": email_id, "label": "urgent"}, ctx)
    assert result2.state["emails"][email_id]["labels"].count("urgent") == 1


def test_apply_label_emits_label_applied_event():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_apply_label(state, {"type": "apply_label", "email_id": email_id, "label": "urgent"}, ctx)
    assert any(e["type"] == "label_applied" for e in result.events)


def test_apply_label_raises_for_unknown_email():
    state, ctx = make_state()
    with pytest.raises(InvalidActionError):
        apply_apply_label(state, {"type": "apply_label", "email_id": "bad", "label": "x"}, ctx)


# --- mark_read ---

def test_mark_read_sets_read_true():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    assert state["emails"][email_id]["read"] is False
    result = apply_mark_read(state, {"type": "mark_read", "email_id": email_id}, ctx)
    assert result.state["emails"][email_id]["read"] is True


def test_mark_read_emits_email_read_event():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_mark_read(state, {"type": "mark_read", "email_id": email_id}, ctx)
    assert any(e["type"] == "email_read" for e in result.events)


# --- escalate_thread ---

def test_escalate_thread_sets_escalated_true_on_thread():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_escalate_thread(state, {"type": "escalate_thread", "thread_id": thread_id}, ctx)
    assert result.state["threads"][thread_id]["escalated"] is True


def test_escalate_thread_emits_thread_escalated_event():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_escalate_thread(state, {"type": "escalate_thread", "thread_id": thread_id}, ctx)
    assert any(e["type"] == "thread_escalated" for e in result.events)


def test_escalate_thread_raises_for_unknown_thread():
    state, ctx = make_state()
    with pytest.raises(InvalidActionError):
        apply_escalate_thread(state, {"type": "escalate_thread", "thread_id": "bad"}, ctx)
