# tests/gmail_env/test_verifiers.py
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import StepSnapshot
from forge.runtime.trajectory import Trajectory
from examples.gmail_env.initial_state import GmailInitialStateFactory
from examples.gmail_env.transitions.reply_email import apply_reply_email
from examples.gmail_env.transitions.apply_label import apply_apply_label
from examples.gmail_env.transitions.archive_email import apply_archive_email
from examples.gmail_env.transitions.escalate_thread import apply_escalate_thread
from examples.gmail_env.verifiers.reply_to_customer import verify_reply_to_customer
from examples.gmail_env.verifiers.label_urgent_request import verify_label_urgent_request
from examples.gmail_env.verifiers.archive_newsletter import verify_archive_newsletter
from examples.gmail_env.verifiers.escalate_billing_complaint import verify_escalate_billing_complaint


def make_state(seed: int = 0):
    ctx = RuntimeContext(seed=seed)
    state = GmailInitialStateFactory().create(ctx, {})
    return state, ctx


def empty_trajectory() -> Trajectory:
    return Trajectory(episode_id="ep_0", steps=[])


def trajectory_with_events(events: list[dict]) -> Trajectory:
    step = StepSnapshot(
        episode_id="ep_0", step_index=0,
        state_hash_before="sha256:a", state_hash_after="sha256:b",
        action={"type": "noop"}, events=events, reward=0.0,
        verifier_results=[], diff={"added": {}, "changed": {}, "removed": {}},
        terminated=False, truncated=False,
    )
    return Trajectory(episode_id="ep_0", steps=[step])


def get_first_thread_id(state):
    return next(iter(state["threads"]))


def get_first_email_id(state):
    return next(iter(state["emails"]))


# --- reply_to_customer ---

def test_reply_to_customer_passes_when_reply_sent():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_reply_email(state, {"type": "reply_email", "thread_id": thread_id, "body": "Hi"}, ctx)
    traj = trajectory_with_events(result.events)
    task = {"name": "reply_to_customer", "verifier_id": "reply_to_customer", "inputs": {"thread_id": thread_id}}
    vr = verify_reply_to_customer(result.state, traj, task)
    assert vr.passed is True


def test_reply_to_customer_fails_with_no_reply():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    task = {"name": "reply_to_customer", "verifier_id": "reply_to_customer", "inputs": {"thread_id": thread_id}}
    vr = verify_reply_to_customer(state, empty_trajectory(), task)
    assert vr.passed is False
    assert vr.checks[0].evidence is not None


# --- label_urgent_request ---

def test_label_urgent_passes_when_urgent_label_applied():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_apply_label(state, {"type": "apply_label", "email_id": email_id, "label": "urgent"}, ctx)
    task = {"name": "label_urgent_request", "verifier_id": "label_urgent_request", "inputs": {"email_id": email_id}}
    vr = verify_label_urgent_request(result.state, empty_trajectory(), task)
    assert vr.passed is True


def test_label_urgent_fails_without_label():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    task = {"name": "label_urgent_request", "verifier_id": "label_urgent_request", "inputs": {"email_id": email_id}}
    vr = verify_label_urgent_request(state, empty_trajectory(), task)
    assert vr.passed is False


# --- archive_newsletter ---

def test_archive_newsletter_passes_when_archived():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_archive_email(state, {"type": "archive_email", "email_id": email_id}, ctx)
    task = {"name": "archive_newsletter", "verifier_id": "archive_newsletter", "inputs": {"email_id": email_id}}
    vr = verify_archive_newsletter(result.state, empty_trajectory(), task)
    assert vr.passed is True


def test_archive_newsletter_fails_when_not_archived():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    task = {"name": "archive_newsletter", "verifier_id": "archive_newsletter", "inputs": {"email_id": email_id}}
    vr = verify_archive_newsletter(state, empty_trajectory(), task)
    assert vr.passed is False


# --- escalate_billing_complaint ---

def test_escalate_billing_passes_when_thread_escalated():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_escalate_thread(state, {"type": "escalate_thread", "thread_id": thread_id}, ctx)
    task = {"name": "escalate_billing_complaint", "verifier_id": "escalate_billing_complaint", "inputs": {"thread_id": thread_id}}
    vr = verify_escalate_billing_complaint(result.state, empty_trajectory(), task)
    assert vr.passed is True


def test_escalate_billing_fails_when_not_escalated():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    task = {"name": "escalate_billing_complaint", "verifier_id": "escalate_billing_complaint", "inputs": {"thread_id": thread_id}}
    vr = verify_escalate_billing_complaint(state, empty_trajectory(), task)
    assert vr.passed is False
