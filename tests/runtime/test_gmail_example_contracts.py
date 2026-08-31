"""The worked example must stay in the shape the generator now emits."""
from __future__ import annotations

from forge.contracts import InitialStateProvider, Rubric, Verifier
from forge.contracts.backend import TransitionHandler
from examples.gmail_env.gym_wrapper import build_gmail_env
from examples.gmail_env.initial_state import GmailInitialStateFactory
from examples.gmail_env.transitions.archive_email import ArchiveEmailHandler
from examples.gmail_env.verifiers.reply_to_customer import ReplyToCustomerVerifier
from examples.gmail_env.rewards.base import GmailRubric


def test_the_initial_state_factory_is_a_provider():
    assert isinstance(GmailInitialStateFactory(), InitialStateProvider)


def test_a_transition_is_a_handler():
    assert isinstance(ArchiveEmailHandler(), TransitionHandler)


def test_a_verifier_is_a_verifier():
    assert isinstance(ReplyToCustomerVerifier(), Verifier)


def test_the_rubric_is_a_rubric():
    assert isinstance(GmailRubric(), Rubric)


def test_the_example_env_builds_and_resets():
    env = build_gmail_env()
    obs, info = env.reset(seed=3)
    assert info["seed"] == 3
    assert set(info.keys()) >= {"episode_id", "seed", "task"}
    assert set(obs.keys()) >= {"actor_id", "emails", "labels", "threads", "users"}


def test_the_example_env_takes_a_step():
    env = build_gmail_env()
    obs, _info = env.reset(seed=3)
    email_id = next(iter(obs["emails"]))
    assert obs["emails"][email_id]["archived"] is False  # precondition
    obs, _reward, _term, _trunc, info = env.step(
        {"type": "archive_email", "email_id": email_id}
    )
    assert obs["emails"][email_id]["archived"] is True  # the action actually did something
    assert any(e["type"] == "email_archived" for e in info["events"])


def test_the_factory_no_longer_exposes_create():
    assert not hasattr(GmailInitialStateFactory(), "create")
