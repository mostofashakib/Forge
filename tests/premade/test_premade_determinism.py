"""The premade Gmail and Slack apps replay an episode byte for byte.

Every id the apps mint and every timestamp they stamp used to come from
`uuid4()` and the wall clock, so the same actions after the same reset gave a
different state each time. Ids now count up and time is a virtual clock that
starts just after the newest seed item. Both live in SQLite with the rest of
the state, so they reset, snapshot, and survive a restart with it.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient


def _boot(load_premade, app_name: str) -> TestClient:
    """Start a fresh process of a premade app on the test's SQLite file."""
    return TestClient(load_premade(app_name).app)


def _gmail_episode(client: TestClient) -> list:
    client.post("/forge/reset")
    draft = client.post("/compose", json={"to": "x@y.com", "subject": "Hi", "body": "b"}).json()
    return [
        draft,
        client.post("/send", json={"draft_id": draft["draft_id"]}).json(),
        # e013 is in an auto-reply thread, so this also mints the reply's reply.
        client.post("/reply", json={"email_id": "e013", "body": "Stopping them now."}).json(),
        client.post("/forward", json={"email_id": "e001", "to": "z@y.com"}).json(),
        client.post("/create_label", json={"name": "Q3"}).json(),
        client.post("/receive", json={"from_addr": "a@b.com", "subject": "S", "body": "B"}).json(),
        client.get("/forge/state").json(),
    ]


def _slack_episode(client: TestClient) -> list:
    client.post("/forge/reset")
    return [
        # #engineering has an auto-responder, so this also mints its reply.
        client.post("/send_message", json={"channel": "engineering", "text": "Deploying"}).json(),
        client.post("/reply_thread", json={"channel": "engineering", "message_id": "m010", "text": "LGTM"}).json(),
        client.post("/add_reaction", json={"channel": "engineering", "message_id": "m011", "emoji": "rocket"}).json(),
        client.post("/create_channel", json={"name": "launch"}).json(),
        client.post("/send_dm", json={"to": "bob", "text": "ping"}).json(),
        client.post("/receive_dm", json={"from_user": "bob", "text": "pong"}).json(),
        client.get("/forge/state").json(),
    ]


EPISODES = {"gmail": _gmail_episode, "slack": _slack_episode}


@pytest.mark.parametrize("app_name", EPISODES)
def test_the_same_actions_after_reset_replay_identically(app_name, load_premade):
    client = _boot(load_premade, app_name)

    assert EPISODES[app_name](client) == EPISODES[app_name](client)


@pytest.mark.parametrize("app_name", EPISODES)
def test_a_restarted_container_replays_the_same_episode(app_name, load_premade):
    # A new process has none of the old one's in-memory state, so anything
    # held outside SQLite would show up here as a difference.
    first = EPISODES[app_name](_boot(load_premade, app_name))
    second = EPISODES[app_name](_boot(load_premade, app_name))

    assert first == second


def test_new_mail_is_stamped_after_the_seed_data_and_never_with_today(load_premade):
    client = _boot(load_premade, "gmail")
    received = client.post(
        "/receive", json={"from_addr": "a@b.com", "subject": "New", "body": "B"}
    ).json()

    inbox = received["state"]["inbox"]
    # Newest first still holds: the virtual clock starts after every seed item.
    assert inbox[0]["id"] == received["email_id"]
    today = datetime.now(timezone.utc).date().isoformat()
    assert not inbox[0]["timestamp"].startswith(today)


def test_restoring_a_snapshot_rewinds_ids_and_time_with_the_data(load_premade):
    client = _boot(load_premade, "gmail")
    client.post("/forge/reset")
    client.post("/forge/snapshot", json={"slot": "before"})

    def compose() -> dict:
        draft = client.post("/compose", json={"to": "x@y.com", "subject": "S"}).json()
        return next(e for e in client.post("/search", json={"query": "x@y.com"}).json()["results"]
                    if e["id"] == draft["draft_id"])

    first = compose()
    client.post("/forge/restore/before")
    again = compose()

    assert (again["id"], again["timestamp"]) == (first["id"], first["timestamp"])


@pytest.mark.parametrize("app_name", EPISODES)
def test_a_restarted_process_never_reuses_an_id_already_in_the_database(
    app_name, load_premade
):
    # Counters live in SQLite, not in the process, so a container restart
    # that keeps its database keeps counting instead of colliding.
    first = EPISODES[app_name](_boot(load_premade, app_name))[0]
    restarted = _boot(load_premade, app_name)
    if app_name == "gmail":
        second = restarted.post("/compose", json={"to": "x@y.com", "subject": "S"})
        key = "draft_id"
    else:
        second = restarted.post("/send_message", json={"channel": "random", "text": "hi"})
        key = "message_id"

    assert second.status_code == 200
    assert second.json()[key] != first[key]


def test_the_slack_auto_responder_rate_limit_runs_on_the_virtual_clock(load_premade):
    # False-positive guard: the responder still fires once per burst of
    # messages instead of on every message.
    client = _boot(load_premade, "slack")
    client.post("/forge/reset")
    for text in ("one", "two", "three"):
        client.post("/send_message", json={"channel": "general", "text": text})

    messages = client.post("/get_channel_messages", json={"channel": "general", "limit": 500}).json()
    responses = [m for m in messages["messages"] if m["user"] == "bob" and m["text"] == "👍 noted!"]
    assert len(responses) == 1
