#!/usr/bin/env python3
"""The transition engine is generic: it knows rules, not tasks.

Every scenario in this file is invented here. None of it mentions the Acme
migration, and the engine is never told which scenario it is running. If any
task-specific knowledge leaks back into `slack_sim.tracker`, these tests
keep passing only by accident -- so each one asserts on a rule shape rather
than on a particular message.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from slack_sim.identity import LOGGED_IN_USER
from slack_sim.models import LatentMessage, ScenarioEvent, ScenarioRule
from slack_sim.scenario import Scenario
from slack_sim.service import execute_tool, export_state, seed_database

# A neutral thread from the base workspace, used only as somewhere to reply.
THREAD = "MSG041"
ACTOR = LOGGED_IN_USER.user_id


def build(root: Path, scenario: Scenario) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    db = root / "slack.db"
    seed_database(db, root / "snapshot.sql", scenario=scenario)
    return db


def events(db: Path) -> dict[str, str]:
    return {event["event_id"]: event["status"] for event in export_state(db)["scenario_events"]}


def visible(db: Path) -> set[str]:
    return {message["message_id"] for message in export_state(db)["messages"]}


def reply(db: Path, body: str, thread: str = THREAD) -> dict:
    return execute_tool(db, "reply_to_thread", {"thread_parent_id": thread, "body": body}, ACTOR)


def read(db: Path, channel_id: str) -> dict:
    return execute_tool(db, "get_channel_messages", {"channel_id": channel_id}, ACTOR)


def latent(message_id: str, event_id: str, text: str = "released") -> LatentMessage:
    return LatentMessage(
        message_id=message_id, event_id=event_id, conversation_id="C001",
        author_id="U025", text=text, thread_parent_id=THREAD, reply_to_id=THREAD,
    )


def test_a_keyword_rule_publishes_only_when_the_reply_matches(root: Path) -> None:
    scenario = Scenario(
        events=(ScenarioEvent("greeted"),),
        latent_messages=(latent("XLAT01", "greeted"),),
        rules=(
            ScenarioRule(
                rule_id="r1", event_id="greeted", trigger="reply_keywords", thread_id=THREAD,
                keyword_groups=(("hello", "hi"), ("world", "everyone")),
            ),
        ),
    )
    db = build(root, scenario)

    reply(db, "hello there")
    assert events(db)["greeted"] == "pending", "one group matched; the rule needs both"
    assert "XLAT01" not in visible(db)

    reply(db, "hi world")
    assert events(db)["greeted"] == "activated"
    assert "XLAT01" in visible(db), "activation must publish the event's latent messages"


def test_an_exact_rule_rejects_a_near_miss(root: Path) -> None:
    scenario = Scenario(
        events=(ScenarioEvent("approved"),),
        latent_messages=(latent("XLAT02", "approved"),),
        rules=(
            ScenarioRule(rule_id="r1", event_id="approved", trigger="reply_exact",
                         thread_id=THREAD, exact_body="ship it"),
        ),
    )
    db = build(root, scenario)

    reply(db, "ship it when you can")
    assert events(db)["approved"] == "pending"

    reply(db, "  Ship It.  ")
    assert events(db)["approved"] == "activated", "case and trailing punctuation are not a miss"


def test_an_observation_rule_fires_on_what_the_agent_actually_received(root: Path) -> None:
    scenario = Scenario(
        events=(ScenarioEvent("saw_it"),),
        latent_messages=(latent("XLAT03", "saw_it"),),
        rules=(
            ScenarioRule(rule_id="r1", event_id="saw_it", trigger="observed",
                         observed_ids=("MSG007",)),
        ),
    )
    db = build(root, scenario)

    read(db, "C003")
    assert events(db)["saw_it"] == "pending", "MSG007 lives in C001, not C003"

    read(db, "C001")
    assert events(db)["saw_it"] == "activated"
    assert "XLAT03" in visible(db)


def test_a_rule_waits_for_the_events_it_depends_on(root: Path) -> None:
    scenario = Scenario(
        events=(ScenarioEvent("first"), ScenarioEvent("second")),
        latent_messages=(latent("XLAT04", "first"), latent("XLAT05", "second")),
        rules=(
            ScenarioRule(rule_id="r1", event_id="first", trigger="reply_exact",
                         thread_id=THREAD, exact_body="one"),
            ScenarioRule(rule_id="r2", event_id="second", trigger="reply_exact",
                         thread_id=THREAD, exact_body="two", requires_activated=("first",)),
        ),
    )
    db = build(root, scenario)

    reply(db, "two")
    assert events(db)["second"] == "pending", "the prerequisite has not happened yet"

    reply(db, "one")
    reply(db, "two")
    assert events(db)["second"] == "activated"


def test_a_dependency_only_rule_closes_in_the_same_call(root: Path) -> None:
    """`all_of` needs no trigger of its own, so it fires the moment it can."""
    scenario = Scenario(
        events=(ScenarioEvent("a"), ScenarioEvent("b"), ScenarioEvent("both")),
        latent_messages=(latent("XLAT06", "both"),),
        rules=(
            ScenarioRule(rule_id="r1", event_id="a", trigger="reply_exact",
                         thread_id=THREAD, exact_body="a"),
            ScenarioRule(rule_id="r2", event_id="b", trigger="reply_exact",
                         thread_id=THREAD, exact_body="b"),
            ScenarioRule(rule_id="r3", event_id="both", trigger="all_of",
                         requires_activated=("a", "b")),
        ),
    )
    db = build(root, scenario)

    reply(db, "a")
    assert events(db)["both"] == "pending"

    reply(db, "b")
    assert events(db)["both"] == "activated", "the closure must run in the call that completed it"
    assert "XLAT06" in visible(db)


def test_the_releasing_call_cannot_see_what_it_released(root: Path) -> None:
    scenario = Scenario(
        events=(ScenarioEvent("later"),),
        latent_messages=(latent("XLAT07", "later"),),
        rules=(
            ScenarioRule(rule_id="r1", event_id="later", trigger="reply_exact",
                         thread_id=THREAD, exact_body="go"),
        ),
    )
    db = build(root, scenario)

    result = reply(db, "go")
    assert "XLAT07" not in str(result), "the triggering observation must predate the release"
    assert "XLAT07" in visible(db)


def test_a_scheduled_event_lands_on_its_own_instant(root: Path) -> None:
    from slack_sim.clock import moment, slack_ts, START_US, STEP_US

    step = moment(20, 8, 0)
    scenario = Scenario(
        events=(ScenarioEvent("standup", scheduled_step=step),),
        latent_messages=(latent("XLAT08", "standup"),),
        rules=(
            ScenarioRule(rule_id="r1", event_id="standup", trigger="reply_exact",
                         thread_id=THREAD, exact_body="go"),
        ),
    )
    db = build(root, scenario)
    reply(db, "go")

    published = {m["message_id"]: m for m in export_state(db)["messages"]}["XLAT08"]
    assert published["ts"] == slack_ts(START_US + step * STEP_US)


def test_a_workspace_with_no_scenario_still_serves_slack(root: Path) -> None:
    db = build(root, Scenario())
    assert events(db) == {}
    posted = reply(db, "just a normal message")
    assert posted["id"] in visible(db), "the simulator must not need a scenario to work"


def test_activation_happens_once(root: Path) -> None:
    scenario = Scenario(
        events=(ScenarioEvent("once"),),
        latent_messages=(latent("XLAT09", "once"),),
        rules=(
            ScenarioRule(rule_id="r1", event_id="once", trigger="reply_exact",
                         thread_id=THREAD, exact_body="go"),
        ),
    )
    db = build(root, scenario)
    reply(db, "go")
    before = export_state(db)["scenario_events"][0]["activated_ts"]
    reply(db, "go")
    after = export_state(db)["scenario_events"][0]["activated_ts"]
    assert before == after
    assert sum(1 for m in export_state(db)["messages"] if m["message_id"] == "XLAT09") == 1


# ---------------------------------------------------------------------------
# Latent side-state: notifications and membership
# ---------------------------------------------------------------------------

def test_a_latent_notification_stays_invisible_until_its_event_fires(root: Path) -> None:
    from slack_sim.models import LatentNotification

    scenario = Scenario(
        events=(ScenarioEvent("answered"),),
        latent_messages=(latent("XLAT10", "answered", "here is the answer"),),
        latent_notifications=(
            LatentNotification("XNTF01", "answered", ACTOR, "mention", "XLAT10", "C001"),
        ),
        rules=(
            ScenarioRule(rule_id="r1", event_id="answered", trigger="reply_exact",
                         thread_id=THREAD, exact_body="go"),
        ),
    )
    db = build(root, scenario)

    before = execute_tool(db, "list_notifications", {}, ACTOR)
    assert all(n["notification_id"] != "XNTF01" for n in before["notifications"])

    reply(db, "go")
    after = execute_tool(db, "list_notifications", {"only_unread": True}, ACTOR)
    landed = [n for n in after["notifications"] if n["notification_id"] == "XNTF01"]
    assert landed and landed[0]["message_id"] == "XLAT10"


def test_a_latent_membership_change_lands_with_its_event(root: Path) -> None:
    from slack_sim.models import LatentMembership

    # C011 has an explicit roster that excludes U030, so an added membership
    # is observable rather than already true.
    scenario = Scenario(
        events=(ScenarioEvent("staffed"),),
        latent_memberships=(LatentMembership("XMEM01", "staffed", "C011", "U030", "member"),),
        rules=(
            ScenarioRule(rule_id="r1", event_id="staffed", trigger="reply_exact",
                         thread_id=THREAD, exact_body="go"),
        ),
    )
    db = build(root, scenario)
    execute_tool(db, "join_channel", {"channel_id": "C011"}, ACTOR)

    def members() -> set:
        return {m["user_id"] for m in
                execute_tool(db, "list_channel_members", {"channel_id": "C011"}, ACTOR)["members"]}

    assert "U030" not in members()
    reply(db, "go")
    assert "U030" in members()


def test_latent_side_state_leaks_through_no_read_path(root: Path) -> None:
    from slack_sim.models import LatentMembership, LatentNotification
    from slack_sim.service import export_state

    scenario = Scenario(
        events=(ScenarioEvent("later"),),
        latent_messages=(latent("XLAT11", "later", "unreleased body text"),),
        latent_notifications=(
            LatentNotification("XNTF02", "later", ACTOR, "mention", "XLAT11", "C001"),
        ),
        latent_memberships=(LatentMembership("XMEM02", "later", "C011", "U030", "member"),),
        rules=(
            ScenarioRule(rule_id="r1", event_id="later", trigger="reply_exact",
                         thread_id=THREAD, exact_body="go"),
        ),
    )
    db = build(root, scenario)

    state = export_state(db)
    assert all(n["notification_id"] != "XNTF02" for n in state["notifications"])
    assert not any(m["channel_id"] == "C011" and m["user_id"] == "U030"
                   for m in state["memberships"])
    assert "unreleased body text" not in str(execute_tool(
        db, "search_messages", {"query": "unreleased"}, ACTOR))
    listing = execute_tool(db, "get_channel_messages", {"channel_id": "C001"}, ACTOR)
    assert all(m["id"] != "XLAT11" for m in listing["messages"])
    root_ts = [m for m in listing["messages"] if m["id"] == THREAD][0]["ts"]
    thread = execute_tool(
        db, "get_thread_replies", {"channel_id": "C001", "thread_ts": root_ts}, ACTOR
    )
    visible_replies = [m for m in thread["messages"] if m["id"] != THREAD]
    assert all(m["id"] != "XLAT11" for m in visible_replies)
    counted = [m for m in thread["messages"] if m["id"] == THREAD][0]["reply_count"]
    assert counted == len(visible_replies), (
        "an unreleased reply must not inflate reply_count"
    )
    assert thread["messages"][-1]["ts"] <= root_ts or all(
        m["id"] != "XLAT11" for m in thread["messages"]
    ), "an unreleased reply must not become the latest reply"

    reply(db, "go")
    after = execute_tool(
        db, "get_thread_replies", {"channel_id": "C001", "thread_ts": root_ts}, ACTOR
    )
    assert any(m["id"] == "XLAT11" for m in after["messages"]), "and it lands once released"


# ---------------------------------------------------------------------------
# channel_joined, a trigger with no message behind it
# ---------------------------------------------------------------------------

def test_joining_a_channel_can_activate_an_event(root: Path) -> None:
    scenario = Scenario(
        events=(ScenarioEvent("noticed"),),
        latent_messages=(
            LatentMessage("XLAT12", "noticed", "C011", "U011", "welcome aboard"),
        ),
        rules=(
            ScenarioRule(rule_id="r1", event_id="noticed", trigger="channel_joined",
                         channel_id="C011"),
        ),
    )
    db = build(root, scenario)
    assert events(db)["noticed"] == "pending"

    # C013 is another channel the actor is not in; joining it is not this rule.
    execute_tool(db, "join_channel", {"channel_id": "C013"}, ACTOR)
    assert events(db)["noticed"] == "pending", "joining another channel is not this trigger"

    execute_tool(db, "join_channel", {"channel_id": "C011"}, ACTOR)
    assert events(db)["noticed"] == "activated"
    assert "XLAT12" in visible(db)


def main() -> None:
    tests = sorted(
        (value for name, value in globals().items()
         if name.startswith("test_") and callable(value)),
        key=lambda fn: fn.__name__,
    )
    with tempfile.TemporaryDirectory() as directory:
        for index, test in enumerate(tests):
            test(Path(directory) / str(index))
            print(f"  {test.__name__}: ok")
    print("scenario engine: ok")


if __name__ == "__main__":
    main()
