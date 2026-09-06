#!/usr/bin/env python3
"""Determinism and monotonicity guarantees for the virtual clock.

The contract these checks pin down:

  virtual_time == START + (successful mutating tool calls) * STEP

Reads never move the clock, rejected calls never move the clock, and every
successful mutation moves it by exactly one step. That makes every stored
timestamp a pure function of the world-mutation sequence, so an agent that
browses more before acting still produces byte-identical timestamps.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import sqlite3

from slack_sim.clock import STEP_US, parse_slack_ts
from slack_sim.environment import SlackIncidentEnvironment
from slack_sim.identity import LOGGED_IN_USER
from slack_sim.service import TOOL_NAMES, execute_tool, export_state
from slack_sim.sqlite_common import connect

COOKIE = "clock-test"

# Every tool that only observes the world. These must never move the clock.
READ_ONLY_TOOLS = {
    "list_users",
    "list_chats",
    "list_user_groups",
    "list_channels",
    "search_messages",
    "search_users",
    "search_channels",
    "get_channel_messages",
    "get_thread_replies",
    "list_channel_members",
    "list_followed_threads",
    "list_pins",
    "list_saved_items",
    "list_notifications",
}

READ_ONLY_CALLS = [
    ("list_users", {}),
    ("list_chats", {}),
    ("list_user_groups", {}),
    ("list_channels", {}),
    ("search_messages", {"query": "Acme"}),
    ("get_channel_messages", {"channel_id": "C019"}),
    ("get_thread_replies", {"channel_id": "C019", "thread_ts": "1786035600.000000"}),
    ("search_users", {"query": "priya"}),
    ("search_channels", {"query": "identity"}),
    ("list_channel_members", {"channel_id": "C019"}),
    ("list_followed_threads", {}),
    ("list_pins", {"conversation_id": "C019"}),
    ("list_saved_items", {}),
    ("list_notifications", {}),
]


def clock_us(environment: SlackIncidentEnvironment) -> int:
    return parse_slack_ts(export_state(environment.db_path)["virtual_time"])


def fresh(root: Path) -> SlackIncidentEnvironment:
    environment = SlackIncidentEnvironment(root / "slack.db", root / "seed.sql")
    # This suite runs well past the default episode budget on purpose; it is
    # exercising the clock, not termination.
    environment.reset(
        session_cookie=COOKIE,
        user_instruction="Exercise the clock.",
        max_turns=10_000,
    )
    return environment


def mutation_script(environment: SlackIncidentEnvironment) -> list[tuple[str, dict]]:
    """One successful call of every mutating tool, in dependency order."""
    channel_id = environment.step(
        COOKIE, "create_channel", {"name": "clock-drill", "is_private": False}
    )["result"]["id"]
    message_id = environment.step(
        COOKIE, "post_message", {"channel_id": channel_id, "body": "anchor"}
    )["result"]["id"]
    group_id = environment.step(
        COOKIE, "create_group", {"name": "Clock Group", "participants": ["U003"]}
    )["result"]["id"]
    dm_id = environment.step(COOKIE, "create_dm_message", {"recipient_id": "U008"})["result"]["id"]
    # A reply of our own to delete later, and a notification to mark read. The
    # notification comes from someone else mentioning the actor, since your own
    # messages never notify you.
    reply_id = environment.step(
        COOKIE, "reply_to_thread", {"thread_parent_id": message_id, "body": "to be deleted"}
    )["result"]["id"]
    execute_tool(
        environment.db_path,
        "tag_people",
        {"channel_id": "C003", "user_ids": [LOGGED_IN_USER.user_id], "body": "ping"},
        "U003",
    )
    notification_id = execute_tool(
        environment.db_path, "list_notifications", {"only_unread": True},
        LOGGED_IN_USER.user_id,
    )["notifications"][0]["notification_id"]
    return [
        ("reply_to_thread", {"thread_parent_id": message_id, "body": "reply"}),
        ("edit_message", {"message_id": message_id, "body": "anchor revised"}),
        ("add_reaction", {"message_id": message_id, "emoji": "eyes"}),
        ("tag_people", {"channel_id": channel_id, "user_ids": ["U003"], "body": "look"}),
        ("tag_here", {"channel_id": channel_id, "body": "here now"}),
        ("tag_everyone", {"channel_id": channel_id, "body": "all hands"}),
        ("tag_user_group", {"channel_id": channel_id, "user_group_id": "S002", "body": "oncall"}),
        ("invite_to_channel", {"channel_id": channel_id, "user_ids": ["U003"]}),
        ("edit_channel_name", {"channel_id": channel_id, "new_name": "clock-drill-2"}),
        ("edit_group_chat_name", {"group_id": group_id, "new_name": "Clock Group 2"}),
        ("send_group_message", {"group_id": group_id, "body": "group ping"}),
        ("add_group_chat_participants", {"group_id": group_id, "user_ids": ["U004"]}),
        ("send_dm_message", {"chat_id": dm_id, "body": "direct ping"}),
        (
            "edit_display_name",
            {"user_id": LOGGED_IN_USER.user_id, "new_display_name": "Ben O."},
        ),
        ("mark_conversation_read", {"conversation_id": "C003"}),
        # Reactions, pins, saved items and follows: each add/remove pair costs
        # one step in each direction.
        ("remove_reaction", {"message_id": message_id, "emoji": "eyes"}),
        ("pin_message", {"message_id": message_id}),
        ("unpin_message", {"message_id": message_id}),
        ("save_message", {"message_id": message_id}),
        ("unsave_message", {"message_id": message_id}),
        ("unfollow_thread", {"thread_parent_id": message_id}),
        ("follow_thread", {"thread_parent_id": message_id}),
        # Presence, status, membership and lifecycle.
        ("set_presence", {"presence": "away"}),
        ("set_status", {"status_text": "running the clock drill"}),
        ("join_channel", {"channel_id": "C011"}),
        ("leave_channel", {"channel_id": "C011"}),
        ("remove_from_channel", {"channel_id": channel_id, "user_ids": ["U003"]}),
        ("mark_notification_read", {"notification_id": notification_id}),
        ("delete_message", {"message_id": reply_id}),
        # Archiving last: it closes the channel to everything above.
        ("archive_channel", {"channel_id": channel_id}),
    ]


def test_read_only_tools_never_advance_the_clock(root: Path) -> None:
    environment = fresh(root)
    for tool_name, payload in READ_ONLY_CALLS:
        before = clock_us(environment)
        result = environment.step(COOKIE, tool_name, payload)
        assert result["ok"], f"{tool_name} should succeed: {result}"
        after = clock_us(environment)
        assert after == before, (
            f"read-only tool {tool_name} moved the clock by "
            f"{(after - before) / STEP_US} steps; reads must not advance time"
        )


def test_each_successful_mutation_advances_exactly_one_step(root: Path) -> None:
    environment = fresh(root)
    covered = {"create_channel", "post_message", "create_group", "create_dm_message"}

    # The four bootstrap calls are themselves mutations; check them in isolation.
    for tool_name, payload in [
        ("create_channel", {"name": "bootstrap-drill", "is_private": False}),
        ("post_message", {"channel_id": "C003", "body": "bootstrap"}),
        ("create_group", {"name": "Bootstrap Group", "participants": ["U003"]}),
        ("create_dm_message", {"recipient_id": "U009"}),
    ]:
        before = clock_us(environment)
        result = environment.step(COOKIE, tool_name, payload)
        assert result["ok"], f"{tool_name} should succeed: {result}"
        after = clock_us(environment)
        assert after - before == STEP_US, (
            f"{tool_name} advanced the clock by {(after - before) / STEP_US} steps, expected 1"
        )

    for tool_name, payload in mutation_script(environment):
        covered.add(tool_name)
        before = clock_us(environment)
        result = environment.step(COOKIE, tool_name, payload)
        assert result["ok"], f"{tool_name} should succeed: {result}"
        after = clock_us(environment)
        assert after - before == STEP_US, (
            f"{tool_name} advanced the clock by {(after - before) / STEP_US} steps, expected 1"
        )

    assert covered == set(TOOL_NAMES) - READ_ONLY_TOOLS, (
        "every mutating tool must be covered; missing "
        f"{set(TOOL_NAMES) - READ_ONLY_TOOLS - covered}"
    )


def test_rejected_calls_never_advance_the_clock(root: Path) -> None:
    environment = fresh(root)
    environment.step(COOKIE, "create_channel", {"name": "reject-drill", "is_private": False})
    failures = [
        ("post_message", {"channel_id": "C999", "body": "no such channel"}),
        ("tag_everyone", {"channel_id": "C008", "body": "not the owner"}),
        ("edit_message", {"message_id": "MSG001", "body": "not my message"}),
        ("create_channel", {"name": "reject-drill", "is_private": False}),
        ("send_dm_message", {"body": "no target"}),
        ("add_reaction", {"message_id": "NO-SUCH-MESSAGE", "emoji": "eyes"}),
        # A valid channel the actor belongs to, rejected purely on the unknown
        # user, so this covers a different path from the C999 case above.
        ("invite_to_channel", {"channel_id": "C002", "user_ids": ["U005", "UZZZ"]}),
    ]
    for tool_name, payload in failures:
        before = clock_us(environment)
        result = environment.step(COOKIE, tool_name, payload)
        assert not result["ok"], f"{tool_name} should have been rejected: {result}"
        after = clock_us(environment)
        assert after == before, (
            f"rejected {tool_name} moved the clock by {(after - before) / STEP_US} steps"
        )


def test_rejected_invite_leaves_no_partial_membership(root: Path) -> None:
    environment = fresh(root)
    channel_id = environment.step(
        COOKIE, "create_channel", {"name": "atomic-drill", "is_private": False}
    )["result"]["id"]
    result = environment.step(
        COOKIE, "invite_to_channel", {"channel_id": channel_id, "user_ids": ["U005", "UZZZ"]}
    )
    assert not result["ok"] and result["error"]["code"] == "user_not_found"
    memberships = export_state(environment.db_path)["memberships"]
    assert not any(
        m["channel_id"] == channel_id and m["user_id"] == "U005" for m in memberships
    ), "a rejected invite must not add the users it managed to validate first"


def test_timestamps_are_unaffected_by_interleaved_reads(root: Path) -> None:
    """The headline property: browsing must not perturb world timestamps."""

    def run(reads_between: int) -> list[str]:
        with tempfile.TemporaryDirectory(prefix="slack-clock-") as temp_dir:
            environment = fresh(Path(temp_dir))
            stamps = []
            for body in ("first", "second", "third"):
                for _ in range(reads_between):
                    for tool_name, payload in READ_ONLY_CALLS:
                        environment.step(COOKIE, tool_name, payload)
                posted = environment.step(
                    COOKIE, "post_message", {"channel_id": "C003", "body": body}
                )
                stamps.append(posted["result"]["message"]["ts"])
            return stamps

    baseline = run(0)
    for reads_between in (1, 3):
        assert run(reads_between) == baseline, (
            f"interleaving {reads_between} rounds of read-only calls changed "
            f"message timestamps: {run(reads_between)} != {baseline}"
        )


def test_one_instant_per_call_is_shared_by_every_record_it_writes(root: Path) -> None:
    environment = fresh(root)
    before = clock_us(environment)
    result = environment.step(COOKIE, "send_dm_message", {"recipient_id": "U010", "body": "hi"})
    assert result["ok"], result
    after = clock_us(environment)
    assert after - before == STEP_US, (
        "send_dm_message that also creates the DM must still be a single tick, "
        f"got {(after - before) / STEP_US}"
    )
    message = result["result"]["message"]
    chat = next(
        c
        for c in export_state(environment.db_path)["chats"]
        if c["chat_id"] == message["channel_id"]
    )
    assert chat["created_ts"] == message["ts"], (
        "the DM and its first message are written by one call and must share its instant"
    )


def test_clock_is_monotonic_across_a_mixed_workload(root: Path) -> None:
    environment = fresh(root)
    observed = [clock_us(environment)]
    script = mutation_script(environment)
    for index, (tool_name, payload) in enumerate(script):
        for read_tool, read_payload in READ_ONLY_CALLS:
            environment.step(COOKIE, read_tool, read_payload)
        environment.step(COOKIE, tool_name, payload)
        environment.step(COOKIE, "post_message", {"channel_id": "C999", "body": "rejected"})
        observed.append(clock_us(environment))
    for earlier, later in zip(observed, observed[1:]):
        assert later > earlier, f"clock went backwards or stalled: {earlier} -> {later}"


def test_history_timestamps_never_exceed_world_time(root: Path) -> None:
    environment = fresh(root)
    environment.step(COOKIE, "list_users", {})
    environment.step(COOKIE, "post_message", {"channel_id": "C003", "body": "history check"})
    now_us = clock_us(environment)
    stamps = [parse_slack_ts(event["ts"]) for event in environment.state(COOKIE)["history"]]
    assert stamps == sorted(stamps), "transcript timestamps must be non-decreasing"
    assert max(stamps) <= now_us, "transcript rows must never be stamped ahead of world time"


def test_clock_cannot_be_moved_backwards_or_stalled(root: Path) -> None:
    """Monotonicity is enforced by the schema, not merely by convention."""
    fresh(root)
    for label, delta in (("backwards", "- 1"), ("stalled", "+ 0")):
        with connect(root / "slack.db") as connection:
            try:
                connection.execute(
                    f"UPDATE virtual_clock SET current_us = current_us {delta} "
                    "WHERE clock_id = 1"
                )
            except sqlite3.IntegrityError:
                continue
            raise AssertionError(f"the database accepted a {label} virtual clock")


def test_connection_is_closed_when_its_tool_call_ends(root: Path) -> None:
    """A leaked handle goes on holding SQLite locks until garbage collection."""
    fresh(root)
    with connect(root / "slack.db") as connection:
        connection.execute("SELECT 1").fetchone()
    try:
        connection.execute("SELECT 1").fetchone()
    except sqlite3.ProgrammingError:
        return
    raise AssertionError("connection outlived its tool call instead of being closed")


def test_each_tool_call_gets_its_own_action_instant(root: Path) -> None:
    """The cached instant is per call, so the next call must tick again."""
    environment = fresh(root)
    first = environment.step(COOKIE, "post_message", {"channel_id": "C003", "body": "one"})
    second = environment.step(COOKIE, "post_message", {"channel_id": "C003", "body": "two"})
    assert (
        parse_slack_ts(second["result"]["message"]["ts"])
        - parse_slack_ts(first["result"]["message"]["ts"])
        == STEP_US
    ), "consecutive calls must land on consecutive instants, not share one"


def test_mutating_tools_advance_even_when_they_change_nothing(root: Path) -> None:
    """The delta depends on the tool called, never on hidden world state.

    Idempotent calls -- opening a DM that already exists, inviting a user who
    is already a member -- still consume an instant. Otherwise the clock leaks
    information about the workspace into what should be a property of the call.
    """
    environment = fresh(root)
    channel_id = environment.step(
        COOKIE, "create_channel", {"name": "noop-drill", "is_private": False}
    )["result"]["id"]
    group_id = environment.step(
        COOKIE, "create_group", {"name": "Noop Group", "participants": ["U003"]}
    )["result"]["id"]
    environment.step(COOKIE, "invite_to_channel", {"channel_id": channel_id, "user_ids": ["U003"]})

    # D002 already pairs the actor with U003, so this creates nothing.
    no_ops = [
        ("create_dm_message", {"recipient_id": "U003"}, "id"),
        ("invite_to_channel", {"channel_id": channel_id, "user_ids": ["U003"]}, "added_user_ids"),
        ("add_group_chat_participants", {"group_id": group_id, "user_ids": ["U003"]}, "added_user_ids"),
    ]
    for tool_name, payload, key in no_ops:
        before = clock_us(environment)
        result = environment.step(COOKIE, tool_name, payload)
        assert result["ok"], f"{tool_name} should succeed: {result}"
        after = clock_us(environment)
        assert after - before == STEP_US, (
            f"{tool_name} wrote nothing and advanced the clock by "
            f"{(after - before) / STEP_US} steps; a mutating call costs one instant "
            "whether or not the world happened to already satisfy it"
        )


def main() -> None:
    tests = [
        test_read_only_tools_never_advance_the_clock,
        test_each_successful_mutation_advances_exactly_one_step,
        test_rejected_calls_never_advance_the_clock,
        test_rejected_invite_leaves_no_partial_membership,
        test_timestamps_are_unaffected_by_interleaved_reads,
        test_one_instant_per_call_is_shared_by_every_record_it_writes,
        test_clock_is_monotonic_across_a_mixed_workload,
        test_history_timestamps_never_exceed_world_time,
        test_clock_cannot_be_moved_backwards_or_stalled,
        test_connection_is_closed_when_its_tool_call_ends,
        test_each_tool_call_gets_its_own_action_instant,
        test_mutating_tools_advance_even_when_they_change_nothing,
    ]
    for test in tests:
        with tempfile.TemporaryDirectory(prefix="slack-clock-") as temp_dir:
            test(Path(temp_dir))
        print(f"  {test.__name__}: ok")
    print("virtual clock: ok")


if __name__ == "__main__":
    main()
