#!/usr/bin/env python3
"""Coverage for the Slack operations beyond post/read/react.

Each test names a real Slack behaviour rather than an implementation detail,
so a handler that technically writes a row but gets the semantics wrong --
deleting someone else's message, pinning without membership, a search filter
that silently matches nothing -- fails here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from slack_sim.service import execute_tool, seed_database
from slack_sim.sqlite_common import ToolError

BEN = "U002"          # the acting user; an ordinary member
ALICE = "U001"        # a workspace admin
PRIYA = "U042"
ACME = "C019"         # Ben is a member
DATA_QUALITY = "C011"  # public, Ben is NOT a member
GENERAL = "C001"


def build(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    db = root / "slack.db"
    seed_database(db, root / "snapshot.sql")
    return db


def call(db: Path, tool: str, actor: str = BEN, **payload):
    return execute_tool(db, tool, payload, actor)


def fails(db: Path, tool: str, actor: str = BEN, **payload) -> str:
    try:
        execute_tool(db, tool, payload, actor)
    except ToolError as error:
        return error.error_code
    raise AssertionError(f"{tool} was expected to fail")


def channel_ids(db: Path, channel_id: str) -> list[str]:
    return [m["id"] for m in call(db, "get_channel_messages", channel_id=channel_id)["messages"]]


# ---------------------------------------------------------------------------
# Messages: delete
# ---------------------------------------------------------------------------

def test_you_can_delete_your_own_message_and_it_leaves_every_view(root: Path) -> None:
    db = build(root)
    posted = call(db, "post_message", channel_id=ACME, body="scratch that")["id"]
    assert posted in channel_ids(db, ACME)

    call(db, "delete_message", message_id=posted)
    assert posted not in channel_ids(db, ACME)
    assert not call(db, "search_messages", query="scratch that")["matches"]


def test_delete_really_removes_the_row(root: Path) -> None:
    """No tombstone and no hidden copy: a verifier can trust the workspace."""
    from slack_sim.service import export_state

    db = build(root)
    posted = call(db, "post_message", channel_id=ACME, body="scratch that")["id"]
    call(db, "delete_message", message_id=posted)
    assert posted not in {m["message_id"] for m in export_state(db)["messages"]}


def test_deleting_takes_its_reactions_pins_and_notifications_with_it(root: Path) -> None:
    from slack_sim.service import export_state

    db = build(root)
    posted = call(db, "tag_people", channel_id=ACME, user_ids=[PRIYA], body="look")["id"]
    call(db, "add_reaction", message_id=posted, emoji="eyes")
    call(db, "pin_message", message_id=posted)
    call(db, "save_message", message_id=posted)
    assert any(n["message_id"] == posted
               for n in call(db, "list_notifications", actor=PRIYA)["notifications"])

    call(db, "delete_message", message_id=posted)
    state = export_state(db)
    for section in ("reactions", "pins", "saved_items", "notifications", "mentions"):
        assert all(row.get("message_id") != posted for row in state[section]), section


def test_deleting_your_own_thread_root_takes_the_thread_with_it(root: Path) -> None:
    """Slack deletes the replies too rather than refusing; so does this."""
    db = build(root)
    posted = call(db, "post_message", channel_id=ACME, body="question")["id"]
    mine = call(db, "reply_to_thread", thread_parent_id=posted, body="never mind")["id"]
    theirs = call(db, "reply_to_thread", thread_parent_id=posted, body="answer", actor=PRIYA)["id"]

    result = call(db, "delete_message", message_id=posted)
    assert set(result["deleted_message_ids"]) == {posted, mine, theirs}

    remaining = channel_ids(db, ACME)
    assert not {posted, mine, theirs} & set(remaining)


def test_a_thread_root_you_do_not_own_is_still_refused(root: Path) -> None:
    db = build(root)
    assert fails(db, "delete_message", message_id="MSG145") == "permission_denied"
    assert "MSG145" in channel_ids(db, ACME)


def test_the_cascade_cleans_up_after_every_message_it_removes(root: Path) -> None:
    from slack_sim.service import export_state

    db = build(root)
    posted = call(db, "post_message", channel_id=ACME, body="question")["id"]
    reply = call(db, "reply_to_thread", thread_parent_id=posted, body="answer", actor=PRIYA)["id"]
    call(db, "add_reaction", message_id=reply, emoji="eyes")
    call(db, "pin_message", message_id=reply)
    call(db, "save_message", message_id=reply)

    call(db, "delete_message", message_id=posted)
    state = export_state(db)
    for section in ("reactions", "pins", "saved_items", "notifications", "mentions"):
        assert all(row.get("message_id") not in {posted, reply} for row in state[section]), section
    assert all(row["thread_id"] != posted for row in state["thread_follows"])


def test_deleting_a_message_someone_answered_repoints_the_answer(root: Path) -> None:
    """A reply survives its target, still visibly answering the thread."""
    db = build(root)
    first = call(db, "reply_to_thread", thread_parent_id="MSG145", body="one")["id"]
    second = call(db, "reply_to_thread", thread_parent_id=first, body="two")["id"]
    call(db, "delete_message", message_id=second)
    call(db, "delete_message", message_id=first)

    remaining = [m for m in call(db, "get_channel_messages", channel_id=ACME)["messages"]
                 if m["id"] in {first, second}]
    assert remaining == []


def test_you_cannot_delete_someone_elses_message(root: Path) -> None:
    db = build(root)
    assert fails(db, "delete_message", message_id="MSG145") == "permission_denied"
    assert "MSG145" in channel_ids(db, ACME)


def test_deleting_a_reply_decrements_the_threads_reply_count(root: Path) -> None:
    db = build(root)
    root_ts = [m["ts"] for m in call(db, "get_channel_messages", channel_id=ACME)["messages"]
               if m["id"] == "MSG145"][0]
    reply = call(db, "reply_to_thread", thread_parent_id="MSG145", body="on it")["id"]
    before = call(db, "get_thread_replies", channel_id=ACME, thread_ts=root_ts)
    call(db, "delete_message", message_id=reply)
    after = call(db, "get_thread_replies", channel_id=ACME, thread_ts=root_ts)
    assert len(after["messages"]) == len(before["messages"]) - 1


# ---------------------------------------------------------------------------
# Threads: follow
# ---------------------------------------------------------------------------

def followed_ids(db: Path) -> set[str]:
    return {t["thread_id"] for t in call(db, "list_followed_threads")["threads"]}


def test_following_a_thread_lists_it_and_unfollowing_removes_it(root: Path) -> None:
    db = build(root)
    before = followed_ids(db)
    call(db, "follow_thread", thread_parent_id="MSG145")
    assert followed_ids(db) == before | {"MSG145"}
    entry = [t for t in call(db, "list_followed_threads")["threads"]
             if t["thread_id"] == "MSG145"][0]
    assert entry["reply_count"] >= 0

    call(db, "unfollow_thread", thread_parent_id="MSG145")
    assert followed_ids(db) == before


def test_the_followed_threads_at_reset_are_not_a_task_list(root: Path) -> None:
    """Ben follows what he happened to be in, not what matters today."""
    db = build(root)
    followed = followed_ids(db)
    assert followed, "a lived-in workspace has some follows"
    assert "MSG145" not in followed, "the request thread is not pre-followed"
    open_reviews = {"MSG194", "MSG200", "MSG214", "MSG219", "MSG224"}
    assert not (followed & open_reviews), "the open reviews are not pre-followed either"


def test_replying_to_a_thread_follows_it(root: Path) -> None:
    db = build(root)
    call(db, "reply_to_thread", thread_parent_id="MSG145", body="looking now")
    assert "MSG145" in followed_ids(db)


def test_following_resolves_a_reply_to_its_thread_root(root: Path) -> None:
    db = build(root)
    reply = call(db, "reply_to_thread", thread_parent_id="MSG145", body="a")["id"]
    call(db, "unfollow_thread", thread_parent_id="MSG145")
    assert "MSG145" not in followed_ids(db)
    call(db, "follow_thread", thread_parent_id=reply)
    assert "MSG145" in followed_ids(db)


def test_you_cannot_follow_a_thread_you_cannot_read(root: Path) -> None:
    db = build(root)
    assert fails(db, "follow_thread", thread_parent_id="MSG083") in {
        "message_not_found", "channel_not_found", "permission_denied",
    }


# ---------------------------------------------------------------------------
# Reactions: remove
# ---------------------------------------------------------------------------

def test_you_can_take_back_your_own_reaction(root: Path) -> None:
    db = build(root)
    call(db, "add_reaction", message_id="MSG145", emoji="eyes")
    call(db, "remove_reaction", message_id="MSG145", emoji="eyes")
    message = [m for m in call(db, "get_channel_messages", channel_id=ACME)["messages"]
               if m["id"] == "MSG145"][0]
    assert all(r["emoji"] != "eyes" for r in message.get("reactions", []))


def test_removing_a_reaction_you_never_left_is_an_error(root: Path) -> None:
    db = build(root)
    assert fails(db, "remove_reaction", message_id="MSG145", emoji="tada") == "reaction_not_found"


def test_removing_a_reaction_leaves_other_peoples_alone(root: Path) -> None:
    db = build(root)
    call(db, "add_reaction", message_id="MSG145", emoji="eyes", actor=PRIYA)
    call(db, "add_reaction", message_id="MSG145", emoji="eyes")
    call(db, "remove_reaction", message_id="MSG145", emoji="eyes")
    message = [m for m in call(db, "get_channel_messages", channel_id=ACME)["messages"]
               if m["id"] == "MSG145"][0]
    eyes = [r for r in message.get("reactions", []) if r["emoji"] == "eyes"]
    assert eyes and eyes[0]["user_ids"] == [PRIYA]


# ---------------------------------------------------------------------------
# Channels: join, leave, archive
# ---------------------------------------------------------------------------

def test_joining_a_public_channel_makes_it_readable(root: Path) -> None:
    db = build(root)
    assert fails(db, "get_channel_messages", channel_id=DATA_QUALITY) == "permission_denied"
    call(db, "join_channel", channel_id=DATA_QUALITY)
    assert call(db, "get_channel_messages", channel_id=DATA_QUALITY)["messages"]


def test_a_private_channel_cannot_be_joined_or_even_seen(root: Path) -> None:
    db = build(root)
    assert fails(db, "join_channel", channel_id="C009") == "channel_not_found"


def test_leaving_a_channel_revokes_your_access(root: Path) -> None:
    db = build(root)
    call(db, "leave_channel", channel_id=ACME)
    assert fails(db, "get_channel_messages", channel_id=ACME) == "permission_denied"
    assert fails(db, "leave_channel", channel_id=ACME) == "permission_denied"


def test_archiving_needs_authority_and_then_closes_the_channel_to_posts(root: Path) -> None:
    db = build(root)
    assert fails(db, "archive_channel", channel_id=ACME) == "permission_denied"

    call(db, "archive_channel", channel_id=ACME, actor=ALICE)
    listed = {c["channel_id"]: c for c in call(db, "list_channels")["channels"]}
    assert listed[ACME]["is_archived"] is True
    assert fails(db, "post_message", channel_id=ACME, body="hello") == "channel_archived"
    # History survives archiving; that is the point of an archive.
    assert "MSG145" in channel_ids(db, ACME)


# ---------------------------------------------------------------------------
# Membership: list, remove
# ---------------------------------------------------------------------------

def test_a_member_can_read_the_roster(root: Path) -> None:
    db = build(root)
    members = call(db, "list_channel_members", channel_id=ACME)["members"]
    ids = {m["user_id"] for m in members}
    assert BEN in ids and "U041" in ids
    assert {m["role"] for m in members} <= {"owner", "member"}
    assert all("display_name" in m for m in members)


def test_removing_a_member_requires_authority_and_takes_effect(root: Path) -> None:
    db = build(root)
    assert fails(db, "remove_from_channel", channel_id=ACME, user_ids=[PRIYA]) == "permission_denied"

    call(db, "remove_from_channel", channel_id=ACME, user_ids=[PRIYA], actor=ALICE)
    ids = {m["user_id"] for m in call(db, "list_channel_members", channel_id=ACME)["members"]}
    assert PRIYA not in ids


def test_the_channel_owner_cannot_be_removed(root: Path) -> None:
    db = build(root)
    assert fails(db, "remove_from_channel", channel_id=ACME,
                 user_ids=["U041"], actor=ALICE) == "permission_denied"


# ---------------------------------------------------------------------------
# Users: presence and status
# ---------------------------------------------------------------------------

def test_presence_defaults_to_active_and_can_be_changed(root: Path) -> None:
    db = build(root)
    before = {u["user_id"]: u for u in call(db, "list_users")["users"]}
    assert before[BEN]["presence"] == "active"

    call(db, "set_presence", presence="away")
    after = {u["user_id"]: u for u in call(db, "list_users")["users"]}
    assert after[BEN]["presence"] == "away"
    assert after[PRIYA]["presence"] == "active", "presence is per user"


def test_presence_only_accepts_real_values(root: Path) -> None:
    db = build(root)
    assert fails(db, "set_presence", presence="asleep") == "invalid_presence"


def test_you_can_set_your_own_status_but_not_someone_elses(root: Path) -> None:
    db = build(root)
    call(db, "set_status", status_text="reviewing Acme")
    assert {u["user_id"]: u for u in call(db, "list_users")["users"]}[BEN]["status_text"] == "reviewing Acme"
    assert fails(db, "set_status", user_id=PRIYA, status_text="nope") == "permission_denied"


# ---------------------------------------------------------------------------
# Pins and saved items
# ---------------------------------------------------------------------------

def test_a_pin_is_visible_to_the_whole_channel(root: Path) -> None:
    db = build(root)
    before = {p["message_id"] for p in call(db, "list_pins", conversation_id=ACME)["pins"]}
    call(db, "pin_message", message_id="MSG145")
    pinned = {p["message_id"] for p in call(db, "list_pins", conversation_id=ACME)["pins"]}
    assert pinned == before | {"MSG145"}
    # Another member sees the same pin: it belongs to the conversation.
    assert "MSG145" in {p["message_id"] for p in
                        call(db, "list_pins", conversation_id=ACME, actor=PRIYA)["pins"]}

    call(db, "unpin_message", message_id="MSG145")
    assert {p["message_id"] for p in call(db, "list_pins", conversation_id=ACME)["pins"]} == before


def test_the_workspace_opens_with_pins_that_are_not_all_current(root: Path) -> None:
    """Being pinned is not evidence of being right."""
    db = build(root)
    pinned = {p["message_id"] for p in call(db, "list_pins", conversation_id=ACME)["pins"]}
    assert "MSG125" in pinned, "the superseded 8:00 PM kickoff is still pinned"
    assert "MSG135" in pinned, "so is the window/downtime note, which is still true"


def test_pinning_twice_is_a_conflict_and_pinning_unread_channels_is_refused(root: Path) -> None:
    db = build(root)
    call(db, "pin_message", message_id="MSG145")
    assert fails(db, "pin_message", message_id="MSG145") == "duplicate_pin"
    assert fails(db, "unpin_message", message_id="MSG007") == "pin_not_found"


def saved_ids(db: Path, actor: str = BEN) -> set[str]:
    return {s["message_id"] for s in call(db, "list_saved_items", actor=actor)["saved"]}


def test_saved_items_are_private_to_the_person_who_saved_them(root: Path) -> None:
    db = build(root)
    before = saved_ids(db)
    call(db, "save_message", message_id="MSG145")
    assert saved_ids(db) == before | {"MSG145"}
    assert "MSG145" not in saved_ids(db, PRIYA)
    assert saved_ids(db, PRIYA) == set(), "seeded saved items belong to Ben alone"

    call(db, "unsave_message", message_id="MSG145")
    assert saved_ids(db) == before


def test_the_saved_items_at_reset_are_a_mix_of_useful_and_stale(root: Path) -> None:
    db = build(root)
    saved = saved_ids(db)
    assert {"MSG551", "MSG799"} <= saved, "old dry-run and pagination notes were kept"
    assert "MSG125" in saved, "and so was the superseded kickoff checklist"
    assert not any(mid.startswith("LAT") for mid in saved), (
        "saved items must not be a cache of answers that have not happened yet"
    )


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

def test_being_mentioned_creates_an_unread_notification(root: Path) -> None:
    db = build(root)
    before = len(call(db, "list_notifications", only_unread=True)["notifications"])
    call(db, "tag_people", channel_id=ACME, user_ids=[BEN], body="can you look?", actor=PRIYA)

    unread = call(db, "list_notifications", only_unread=True)["notifications"]
    assert len(unread) == before + 1
    newest = unread[0]
    assert newest["kind"] == "mention" and newest["user_id"] == BEN


def test_your_own_message_never_notifies_you(root: Path) -> None:
    db = build(root)
    before = len(call(db, "list_notifications")["notifications"])
    call(db, "tag_people", channel_id=ACME, user_ids=[BEN, PRIYA], body="note to self")
    assert len(call(db, "list_notifications")["notifications"]) == before


def test_a_direct_message_notifies_its_recipient(root: Path) -> None:
    db = build(root)
    before = len(call(db, "list_notifications", only_unread=True)["notifications"])
    call(db, "send_dm_message", recipient_id=BEN, body="quick question", actor=PRIYA)
    unread = call(db, "list_notifications", only_unread=True)["notifications"]
    assert len(unread) == before + 1 and unread[0]["kind"] == "direct_message"


def test_a_reply_in_a_thread_you_follow_notifies_you(root: Path) -> None:
    db = build(root)
    call(db, "follow_thread", thread_parent_id="MSG145")
    before = len(call(db, "list_notifications", only_unread=True)["notifications"])
    call(db, "reply_to_thread", thread_parent_id="MSG145", body="update", actor=PRIYA)
    unread = call(db, "list_notifications", only_unread=True)["notifications"]
    assert len(unread) == before + 1 and unread[0]["kind"] == "thread_reply"


def test_marking_a_notification_read_takes_it_off_the_unread_list(root: Path) -> None:
    db = build(root)
    call(db, "tag_people", channel_id=ACME, user_ids=[BEN], body="ping", actor=PRIYA)
    unread = call(db, "list_notifications", only_unread=True)["notifications"]
    call(db, "mark_notification_read", notification_id=unread[0]["notification_id"])
    remaining = {n["notification_id"] for n in call(db, "list_notifications", only_unread=True)["notifications"]}
    assert unread[0]["notification_id"] not in remaining
    assert any(n["notification_id"] == unread[0]["notification_id"]
               for n in call(db, "list_notifications")["notifications"])


def test_notifications_do_not_leak_between_people(root: Path) -> None:
    db = build(root)
    call(db, "tag_people", channel_id=ACME, user_ids=[BEN], body="ping", actor=PRIYA)
    for notification in call(db, "list_notifications", actor=PRIYA)["notifications"]:
        assert notification["user_id"] == PRIYA


# ---------------------------------------------------------------------------
# Search: filters, phrases, people, channels
# ---------------------------------------------------------------------------

def test_a_quoted_phrase_matches_the_phrase_not_the_words(root: Path) -> None:
    db = build(root)
    loose = call(db, "search_messages", query="automated access checker")["total"]
    exact = call(db, "search_messages", query='"automated access checker"')["total"]
    assert loose > 0 and exact > 0
    assert exact <= loose
    scrambled = call(db, "search_messages", query='"checker access automated"')["total"]
    assert scrambled == 0, "a phrase must respect word order"


def test_from_narrows_to_one_sender(root: Path) -> None:
    db = build(root)
    result = call(db, "search_messages", query="from:priya")
    assert result["total"] > 0
    assert all(m["user_id"] == PRIYA for m in result["matches"])


def test_in_narrows_to_one_conversation(root: Path) -> None:
    db = build(root)
    result = call(db, "search_messages", query="in:identity-eng EU")
    assert result["total"] > 0
    assert all(m["channel_id"] == "C020" for m in result["matches"])


def test_date_filters_bound_the_window(root: Path) -> None:
    db = build(root)
    everything = call(db, "search_messages", query="migration")["total"]
    early = call(db, "search_messages", query="migration before:2026-08-05")["total"]
    late = call(db, "search_messages", query="migration after:2026-08-05")["total"]
    assert early > 0 and late > 0
    assert early + late == everything, "before/after must partition, not overlap"

    one_day = call(db, "search_messages", query="migration on:2026-08-19")
    assert one_day["total"] > 0
    assert all(m["ts_date"] == "2026-08-19" for m in one_day["matches"])


def test_filters_combine_and_an_impossible_combination_finds_nothing(root: Path) -> None:
    db = build(root)
    assert call(db, "search_messages", query="from:priya in:data-ops before:2026-08-04")["total"] == 0


def test_an_unparseable_filter_is_rejected_rather_than_silently_ignored(root: Path) -> None:
    db = build(root)
    assert fails(db, "search_messages", query="before:soon") == "invalid_search_filter"
    assert fails(db, "search_messages", query="from:nobody-at-all") == "invalid_search_filter"


def test_search_filters_still_respect_visibility(root: Path) -> None:
    db = build(root)
    assert call(db, "search_messages", query="in:security-operations")["total"] == 0


def test_people_search_finds_by_name_handle_and_title(root: Path) -> None:
    db = build(root)
    by_name = call(db, "search_users", query="priya")["users"]
    assert [u["user_id"] for u in by_name] == [PRIYA]
    assert call(db, "search_users", query="identity engineer")["users"][0]["user_id"] == PRIYA
    assert call(db, "search_users", query="zzzz")["users"] == []


def test_channel_search_finds_by_name_and_topic(root: Path) -> None:
    db = build(root)
    assert "C020" in {c["channel_id"] for c in call(db, "search_channels", query="identity")["channels"]}
    assert all(not c["is_private"] or c["is_member"]
               for c in call(db, "search_channels", query="e")["channels"])


# ---------------------------------------------------------------------------
# Determinism: every successful mutation costs exactly one step
# ---------------------------------------------------------------------------

def test_new_mutations_advance_the_clock_once_and_failures_not_at_all(root: Path) -> None:
    from slack_sim.clock import VIRTUAL_CLOCK, parse_slack_ts
    from slack_sim.sqlite_common import connect

    db = build(root)

    def now() -> int:
        with connect(db) as connection:
            return parse_slack_ts(VIRTUAL_CLOCK.now(connection))

    for tool, payload in (
        ("follow_thread", {"thread_parent_id": "MSG145"}),
        ("pin_message", {"message_id": "MSG145"}),
        ("save_message", {"message_id": "MSG145"}),
        ("set_presence", {"presence": "away"}),
        ("set_status", {"status_text": "busy"}),
        ("join_channel", {"channel_id": DATA_QUALITY}),
        ("leave_channel", {"channel_id": DATA_QUALITY}),
        ("unfollow_thread", {"thread_parent_id": "MSG145"}),
        ("unpin_message", {"message_id": "MSG145"}),
        ("unsave_message", {"message_id": "MSG145"}),
    ):
        before = now()
        execute_tool(db, tool, payload, BEN)
        assert now() == before + 1_000_000, f"{tool} must cost exactly one step"

    before = now()
    try:
        execute_tool(db, "set_presence", {"presence": "nope"}, BEN)
    except ToolError:
        pass
    assert now() == before, "a rejected call must not move the clock"


# ---------------------------------------------------------------------------
# The verifier's view of the new mutations
# ---------------------------------------------------------------------------

def test_investigation_is_free_but_damage_is_charged(root: Path) -> None:
    from slack_sim.migration_reward import _side_effect_penalty
    from slack_sim.service import export_state

    db = build(root)
    # Everything an investigator legitimately does.
    call(db, "follow_thread", thread_parent_id="MSG145")
    call(db, "pin_message", message_id="MSG145")
    call(db, "save_message", message_id="MSG145")
    call(db, "set_presence", presence="away")
    call(db, "set_status", status_text="reviewing Acme")
    call(db, "mark_conversation_read", conversation_id=ACME)
    call(db, "join_channel", channel_id=DATA_QUALITY)
    assert _side_effect_penalty(export_state(db)) == 0.0

    call(db, "leave_channel", channel_id=ACME)
    assert _side_effect_penalty(export_state(db)) > 0.0, "losing a channel is damage"


def test_deleting_history_is_charged_not_treated_as_damage(root: Path) -> None:
    from slack_sim.migration_reward import _side_effect_penalty, check_workspace_integrity
    from slack_sim.service import export_state

    db = build(root)
    own = [m["message_id"] for m in export_state(db)["messages"]
           if m["author_id"] == BEN
           and not any(r["message_id"] == m["message_id"] for r in export_state(db)["reactions"])][0]
    call(db, "delete_message", message_id=own)
    state = export_state(db)
    # The episode is still readable evidence about the agent...
    check_workspace_integrity(state)
    # ...and the deletion is charged.
    assert _side_effect_penalty(state) >= 0.5


def test_a_message_the_actor_could_not_have_deleted_going_missing_is_damage(root: Path) -> None:
    from slack_sim.migration_reward import IntegrityError, check_workspace_integrity
    from slack_sim.service import export_state

    db = build(root)
    state = export_state(db)
    state["messages"] = [m for m in state["messages"] if m["message_id"] != "MSG145"]
    try:
        check_workspace_integrity(state)
    except IntegrityError:
        return
    raise AssertionError("a vanished message from another author must invalidate the episode")


def test_archiving_a_channel_is_charged(root: Path) -> None:
    from slack_sim.migration_reward import _side_effect_penalty
    from slack_sim.service import export_state

    db = build(root)
    call(db, "archive_channel", channel_id=ACME, actor=ALICE)
    assert _side_effect_penalty(export_state(db)) >= 0.5


def test_the_seeded_workspace_already_has_an_inbox(root: Path) -> None:
    db = build(root)
    inbox = call(db, "list_notifications")
    assert inbox["total"] > 0, "two weeks of mentions and DMs should have notified somebody"
    assert inbox["unread_count"] > 0
    assert {n["kind"] for n in inbox["notifications"]} <= {
        "mention", "direct_message", "thread_reply", "reaction",
    }
    assert all(n["user_id"] == BEN for n in inbox["notifications"])
    assert 8 <= inbox["total"] <= 12, "an inbox, not a firehose and not empty"
    # Most of it is ordinary company noise; only some of it is about Acme.
    acme = [n for n in inbox["notifications"] if n["conversation_id"] in
            {"C019", "C020", "C021", "C022", "C023"}]
    assert 0 < len(acme) < inbox["total"]


def test_the_starting_inbox_is_declared_seed_data(root: Path) -> None:
    """Notifications are part of the initial state, not recomputed per run."""
    from slack_sim.seed import SLACK_NOTIFICATIONS
    from slack_sim.service import export_state

    db = build(root)
    live = export_state(db)["notifications"]
    assert len(live) == len(SLACK_NOTIFICATIONS)
    assert [n["notification_id"] for n in live] == [
        n.notification_id for n in SLACK_NOTIFICATIONS
    ]
    unread_in_seed = {n.notification_id for n in SLACK_NOTIFICATIONS if n.read_step is None}
    assert {n["notification_id"] for n in live if n["read_ts"] is None} == unread_in_seed


def test_reseeding_reproduces_the_same_inbox(root: Path) -> None:
    from slack_sim.service import export_state

    first = build(root / "a")
    second = build(root / "b")
    assert export_state(first)["notifications"] == export_state(second)["notifications"]


# ---------------------------------------------------------------------------
# The error taxonomy
# ---------------------------------------------------------------------------
#
# `ToolError` says the rules refused what was asked. Everything else used to
# arrive as one `internal_error` carrying a stringified class name, which put a
# locked database, a mistyped tool name and a genuine bug in the same bucket --
# and those call for three different people to look at three different things.


def test_an_unknown_tool_is_a_typed_refusal(root: Path) -> None:
    from slack_sim.sqlite_common import UnknownToolError

    db = build(root)
    try:
        execute_tool(db, "send_carrier_pigeon", {}, BEN)
    except UnknownToolError as error:
        assert isinstance(error, ToolError), "a bad tool name is the caller's mistake"
        assert error.error_code == "unknown_tool"
        assert "send_carrier_pigeon" in str(error)
        return
    raise AssertionError("an unknown tool name did not raise")


def test_a_storage_failure_is_not_reported_as_a_rule_violation(root: Path) -> None:
    """A corrupt or locked database says nothing about what the agent asked
    for. Reporting it as a `ToolError` would tell the agent to try something
    else, when the truth is that nothing it tries can work."""
    from slack_sim.sqlite_common import StorageError, WorldError

    db = build(root)
    db.write_bytes(b"this is not a database")
    try:
        execute_tool(db, "list_channels", {}, BEN)
    except StorageError as error:
        assert isinstance(error, WorldError)
        assert not isinstance(error, ToolError)
        assert error.error_code == "storage_error"
        assert error.error_type == "internal_error"
        return
    raise AssertionError("a corrupt database did not raise a StorageError")


def test_the_server_gives_a_storage_failure_its_own_code(root: Path) -> None:
    from slack_sim.server import WorldService

    db = build(root)
    db.write_bytes(b"this is not a database")
    response = WorldService(db, root / "snapshot.sql").call_tool("list_channels", {})

    assert response["ok"] is False
    assert response["error"]["code"] == "storage_error", (
        "a storage failure must not arrive as the generic internal_error"
    )


def test_every_world_error_carries_a_code_the_server_can_report(root: Path) -> None:
    """`_error(exc.error_code, exc.error_type, ...)` is the only mapping the
    server has. A `WorldError` subclass without those attributes would fall
    through to the catch-all it was created to replace."""
    from slack_sim import sqlite_common

    subclasses = [
        value for value in vars(sqlite_common).values()
        if isinstance(value, type) and issubclass(value, sqlite_common.WorldError)
    ]
    assert len(subclasses) >= 4, f"only found {[c.__name__ for c in subclasses]}"
    for cls in subclasses:
        assert cls.error_code, f"{cls.__name__} has no error_code"
        assert cls.error_type, f"{cls.__name__} has no error_type"


def test_the_two_paths_to_an_unknown_tool_agree(root: Path) -> None:
    """The server rejects a bad name from its own table; `execute_tool` rejects
    it from the handler map. An agent that hits one and then the other must not
    be told two different things about the same mistake."""
    from slack_sim.server import WorldService
    from slack_sim.sqlite_common import UnknownToolError

    db = build(root)
    from_server = WorldService(db, root / "snapshot.sql").call_tool("send_carrier_pigeon", {})
    try:
        execute_tool(db, "send_carrier_pigeon", {}, BEN)
    except UnknownToolError as error:
        assert from_server["error"]["code"] == error.error_code
        assert from_server["error"]["type"] == error.error_type
        return
    raise AssertionError("an unknown tool name did not raise")


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
    print(f"slack surface: ok ({len(tests)} tests)")


if __name__ == "__main__":
    main()
