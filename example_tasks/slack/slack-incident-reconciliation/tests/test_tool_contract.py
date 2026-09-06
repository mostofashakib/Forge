#!/usr/bin/env python3
"""Interface hygiene: naming, error taxonomy, visibility, and fault reporting.

These guard properties an agent has to be able to rely on:

  * one operation has exactly one name, everywhere;
  * one error code always means one kind of failure;
  * an error never discloses the existence of something the actor cannot see;
  * a verifier fault is reported as a fault, not as a score of zero.
"""

from __future__ import annotations

import ast
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import slack_sim.service
from slack_sim.identity import LOGGED_IN_USER
from slack_sim.seed import (
    SLACK_CHANNELS,
    SLACK_MESSAGES,
    SLACK_CHATS,
    SLACK_REACTIONS,
    build_slack_memberships,
    conversations_visible_to,
)
from slack_sim.service import (
    TOOL_NAMES,
    create_channel,
    create_dm_message,
    get_channel_messages,
    get_thread_replies,
    list_channels,
    list_chats,
    list_users,
    post_message,
    reply_to_thread,
    search_messages,
    seed_database,
    send_dm_message,
)
from slack_sim.sqlite_common import ToolError
from slack_sim.tool_definitions import get_tool_definitions

# Locate the module as imported, not by a path relative to this file: in the
# container these tests live at /tests and the package at /opt/grading.
SERVICE_SOURCE = Path(slack_sim.service.__file__).resolve()
TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _seeded(root: Path) -> Path:
    db = root / "slack.db"
    seed_database(db, root / "seed.sql")
    return db


# --------------------------------------------------------------------------
# One operation, one name
# --------------------------------------------------------------------------

def test_every_tool_name_uses_one_flat_convention() -> None:
    offenders = [name for name in TOOL_NAMES if not TOOL_NAME_PATTERN.fullmatch(name)]
    assert not offenders, (
        f"tool names must be flat snake_case so the CLI verb, the handler key and "
        f"the schema name are the same string; found {offenders}"
    )


def test_schema_names_match_handler_names() -> None:
    assert {tool["name"] for tool in get_tool_definitions()} == set(TOOL_NAMES)


def test_the_service_cli_exposes_the_same_verbs() -> None:
    """The admin/build CLI must not invent a second spelling for a tool."""
    tree = ast.parse(SERVICE_SOURCE.read_text(encoding="utf-8"))
    choices: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "command"
        ):
            for keyword in node.keywords:
                if keyword.arg == "choices" and isinstance(keyword.value, ast.List):
                    choices = {
                        element.value for element in keyword.value.elts
                        if isinstance(element, ast.Constant)
                    }
    missing = set(TOOL_NAMES) - choices
    assert not missing, f"CLI is missing verbs for {sorted(missing)}"


# --------------------------------------------------------------------------
# One code, one meaning
# --------------------------------------------------------------------------

def test_each_error_code_maps_to_exactly_one_error_type() -> None:
    tree = ast.parse(SERVICE_SOURCE.read_text(encoding="utf-8"))
    mapping: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "ToolError"
            and len(node.args) >= 2
            and all(isinstance(arg, ast.Constant) for arg in node.args[:2])
        ):
            mapping.setdefault(node.args[0].value, set()).add(node.args[1].value)
    ambiguous = {code: sorted(kinds) for code, kinds in mapping.items() if len(kinds) > 1}
    assert not ambiguous, (
        "an error code must always mean the same kind of failure; "
        f"these map to several types: {ambiguous}"
    )


# --------------------------------------------------------------------------
# Errors must not disclose what the actor cannot see
# --------------------------------------------------------------------------

def _error_for(db: Path, conversation_id: str) -> ToolError:
    try:
        get_channel_messages(db, conversation_id, LOGGED_IN_USER.user_id)
    except ToolError as exc:
        return exc
    raise AssertionError(f"{conversation_id} unexpectedly readable")


def test_private_channels_are_not_enumerable(root: Path) -> None:
    """A private channel the actor is not in must look exactly like no channel."""
    db = _seeded(root)
    absent = _error_for(db, "C099")
    private = _error_for(db, "C009")  # exists, private, actor is not a member
    assert (private.error_code, private.error_type) == (absent.error_code, absent.error_type), (
        f"a private channel answers {private.error_code}/{private.error_type} while a "
        f"nonexistent one answers {absent.error_code}/{absent.error_type}; the difference "
        "lets an agent enumerate private channels"
    )
    assert str(private) == str(absent), "the message text discloses existence too"


def test_private_chats_are_not_enumerable(root: Path) -> None:
    db = _seeded(root)
    absent = _error_for(db, "D099")
    for chat_id in ("D003", "G002"):  # real chats the actor does not belong to
        hidden = _error_for(db, chat_id)
        assert (hidden.error_code, hidden.error_type) == (absent.error_code, absent.error_type), (
            f"{chat_id} is distinguishable from a nonexistent conversation"
        )


def test_seed_visibility_matches_the_live_service(root: Path) -> None:
    """The verifier restates the visibility rule; the two must not drift.

    `conversations_visible_to` computes from the seed dataclasses what the
    service computes from the database. The answer key depends on the first and
    the agent experiences the second, so a disagreement between them is a task
    that cannot be solved.
    """
    db = _seeded(root)
    claimed = conversations_visible_to(LOGGED_IN_USER.user_id)
    everything = (
        {c.channel_id for c in SLACK_CHANNELS}
        | {c.chat_id for c in SLACK_CHATS}
    )
    assert claimed < everything, "the seed must contain something the actor cannot see"

    for conversation_id in sorted(everything):
        try:
            get_channel_messages(db, conversation_id, LOGGED_IN_USER.user_id)
            readable = True
        except ToolError:
            readable = False
        # Public channels the actor has not joined are enumerable and
        # searchable but not readable, which is exactly the state the evidence
        # policy calls visible.
        searchable = readable or _is_unjoined_public_channel(conversation_id)
        assert searchable == (conversation_id in claimed), (
            f"{conversation_id}: the seed says visible={conversation_id in claimed} "
            f"but the service says readable={readable}"
        )


def _is_unjoined_public_channel(conversation_id: str) -> bool:
    channel = next(
        (c for c in SLACK_CHANNELS if c.channel_id == conversation_id), None
    )
    if channel is None or channel.is_private:
        return False
    return not any(
        m.channel_id == conversation_id and m.user_id == LOGGED_IN_USER.user_id
        for m in build_slack_memberships()
    )


def test_reactions_left_by_other_people_are_visible(root: Path) -> None:
    """Seeded reactions the agent cannot see are seeded reactions that do nothing."""
    db = _seeded(root)
    seeded = [r for r in SLACK_REACTIONS if r.message_id == "MSG001"]
    assert seeded, "MSG001 is the anchor this test reads; it must carry reactions"

    messages = get_channel_messages(db, "C003", LOGGED_IN_USER.user_id)["messages"]
    opening = next(m for m in messages if m["id"] == "MSG001")
    shown = {
        (entry["emoji"], user_id)
        for entry in opening["reactions"]
        for user_id in entry["user_ids"]
    }
    assert shown == {(r.emoji, r.user_id) for r in seeded}, (
        f"get_channel_messages reported {shown} for MSG001"
    )
    eyes = next(entry for entry in opening["reactions"] if entry["emoji"] == "eyes")
    assert eyes["count"] == len(eyes["user_ids"]) > 1, "identical emoji must group"


def test_unjoined_channels_do_not_advertise_unread_traffic(root: Path) -> None:
    """An unread badge on a channel you cannot open is an invitation to fail.

    Public channels the actor has not joined are listed, but reading one is
    denied, so reporting a nonzero unread count only steers the agent into a
    permission error.
    """
    db = _seeded(root)
    listed = list_channels(db, LOGGED_IN_USER.user_id)["channels"]
    unjoined = [c for c in listed if not c["is_member"]]
    assert unjoined, "the seed must contain a public channel the actor has not joined"
    noisy = {c["name"]: c["unread_count"] for c in unjoined if c["unread_count"]}
    assert not noisy, f"unjoined channels report unread traffic: {noisy}"

    joined = {c["name"]: c["unread_count"] for c in listed if c["is_member"]}
    assert any(count > 0 for count in joined.values()), (
        "the workspace should look lived-in: some joined channel must be unread"
    )


# --------------------------------------------------------------------------
# Tool semantics an agent has to be able to predict
# --------------------------------------------------------------------------

def test_replying_to_a_thread_reply_lands_in_that_thread(root: Path) -> None:
    """Slack threads are one level deep, and this one must be too.

    The customer-status thread carries several replies. An agent that reads it
    and replies to the message it is echoing -- rather than to the thread root
    it had to scroll back for -- is doing the right thing, and in Slack its
    reply lands in the same thread. If the parent is stored verbatim instead,
    that reply forms a second-level thread that exists nowhere in the seed, and
    the verifier scores a correct action as a miss.
    """
    db = _seeded(root)
    root_id, reply_id = "MSG003", "MSG020"
    seeded_reply = next(m for m in SLACK_MESSAGES if m.message_id == reply_id)
    assert seeded_reply.thread_parent_id == root_id, "fixture drifted"

    posted = reply_to_thread(db, reply_id, "echoing the approved update", LOGGED_IN_USER.user_id)
    assert posted["message"]["thread_parent_id"] == root_id, (
        f"replying to {reply_id} produced a nested thread under "
        f"{posted['message']['thread_parent_id']} instead of joining {root_id}"
    )

    root = next(
        m for m in get_channel_messages(db, "C004", LOGGED_IN_USER.user_id)["messages"]
        if m["id"] == root_id
    )
    shown = get_thread_replies(db, "C004", root["thread_ts"], LOGGED_IN_USER.user_id)["messages"]
    assert all(m["thread_id"] == root_id for m in shown), "replies escaped their root thread"
    assert all(not m["is_thread_reply"] or m["thread_ts"] == root["ts"] for m in shown)


def test_a_channel_page_does_not_repeat_its_own_conversation(root: Path) -> None:
    """`conversation` and `channel` were the same object under two names.

    A page that says which channel it is twice is not clearer for it, and the
    duplicate invited the question of which one was authoritative.
    """
    db = _seeded(root)
    page = get_channel_messages(db, "C023", LOGGED_IN_USER.user_id)
    assert page["conversation"]["channel_id"] == "C023"
    assert "channel" not in page and "chat" not in page, (
        "the conversation is still announced twice"
    )

    chat = list_chats(db, LOGGED_IN_USER.user_id)["chats"][0]
    chat_page = get_channel_messages(db, chat["chat_id"], LOGGED_IN_USER.user_id)
    assert chat_page["conversation"]["chat_id"] == chat["chat_id"]
    assert "chat" not in chat_page


def test_a_person_is_identified_once(root: Path) -> None:
    """`id` was a second copy of `user_id`, character for character."""
    db = _seeded(root)
    people = list_users(db, LOGGED_IN_USER.user_id)["users"]
    assert people and all(person["user_id"] for person in people)
    assert not any("id" in person for person in people), "user identity is still doubled"


def test_a_message_with_no_thread_carries_no_thread_fields(root: Path) -> None:
    """Absent means absent, which is the only thing null ever meant here.

    Ten fields per message reported that nothing had happened -- no edit, no
    reply, no reaction. Across one page of fifty that is most of the payload,
    and every one of them said the same thing an omission says.
    """
    db = _seeded(root)
    optional = ("thread_id", "thread_ts", "latest_reply", "edited_ts",
                "reply_to_id", "mentions", "reactions")
    page = get_channel_messages(db, "C023", LOGGED_IN_USER.user_id)
    for message in page["messages"]:
        empty = [f for f in optional if f in message and not message[f]]
        assert not empty, f"{message['id']} still reports {empty} as empty"

    plain = next(m for m in page["messages"]
                 if m["reply_count"] == 0 and not m.get("mentions"))
    assert not any(field in plain for field in optional), (
        f"a message with nothing attached still carries "
        f"{[f for f in optional if f in plain]}"
    )

    threaded = next(m for m in page["messages"] if m["reply_count"] > 0)
    assert threaded["thread_ts"] == threaded["ts"]
    assert threaded["thread_id"] == threaded["id"]
    assert threaded["latest_reply"] > threaded["ts"]


def test_nothing_that_carries_meaning_is_dropped(root: Path) -> None:
    """The omission rule is about absence, never about falsity.

    `is_member: false` is how the agent learns there is a channel it has not
    joined -- the cutover channel is exactly that -- and `ordinal: 0` is the
    first mention rather than no mention. Dropping a falsy value because it is
    falsy would delete the answer along with the padding.
    """
    db = _seeded(root)
    actor = LOGGED_IN_USER.user_id

    channels = list_channels(db, actor)["channels"]
    assert all("is_member" in c and "is_archived" in c and "is_private" in c
               for c in channels)
    assert any(c["is_member"] is False for c in channels), "fixture drifted"

    page = get_channel_messages(db, "C019", actor)
    assert "unread_count" in page and "last_read_ts" in page
    assert all("is_unread" in m and "reply_count" in m for m in page["messages"])

    mentioned = next(m for m in page["messages"] if m.get("mentions"))
    assert all("ordinal" in mention for mention in mentioned["mentions"])


def test_channel_history_requires_explicit_thread_reads(root: Path) -> None:
    db = _seeded(root)
    history = get_channel_messages(db, "C023", LOGGED_IN_USER.user_id)
    assert history["messages"], "debugging history unexpectedly empty"
    assert all(not message["is_thread_reply"] for message in history["messages"])
    threaded = next(message for message in history["messages"] if message["reply_count"] > 0)
    assert threaded["thread_ts"] == threaded["ts"]
    assert threaded["latest_reply"] > threaded["ts"]

    thread = get_thread_replies(
        db, "C023", threaded["thread_ts"], LOGGED_IN_USER.user_id
    )
    assert thread["messages"][0]["id"] == threaded["id"]
    assert len(thread["messages"]) == threaded["reply_count"] + 1
    assert [m["ts"] for m in thread["messages"]] == sorted(
        m["ts"] for m in thread["messages"]
    )


def test_channel_history_scrolls_in_fifty_message_pages(root: Path) -> None:
    db = _seeded(root)
    actor = LOGGED_IN_USER.user_id
    channel_id = create_channel(db, "scrollback-test", False, actor)["channel"][
        "channel_id"
    ]
    posted_ids = [
        post_message(db, channel_id, f"scrollback message {index}", actor)["message"][
            "message_id"
        ]
        for index in range(125)
    ]

    history_query_sizes: list[int] = []
    original_query_rows = slack_sim.service.query_rows

    def tracked_query_rows(connection, sql, params=()):
        rows = original_query_rows(connection, sql, params)
        if "ORDER BY m.ts DESC, m.message_id DESC" in sql:
            assert "LIMIT ?" in sql
            history_query_sizes.append(len(rows))
        return rows

    slack_sim.service.query_rows = tracked_query_rows
    try:
        first = get_channel_messages(db, channel_id, actor)
        assert get_channel_messages(db, channel_id, actor)["messages"] == first["messages"]
        # New traffic arriving at the top must not shift the cursor and cause a
        # duplicate or skipped older message on page two.
        post_message(db, channel_id, "arrived while scrolling", actor)
        second = get_channel_messages(db, channel_id, actor, first["next_cursor"])
        third = get_channel_messages(db, channel_id, actor, second["next_cursor"])
    finally:
        slack_sim.service.query_rows = original_query_rows

    assert [len(page["messages"]) for page in (first, second, third)] == [50, 50, 25]
    assert history_query_sizes and max(history_query_sizes) == 51
    assert third["next_cursor"] == ""
    observed_ids = [
        message["id"]
        for page in (first, second, third)
        for message in page["messages"]
    ]
    assert observed_ids == list(reversed(posted_ids))
    assert len(observed_ids) == len(set(observed_ids)) == 125

    try:
        get_channel_messages(db, channel_id, actor, limit=51)
    except ToolError as exc:
        assert exc.error_code == "invalid_limit"
    else:
        raise AssertionError("channel history returned more than its 50-message window")


def test_collection_pagination_uses_bound_opaque_cursors(root: Path) -> None:
    db = _seeded(root)
    first = list_users(db, LOGGED_IN_USER.user_id, limit=7)
    assert len(first["users"]) == 7 and first["next_cursor"]
    second = list_users(db, LOGGED_IN_USER.user_id, first["next_cursor"], 7)
    assert {u["user_id"] for u in first["users"]}.isdisjoint(
        u["user_id"] for u in second["users"]
    )
    try:
        list_channels(db, LOGGED_IN_USER.user_id, first["next_cursor"], 7)
    except ToolError as exc:
        assert exc.error_code == "invalid_cursor"
    else:
        raise AssertionError("a list_users cursor was accepted by list_channels")


def test_search_returns_matches_with_bounded_context(root: Path) -> None:
    db = _seeded(root)
    result = search_messages(db, "Acme", LOGGED_IN_USER.user_id, limit=3)
    assert len(result["matches"]) == 3
    assert result["next_cursor"]
    for match in result["matches"]:
        # Search spans channels, so every match locates itself. The thread
        # fields ride along only when there is a thread to open, and then both
        # of them do -- `thread_ts` alone is what get_thread_replies needs.
        assert {"id", "channel_id", "channel_name", "user_id", "ts", "text"} <= match.keys()
        assert ("thread_id" in match) == ("thread_ts" in match)
        assert len(match["context"]["before"]) <= 1
        assert len(match["context"]["after"]) <= 1
    following = search_messages(
        db, "Acme", LOGGED_IN_USER.user_id, result["next_cursor"], 3
    )
    assert {m["id"] for m in result["matches"]}.isdisjoint(
        m["id"] for m in following["matches"]
    )


def test_a_dm_to_yourself_is_your_own_chat(root: Path) -> None:
    """Opening a DM must match the participant set exactly.

    The lookup joins the participant table once per user. Asking for a DM with
    yourself makes both joins the same row, so every DM you are in matches and
    the first one wins -- handing back a private conversation with somebody
    else, under a chat ID the agent will then send into.
    """
    db = _seeded(root)
    actor = LOGGED_IN_USER.user_id
    opened = create_dm_message(db, actor, actor)["chat"]
    participants = {p["user_id"] for p in opened["participants"]}
    assert participants == {actor}, (
        f"a self-DM resolved to chat {opened['chat_id']} whose members are "
        f"{sorted(participants)}"
    )

    for other in ("U001", "U003"):
        found = create_dm_message(db, other, actor)["chat"]
        assert {p["user_id"] for p in found["participants"]} == {actor, other}, (
            f"opening a DM with {other} resolved to {found['chat_id']}"
        )


def test_unread_flags_are_absent_where_there_is_nothing_to_read(root: Path) -> None:
    """Search spans public channels the actor has not joined; unread does not.

    Unread means "waiting for you". A message in a channel the actor cannot
    open is not waiting for anyone, and marking it unread invites the agent to
    go and fail to read it.
    """
    db = _seeded(root)
    unjoined = "C011"
    assert _is_unjoined_public_channel(unjoined), "fixture drifted"

    hits = search_messages(db, "backfill", LOGGED_IN_USER.user_id)["matches"]
    from_unjoined = [m for m in hits if m["channel_id"] == unjoined]
    assert from_unjoined, "search must still reach public channels the actor has not joined"
    flagged = [m["id"] for m in from_unjoined if m["is_unread"]]
    assert not flagged, f"unreadable messages reported as unread: {flagged}"

    joined = [m for m in search_messages(db, "Acme", LOGGED_IN_USER.user_id)["matches"]
              if m["channel_id"] == "C019"]
    assert any(m["is_unread"] for m in joined), (
        "search must still report genuine unread traffic in joined channels"
    )


def test_you_always_have_a_chat_with_yourself(root: Path) -> None:
    """Slack gives everyone a self-DM that is simply there.

    Messaging yourself is how people park a note mid-incident. Making the agent
    spend a turn creating that chat before it can use it -- and only discover it
    exists by guessing -- is a difference from Slack with no upside.
    """
    db = _seeded(root)
    actor = LOGGED_IN_USER.user_id
    listed = list_chats(db, actor)["chats"]
    self_chats = [c for c in listed if {p["user_id"] for p in c["participants"]} == {actor}]
    assert len(self_chats) == 1, (
        f"expected exactly one self-DM in the seeded workspace, found {len(self_chats)}"
    )
    self_chat = self_chats[0]
    assert self_chat["type"] == "dm"
    assert self_chat["unread_count"] == 0, "your own notes are never unread"

    # Both routes reach it, and neither invents a second one.
    assert create_dm_message(db, actor, actor)["id"] == self_chat["chat_id"]
    sent = send_dm_message(db, actor, None, "parking this for later", actor)
    assert sent["message"]["channel_id"] == self_chat["chat_id"]
    assert len(list_chats(db, actor)["chats"]) == len(listed), "a duplicate self-DM appeared"

    body = [m["text"] for m in get_channel_messages(db, self_chat["chat_id"], actor)["messages"]]
    assert "parking this for later" in body


def test_a_reply_records_both_its_thread_and_what_it_answers(root: Path) -> None:
    """A thread is flat, but each reply still knows which message it answers.

    `thread_parent_id` says which conversation the reply belongs to;
    `reply_to_message_id` says which message prompted it. Without the second,
    a long thread collapses into an undifferentiated list and the agent cannot
    tell a rebuttal of the scope call from a rebuttal of the mitigation.
    """
    db = _seeded(root)
    actor = LOGGED_IN_USER.user_id
    root_id, sibling_id = "MSG003", "MSG020"

    direct = reply_to_thread(db, root_id, "answering the thread", actor)["message"]
    assert (direct["thread_parent_id"], direct["reply_to_message_id"]) == (root_id, root_id)

    nested = reply_to_thread(db, sibling_id, "answering the approved update", actor)["message"]
    assert nested["thread_parent_id"] == root_id, "the reply must stay in the same thread"
    assert nested["reply_to_message_id"] == sibling_id, (
        f"the reply lost its origin: {nested['reply_to_message_id']}"
    )

    roots = {m["id"]: m for m in get_channel_messages(db, "C004", actor)["messages"]}
    shown = {
        m["id"]: m
        for m in get_thread_replies(db, "C004", roots[root_id]["thread_ts"], actor)["messages"]
    }
    assert shown[root_id]["reply_count"] >= 2, "a thread root must report how many replies it has"
    assert "reply_to_id" not in shown[root_id], "a thread root answers nothing"
    assert shown[sibling_id]["reply_count"] == 0, "only roots carry a reply count"


def test_the_seed_contains_threads_people_actually_reply_within(root: Path) -> None:
    """Realism check: seeded threads must exercise the origin field."""
    within_thread = [
        m for m in SLACK_MESSAGES
        if m.reply_to_id is not None and m.reply_to_id != m.thread_parent_id
    ]
    assert len(within_thread) >= 5, (
        "the seed's threads are all flat answers to the root, so nothing "
        f"exercises reply origin (found {len(within_thread)})"
    )
    by_id = {m.message_id: m for m in SLACK_MESSAGES}
    for message in SLACK_MESSAGES:
        if message.reply_to_id is None:
            continue
        target = by_id[message.reply_to_id]
        assert target.created_step < message.created_step, f"{message.message_id} answers the future"
        assert (target.thread_parent_id or target.message_id) == message.thread_parent_id, (
            f"{message.message_id} answers a message in a different thread"
        )


def test_public_channels_still_report_permission_plainly(root: Path) -> None:
    """Public channel existence is already public, so denial leaks nothing."""
    db = _seeded(root)
    public = _error_for(db, "C011")  # public, actor is not a member
    assert public.error_type == "permission_denied", (
        "hiding public channels would make the workspace harder to reason about "
        "for no privacy gain, since list_channels already lists them"
    )


# --------------------------------------------------------------------------
# A verifier fault is a fault
# --------------------------------------------------------------------------

def test_scoring_faults_report_invalid_rather_than_zero(root: Path) -> None:
    import test_migration_readiness as entry
    from slack_sim.service import export_state

    broken = export_state(_seeded(root))
    broken.pop("scenario_events")
    rewards, details = entry.evaluate(broken)
    assert rewards["valid"] == 0.0, (
        "a verifier that cannot score must say so; reporting reward 0 with valid 1 "
        "makes an infrastructure fault indistinguishable from a failing agent"
    )
    assert "integrity_error" in details


def test_the_agent_has_a_real_shell() -> None:
    """Only meaningful inside the task container; skipped elsewhere.

    Debian's useradd default is /bin/sh (dash). Under dash an agent's bashisms
    fail and Claude Code has no bash-shaped shell to snapshot, which is why
    shell-snapshots came back empty despite a run making 11 Bash calls.
    """
    try:
        entry = next(
            line for line in Path("/etc/passwd").read_text().splitlines()
            if line.startswith("agent:")
        )
    except (OSError, StopIteration):
        return  # not the agent container
    shell = entry.rsplit(":", 1)[-1]
    assert shell.endswith("bash"), (
        f"the agent's login shell is {shell!r}; it must be bash so shell "
        "snapshots are produced and bashisms in Bash tool calls work"
    )


def test_the_rl_contract_and_the_server_share_one_workspace() -> None:
    """The RL lifecycle must drive the same world the agent's tools reach.

    When these diverge, rl_env.py silently creates a second database and every
    RL rollout mutates a shadow workspace that the tools and the verifier never
    observe.
    """
    from slack_sim.environment import SlackIncidentEnvironment
    from slack_sim.server import DEFAULT_DB, DEFAULT_SNAPSHOT
    from slack_sim.service import DEFAULT_DB_PATH, DEFAULT_SNAPSHOT_PATH

    adapter = SlackIncidentEnvironment()
    assert Path(adapter.db_path) == Path(DEFAULT_DB) == Path(DEFAULT_DB_PATH), (
        f"RL contract uses {adapter.db_path} but the server serves {DEFAULT_DB}"
    )
    assert Path(adapter.snapshot_path) == Path(DEFAULT_SNAPSHOT) == Path(DEFAULT_SNAPSHOT_PATH)


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        if test.__code__.co_argcount:
            with tempfile.TemporaryDirectory(prefix="slack-contract-") as temp_dir:
                test(Path(temp_dir))
        else:
            test()
        print(f"  {test.__name__}: ok")
    print("tool contract: ok")


if __name__ == "__main__":
    main()
