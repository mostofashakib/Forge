#!/usr/bin/env python3
"""SQLite-backed Slack service used by Harbor task environments.

This module is the service layer only: schema, seeding, tool functions, and
the CLI. The in-container agents that drive it live in
:mod:`fleet.environments.slack.reference_agents`.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

from fleet.environments.sqlite_common import ToolError, connect, query_rows, remove_pycaches
from fleet.environments.slack.search import matches_all_terms, message_haystack, search_terms
from fleet.environments.slack.seed import (
    SLACK_CHANNELS,
    SLACK_CHAT_PARTICIPANTS,
    SLACK_CHATS,
    SLACK_EDITED_MESSAGES,
    SLACK_MESSAGES,
    SLACK_THREAD_PARENTS,
    SLACK_USERS,
    START_MS,
    STEP_MS,
    build_slack_memberships,
)


def seed_database(db_path: Path, snapshot_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE users (
                user_id TEXT PRIMARY KEY,
                id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                email TEXT NOT NULL,
                role TEXT NOT NULL,
                team TEXT NOT NULL,
                handle TEXT NOT NULL
            );
            CREATE TABLE channels (
                channel_id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                is_private INTEGER NOT NULL,
                owner_id TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL
            );
            CREATE TABLE memberships (
                membership_id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                joined_at_ms INTEGER NOT NULL
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                author_id TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                thread_parent_id TEXT,
                edited_at_ms INTEGER,
                deleted INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE reactions (
                reaction_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                emoji TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL
            );
            CREATE TABLE chats (
                chat_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT,
                created_at_ms INTEGER NOT NULL
            );
            CREATE TABLE chat_participants (
                chat_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                PRIMARY KEY (chat_id, user_id),
                FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            );
            """
        )
        # Seed users (inserting user_id as both user_id and id columns)
        seeded_users = [(u[0], u[0], u[1], u[2], u[3], u[4], u[5]) for u in SLACK_USERS]
        connection.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)", seeded_users)
        connection.executemany(
            "INSERT INTO channels VALUES (?, ?, ?, ?, ?)",
            [(channel_id, name, int(is_private), owner_id, created_at_ms) for channel_id, name, is_private, owner_id, created_at_ms in SLACK_CHANNELS],
        )
        connection.executemany("INSERT INTO memberships VALUES (?, ?, ?, ?, ?)", build_slack_memberships())

        messages = []
        for message_id, channel_id, author_id, body, offset in SLACK_MESSAGES:
            thread_parent = SLACK_THREAD_PARENTS.get(message_id)
            edited_offset = SLACK_EDITED_MESSAGES.get(message_id)
            edited_at = START_MS + edited_offset * STEP_MS if edited_offset is not None else None
            messages.append((message_id, channel_id, author_id, body, START_MS + offset * STEP_MS, thread_parent, edited_at, 0))

        connection.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)", messages)

        connection.executemany("INSERT INTO chats VALUES (?, ?, ?, ?)", SLACK_CHATS)
        connection.executemany("INSERT INTO chat_participants VALUES (?, ?)", SLACK_CHAT_PARTICIPANTS)

        connection.commit()
        snapshot_path.write_text("\n".join(connection.iterdump()), encoding="utf-8")


def teardown_database(db_path: Path, snapshot_path: Path) -> None:
    remove_pycaches()
    db_shm = db_path.with_name(db_path.name + "-shm")
    db_wal = db_path.with_name(db_path.name + "-wal")
    db_journal = db_path.with_name(db_path.name + "-journal")
    for p in [db_path, db_shm, db_wal, db_journal, snapshot_path]:
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    seed_database(db_path, snapshot_path)


def require_user(connection: sqlite3.Connection, user_id: str) -> dict[str, Any]:
    user = connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if user is None:
        raise ToolError("user_not_found", "permission_denied", "Acting user was not found.")
    return dict(user)


# ---------------------------------------------------------------------------
# Internal helpers shared by the tool functions
# ---------------------------------------------------------------------------

def _require_body(body: str, label: str) -> str:
    body = body.strip()
    if not body:
        raise ToolError("invalid_arguments", "validation_error", f"{label} is required.")
    return body


def _is_channel_member(connection: sqlite3.Connection, channel_id: str, user_id: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM memberships WHERE channel_id = ? AND user_id = ?",
        (channel_id, user_id),
    ).fetchone() is not None


def _is_chat_participant(connection: sqlite3.Connection, chat_id: str, user_id: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM chat_participants WHERE chat_id = ? AND user_id = ?",
        (chat_id, user_id),
    ).fetchone() is not None


def _require_container_member(
    connection: sqlite3.Connection,
    container_id: str,
    actor_id: str,
    missing_message: str,
) -> None:
    """The container is a channel or a chat; enforce the matching access rule."""
    channel = connection.execute("SELECT * FROM channels WHERE channel_id = ?", (container_id,)).fetchone()
    if channel is not None:
        if channel["is_private"] and not _is_channel_member(connection, container_id, actor_id):
            raise ToolError("permission_denied", "permission_denied", "User is not a member of the private channel.")
        return
    chat = connection.execute("SELECT * FROM chats WHERE chat_id = ?", (container_id,)).fetchone()
    if chat is None:
        raise ToolError("channel_not_found", "not_found", missing_message)
    if not _is_chat_participant(connection, container_id, actor_id):
        raise ToolError("permission_denied", "permission_denied", "User is not a participant of this chat.")


def _next_message_ts(connection: sqlite3.Connection) -> int:
    max_ts = connection.execute("SELECT MAX(created_at_ms) FROM messages").fetchone()[0]
    if max_ts is None:
        max_ts = START_MS
    return max_ts + 1000


def _insert_message(
    connection: sqlite3.Connection,
    container_id: str,
    actor_id: str,
    body: str,
    thread_parent_id: str | None = None,
) -> dict[str, Any]:
    msg_count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    message_id = f"MSG{msg_count + 1:03d}"
    now_ms = _next_message_ts(connection)
    connection.execute(
        "INSERT INTO messages (message_id, channel_id, author_id, body, created_at_ms, thread_parent_id, deleted) VALUES (?, ?, ?, ?, ?, ?, 0)",
        (message_id, container_id, actor_id, body, now_ms, thread_parent_id),
    )
    connection.commit()
    msg = dict(connection.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,)).fetchone())
    msg["deleted"] = bool(msg["deleted"])
    return {"id": message_id, "message": msg}


def _chat_with_participants(connection: sqlite3.Connection, chat_id: str) -> dict[str, Any]:
    chat_row = dict(connection.execute("SELECT * FROM chats WHERE chat_id = ?", (chat_id,)).fetchone())
    chat_row["participants"] = query_rows(
        connection,
        "SELECT u.* FROM chat_participants cp JOIN users u ON cp.user_id = u.user_id WHERE cp.chat_id = ? ORDER BY u.user_id",
        (chat_id,),
    )
    return chat_row


def _find_or_create_dm(connection: sqlite3.Connection, actor_id: str, recipient_id: str) -> str:
    recipient = connection.execute("SELECT * FROM users WHERE user_id = ?", (recipient_id,)).fetchone()
    if recipient is None:
        raise ToolError("user_not_found", "not_found", "Recipient user was not found.")
    existing = connection.execute(
        """
        SELECT c.chat_id FROM chats c
        JOIN chat_participants cp1 ON c.chat_id = cp1.chat_id AND cp1.user_id = ?
        JOIN chat_participants cp2 ON c.chat_id = cp2.chat_id AND cp2.user_id = ?
        WHERE c.type = 'dm'
        """,
        (actor_id, recipient_id),
    ).fetchone()
    if existing:
        return existing[0]
    chat_count = connection.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
    chat_id = f"D{chat_count + 1:03d}"
    now_ms = START_MS + 1000
    connection.execute("INSERT INTO chats VALUES (?, 'dm', NULL, ?)", (chat_id, now_ms))
    connection.execute("INSERT INTO chat_participants VALUES (?, ?)", (chat_id, actor_id))
    if recipient_id != actor_id:
        connection.execute("INSERT INTO chat_participants VALUES (?, ?)", (chat_id, recipient_id))
    connection.commit()
    return chat_id


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

def export_state(db_path: Path) -> dict[str, Any]:
    with connect(db_path) as connection:
        users = query_rows(connection, "SELECT * FROM users ORDER BY user_id")
        channels = query_rows(connection, "SELECT * FROM channels ORDER BY channel_id")
        memberships = query_rows(connection, "SELECT * FROM memberships ORDER BY membership_id")
        messages = query_rows(connection, "SELECT * FROM messages ORDER BY message_id")
        reactions = query_rows(connection, "SELECT * FROM reactions ORDER BY reaction_id")

        for c in channels:
            c["is_private"] = bool(c["is_private"])
        for msg in messages:
            msg["deleted"] = bool(msg["deleted"])

        chats = query_rows(connection, "SELECT * FROM chats ORDER BY chat_id")
        for chat in chats:
            chat["participants"] = query_rows(
                connection,
                """
                SELECT u.*
                FROM chat_participants cp
                JOIN users u ON cp.user_id = u.user_id
                WHERE cp.chat_id = ?
                ORDER BY u.user_id
                """,
                (chat["chat_id"],),
            )

        return {
            "users": users,
            "channels": channels,
            "memberships": memberships,
            "messages": messages,
            "reactions": reactions,
            "chats": chats,
        }


def list_channels(db_path: Path, actor_id: str) -> dict[str, Any]:
    with connect(db_path) as connection:
        require_user(connection, actor_id)
        channels = query_rows(
            connection,
            """
            SELECT c.* FROM channels c
            WHERE c.is_private = 0
            OR EXISTS (
                SELECT 1 FROM memberships m
                WHERE m.channel_id = c.channel_id AND m.user_id = ?
            )
            ORDER BY c.name
            """,
            (actor_id,),
        )
        for c in channels:
            c["is_private"] = bool(c["is_private"])
        return {"channels": channels}


def search_messages(db_path: Path, query: str, actor_id: str | None = None) -> dict[str, Any]:
    terms = search_terms(query)
    with connect(db_path) as connection:
        all_messages = query_rows(
            connection,
            """
            SELECT m.*,
                   c.name AS channel_name,
                   c.is_private AS channel_is_private,
                   g.name AS group_name,
                   g.type AS chat_type,
                   u.handle AS author_handle,
                   u.display_name AS author_display_name
            FROM messages m
            LEFT JOIN channels c ON c.channel_id = m.channel_id
            LEFT JOIN chats g ON g.chat_id = m.channel_id
            JOIN users u ON u.user_id = m.author_id
            WHERE m.deleted = 0
            ORDER BY m.created_at_ms DESC
            """
        )

        matches = []
        for msg in all_messages:
            # Without an actor (trusted CLI calls), everything is visible.
            is_visible = True
            if actor_id:
                if msg["channel_name"] is not None:
                    is_visible = not bool(msg["channel_is_private"]) or _is_channel_member(
                        connection, msg["channel_id"], actor_id
                    )
                elif msg["chat_type"] is not None:
                    is_visible = _is_chat_participant(connection, msg["channel_id"], actor_id)
            if not is_visible:
                continue

            location_name = msg["channel_name"] or msg["group_name"] or ""
            haystack = message_haystack(
                str(msg["body"]),
                str(location_name),
                str(msg["author_display_name"]),
                str(msg["author_handle"]),
            )
            if matches_all_terms(haystack, terms):
                matches.append(
                    {
                        "message_id": msg["message_id"],
                        "channel_id": msg["channel_id"],
                        "author_id": msg["author_id"],
                        "body": msg["body"],
                        "created_at_ms": msg["created_at_ms"],
                        "thread_parent_id": msg["thread_parent_id"],
                        "edited_at_ms": msg["edited_at_ms"],
                        "deleted": bool(msg["deleted"]),
                        "channel_name": location_name,
                        "author_handle": msg["author_handle"],
                    }
                )

        return {"count": len(matches), "messages": matches}


def get_channel_messages(db_path: Path, channel_id: str, actor_id: str | None = None) -> dict[str, Any]:
    with connect(db_path) as connection:
        channel = connection.execute("SELECT * FROM channels WHERE channel_id = ?", (channel_id,)).fetchone()
        if channel is None:
            raise ToolError("channel_not_found", "not_found", "Channel was not found.")
        channel = dict(channel)
        channel["is_private"] = bool(channel["is_private"])

        if actor_id:
            require_user(connection, actor_id)
            if channel["is_private"] and not _is_channel_member(connection, channel_id, actor_id):
                raise ToolError("permission_denied", "permission_denied", "User is not a member of the private channel.")

        messages = query_rows(
            connection,
            """
            SELECT m.*, c.name AS channel_name, u.handle AS author_handle
            FROM messages m
            JOIN channels c ON c.channel_id = m.channel_id
            JOIN users u ON u.user_id = m.author_id
            WHERE m.channel_id = ? AND m.deleted = 0
            ORDER BY m.created_at_ms ASC
            """,
            (channel_id,),
        )
        for msg in messages:
            msg["deleted"] = bool(msg["deleted"])
        return {"channel": channel, "count": len(messages), "messages": messages}


# ---------------------------------------------------------------------------
# Mutation tools
# ---------------------------------------------------------------------------

def post_message(db_path: Path, channel_id: str, body: str, actor_id: str) -> dict[str, Any]:
    body = _require_body(body, "Message body")
    with connect(db_path) as connection:
        require_user(connection, actor_id)
        channel = connection.execute("SELECT * FROM channels WHERE channel_id = ?", (channel_id,)).fetchone()
        if channel is None:
            raise ToolError("channel_not_found", "not_found", "Channel was not found.")
        if channel["is_private"] and not _is_channel_member(connection, channel_id, actor_id):
            raise ToolError("permission_denied", "permission_denied", "User is not a member of the private channel.")
        return _insert_message(connection, channel_id, actor_id, body)


def reply_to_thread(db_path: Path, thread_parent_id: str, body: str, actor_id: str) -> dict[str, Any]:
    body = _require_body(body, "Reply body")
    with connect(db_path) as connection:
        require_user(connection, actor_id)
        parent = connection.execute(
            "SELECT * FROM messages WHERE message_id = ? AND deleted = 0", (thread_parent_id,)
        ).fetchone()
        if parent is None:
            raise ToolError("invalid_thread_reference", "not_found", "Thread parent message was not found.")
        channel_id = parent["channel_id"]
        _require_container_member(connection, channel_id, actor_id, "Thread parent channel/chat was not found.")
        return _insert_message(connection, channel_id, actor_id, body, thread_parent_id=thread_parent_id)


def update_message(db_path: Path, message_id: str, body: str, actor_id: str) -> dict[str, Any]:
    body = _require_body(body, "Message body")
    with connect(db_path) as connection:
        require_user(connection, actor_id)
        message = connection.execute("SELECT * FROM messages WHERE message_id = ? AND deleted = 0", (message_id,)).fetchone()
        if message is None:
            raise ToolError("message_not_found", "not_found", "Message was not found.")
        message = dict(message)
        if message["author_id"] != actor_id:
            raise ToolError("permission_denied", "permission_denied", "Only the original author can edit a message.")

        # Virtual clock: advance strictly past every timestamp in the table
        # (creations and prior edits), so each edit gets a distinct, larger
        # edited_at_ms — including repeat edits of the same message.
        max_created = connection.execute("SELECT MAX(created_at_ms) FROM messages").fetchone()[0] or START_MS
        max_edited = connection.execute("SELECT MAX(edited_at_ms) FROM messages").fetchone()[0] or 0
        now_ms = max(max_created, max_edited) + 1000

        connection.execute(
            "UPDATE messages SET body = ?, edited_at_ms = ? WHERE message_id = ?",
            (body, now_ms, message_id)
        )
        connection.commit()
        updated = dict(connection.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,)).fetchone())
        updated["deleted"] = bool(updated["deleted"])
        return {"id": message_id, "message": updated}


def add_reaction(db_path: Path, message_id: str, emoji: str, actor_id: str) -> dict[str, Any]:
    emoji = _require_body(emoji, "Emoji")
    with connect(db_path) as connection:
        require_user(connection, actor_id)
        message = connection.execute("SELECT * FROM messages WHERE message_id = ? AND deleted = 0", (message_id,)).fetchone()
        if message is None:
            raise ToolError("message_not_found", "not_found", "Message was not found.")
        message = dict(message)
        _require_container_member(connection, message["channel_id"], actor_id, "Message channel/chat was not found.")

        duplicate = connection.execute(
            "SELECT 1 FROM reactions WHERE message_id = ? AND user_id = ? AND emoji = ?",
            (message_id, actor_id, emoji)
        ).fetchone()
        if duplicate:
            raise ToolError("duplicate_reaction", "conflict", "Reaction already exists.")

        reaction_count = connection.execute("SELECT COUNT(*) FROM reactions").fetchone()[0]
        reaction_id = f"REA{reaction_count + 1:03d}"
        now_ms = _next_message_ts(connection)

        connection.execute(
            "INSERT INTO reactions VALUES (?, ?, ?, ?, ?)",
            (reaction_id, message_id, actor_id, emoji, now_ms)
        )
        connection.commit()
        reaction = dict(connection.execute("SELECT * FROM reactions WHERE reaction_id = ?", (reaction_id,)).fetchone())
        return {"id": reaction_id, "reaction": reaction}


def create_channel(db_path: Path, name: str, is_private: bool, actor_id: str) -> dict[str, Any]:
    name = _require_body(name, "Channel name")
    with connect(db_path) as connection:
        require_user(connection, actor_id)
        duplicate = connection.execute("SELECT 1 FROM channels WHERE name = ?", (name,)).fetchone()
        if duplicate:
            raise ToolError("duplicate_channel_name", "conflict", "Channel name already exists.")

        channel_count = connection.execute("SELECT COUNT(*) FROM channels").fetchone()[0]
        channel_id = f"C{channel_count + 1:03d}"

        max_ts = connection.execute("SELECT MAX(created_at_ms) FROM channels").fetchone()[0] or START_MS
        now_ms = max_ts + 1000

        connection.execute(
            "INSERT INTO channels VALUES (?, ?, ?, ?, ?)",
            (channel_id, name, int(is_private), actor_id, now_ms)
        )

        membership_count = connection.execute("SELECT COUNT(*) FROM memberships").fetchone()[0]
        membership_id = f"MBR{membership_count + 1:03d}"
        connection.execute(
            "INSERT INTO memberships VALUES (?, ?, ?, ?, ?)",
            (membership_id, channel_id, actor_id, "owner", now_ms)
        )
        connection.commit()

        channel = dict(connection.execute("SELECT * FROM channels WHERE channel_id = ?", (channel_id,)).fetchone())
        channel["is_private"] = bool(channel["is_private"])
        return {"id": channel_id, "channel": channel}


def create_group(db_path: Path, name: str, participants: list[str], actor_id: str) -> dict[str, Any]:
    name = _require_body(name, "Group name")
    with connect(db_path) as connection:
        require_user(connection, actor_id)

        all_participants = list(set(participants + [actor_id]))
        for p_id in all_participants:
            if connection.execute("SELECT 1 FROM users WHERE user_id = ?", (p_id,)).fetchone() is None:
                raise ToolError("user_not_found", "not_found", f"User {p_id} was not found.")

        chat_count = connection.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
        chat_id = f"G{chat_count + 1:03d}"
        now_ms = START_MS + 1000

        connection.execute("INSERT INTO chats VALUES (?, 'group', ?, ?)", (chat_id, name, now_ms))
        for p_id in all_participants:
            connection.execute("INSERT INTO chat_participants VALUES (?, ?)", (chat_id, p_id))
        connection.commit()

        return {"id": chat_id, "chat": _chat_with_participants(connection, chat_id)}


def change_channel_name(db_path: Path, channel_id: str, new_name: str, actor_id: str) -> dict[str, Any]:
    new_name = _require_body(new_name, "New channel name")
    with connect(db_path) as connection:
        require_user(connection, actor_id)
        channel = connection.execute("SELECT * FROM channels WHERE channel_id = ?", (channel_id,)).fetchone()
        if channel is None:
            raise ToolError("channel_not_found", "not_found", "Channel was not found.")
        channel = dict(channel)
        if channel["owner_id"] != actor_id:
            raise ToolError("permission_denied", "permission_denied", "Only the owner can rename a channel.")

        duplicate = connection.execute("SELECT 1 FROM channels WHERE name = ? AND channel_id != ?", (new_name, channel_id)).fetchone()
        if duplicate:
            raise ToolError("duplicate_channel_name", "conflict", "Channel name already exists.")

        connection.execute(
            "UPDATE channels SET name = ? WHERE channel_id = ?",
            (new_name, channel_id)
        )
        connection.commit()
        updated = dict(connection.execute("SELECT * FROM channels WHERE channel_id = ?", (channel_id,)).fetchone())
        updated["is_private"] = bool(updated["is_private"])
        return {"id": channel_id, "channel": updated}


def change_group_name(db_path: Path, group_id: str, new_name: str, actor_id: str) -> dict[str, Any]:
    new_name = _require_body(new_name, "New group name")
    with connect(db_path) as connection:
        require_user(connection, actor_id)
        chat = connection.execute("SELECT * FROM chats WHERE chat_id = ? AND type = 'group'", (group_id,)).fetchone()
        if chat is None:
            raise ToolError("group_not_found", "not_found", "Group chat was not found.")
        if not _is_chat_participant(connection, group_id, actor_id):
            raise ToolError("permission_denied", "permission_denied", "User is not a participant of this group chat.")

        connection.execute("UPDATE chats SET name = ? WHERE chat_id = ?", (new_name, group_id))
        connection.commit()
        return {"id": group_id, "chat": _chat_with_participants(connection, group_id)}


def create_dm_message(db_path: Path, recipient_id: str, actor_id: str) -> dict[str, Any]:
    with connect(db_path) as connection:
        require_user(connection, actor_id)
        chat_id = _find_or_create_dm(connection, actor_id, recipient_id)
        return {"id": chat_id, "chat": _chat_with_participants(connection, chat_id)}


def send_dm_message(db_path: Path, recipient_id: str | None, chat_id: str | None, body: str, actor_id: str) -> dict[str, Any]:
    body = _require_body(body, "Message body")
    if not recipient_id and not chat_id:
        raise ToolError("invalid_arguments", "validation_error", "Either recipient_id or chat_id is required.")

    with connect(db_path) as connection:
        require_user(connection, actor_id)

        if recipient_id:
            target_chat_id = _find_or_create_dm(connection, actor_id, recipient_id)
        else:
            target_chat_id = chat_id
            chat = connection.execute("SELECT * FROM chats WHERE chat_id = ? AND type = 'dm'", (target_chat_id,)).fetchone()
            if chat is None:
                raise ToolError("chat_not_found", "not_found", "DM chat was not found.")
            if not _is_chat_participant(connection, target_chat_id, actor_id):
                raise ToolError("permission_denied", "permission_denied", "User is not a participant of this DM.")

        return _insert_message(connection, target_chat_id, actor_id, body)


def send_group_message(db_path: Path, group_id: str, body: str, actor_id: str) -> dict[str, Any]:
    body = _require_body(body, "Message body")
    with connect(db_path) as connection:
        require_user(connection, actor_id)
        chat = connection.execute("SELECT * FROM chats WHERE chat_id = ? AND type = 'group'", (group_id,)).fetchone()
        if chat is None:
            raise ToolError("group_not_found", "not_found", "Group chat was not found.")
        if not _is_chat_participant(connection, group_id, actor_id):
            raise ToolError("permission_denied", "permission_denied", "User is not a participant of this group chat.")
        return _insert_message(connection, group_id, actor_id, body)


def change_user_display_name(db_path: Path, user_id: str, new_display_name: str, actor_id: str) -> dict[str, Any]:
    new_display_name = _require_body(new_display_name, "New display name")
    with connect(db_path) as connection:
        require_user(connection, actor_id)
        user = connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if user is None:
            raise ToolError("user_not_found", "not_found", "User was not found.")
        actor = dict(connection.execute("SELECT * FROM users WHERE user_id = ?", (actor_id,)).fetchone())
        if actor_id != user_id and actor["role"] != "admin":
            raise ToolError("permission_denied", "permission_denied", "Permission denied to change other user's display name.")

        connection.execute(
            "UPDATE users SET display_name = ? WHERE user_id = ?",
            (new_display_name, user_id)
        )
        connection.commit()
        updated = dict(connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone())
        return {"id": user_id, "user": updated}


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

_TOOL_HANDLERS: dict[str, Callable[[Path, dict[str, Any], str], dict[str, Any]]] = {
    "search_messages": lambda db, p, actor: search_messages(db, str(p.get("query", "")), actor),
    "get_channel_messages": lambda db, p, actor: get_channel_messages(db, str(p.get("channel_id", "")), actor),
    "list_channels": lambda db, p, actor: list_channels(db, actor),
    "post_message": lambda db, p, actor: post_message(db, str(p.get("channel_id", "")), str(p.get("body", "")), actor),
    "slack.reply_to_thread": lambda db, p, actor: reply_to_thread(db, str(p.get("thread_parent_id", "")), str(p.get("body", "")), actor),
    "update_message": lambda db, p, actor: update_message(db, str(p.get("message_id", "")), str(p.get("body", "")), actor),
    "add_reaction": lambda db, p, actor: add_reaction(db, str(p.get("message_id", "")), str(p.get("emoji", "")), actor),
    "create_channel": lambda db, p, actor: create_channel(db, str(p.get("name", "")), bool(p.get("is_private", False)), actor),
    "create_group": lambda db, p, actor: create_group(db, str(p.get("name", "")), list(p.get("participants", [])), actor),
    "change_channel_name": lambda db, p, actor: change_channel_name(db, str(p.get("channel_id", "")), str(p.get("new_name", "")), actor),
    "change_group_name": lambda db, p, actor: change_group_name(db, str(p.get("group_id") or p.get("chat_id") or ""), str(p.get("new_name", "")), actor),
    "create_dm_message": lambda db, p, actor: create_dm_message(db, str(p.get("recipient_id", "")), actor),
    "send_dm_message": lambda db, p, actor: send_dm_message(db, p.get("recipient_id"), p.get("chat_id"), str(p.get("body", "")), actor),
    "send_group_message": lambda db, p, actor: send_group_message(db, str(p.get("group_id") or p.get("chat_id") or ""), str(p.get("body", "")), actor),
    "change_user_display_name": lambda db, p, actor: change_user_display_name(db, str(p.get("user_id", "")), str(p.get("new_display_name", "")), actor),
}


TOOL_NAMES: tuple[str, ...] = tuple(_TOOL_HANDLERS)


def execute_tool(db_path: Path, tool_name: str, input_payload: dict[str, Any], actor_id: str = "U002") -> dict[str, Any]:
    handler = _TOOL_HANDLERS.get(tool_name)
    if handler is None:
        raise RuntimeError(f"Unknown tool: {tool_name}")
    return handler(db_path, input_payload, actor_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from fleet.agents.model_adapters import DEFAULT_MODEL

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "seed", "teardown", "state", "search_messages", "get_channel_messages",
        "list_channels", "post_message", "reply_to_thread", "update_message", "add_reaction",
        "create_channel", "create_group", "change_channel_name", "change_group_name",
        "create_dm_message", "send_dm_message", "send_group_message", "change_user_display_name",
        "run-reference-agent", "run-ollama-agent", "execute_tool"
    ])
    parser.add_argument("--db", default="/app/slack.db")
    parser.add_argument("--snapshot", default="/app/slack_seed_snapshot.sql")
    parser.add_argument("--query", default="")
    parser.add_argument("--channel-id", default="")
    parser.add_argument("--message-id", default="")
    parser.add_argument("--thread-parent-id", default="")
    parser.add_argument("--body", default="")
    parser.add_argument("--emoji", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--is-private", action="store_true")
    parser.add_argument("--participants", default="")
    parser.add_argument("--new-name", default="")
    parser.add_argument("--recipient-id", default="")
    parser.add_argument("--chat-id", default="")
    parser.add_argument("--group-id", default="")
    parser.add_argument("--user-id", default="")
    parser.add_argument("--new-display-name", default="")
    parser.add_argument("--actor-id", default="U002")

    parser.add_argument("--tool-name", default="")
    parser.add_argument("--input-payload", default="")
    parser.add_argument("tool_name_pos", nargs="?", default="")
    parser.add_argument("input_payload_pos", nargs="?", default="")

    parser.add_argument("--instruction", default="")
    parser.add_argument("--trajectory", default="/logs/agent/trajectory.json")
    parser.add_argument("--transcript", default="/logs/agent/trajectory.txt")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="http://host.docker.internal:11434")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    args = parser.parse_args()

    db_path = Path(args.db)
    snapshot_path = Path(args.snapshot)

    def run_execute_tool() -> dict[str, Any]:
        tool_name = args.tool_name or args.tool_name_pos
        payload_str = args.input_payload or args.input_payload_pos or "{}"
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON input payload: {payload_str}") from exc
        return execute_tool(db_path, tool_name, payload, args.actor_id)

    def run_reference() -> None:
        from fleet.environments.slack.reference_agents import run_reference_agent

        run_reference_agent(db_path, args.instruction, Path(args.trajectory), Path(args.transcript))

    def run_ollama() -> None:
        from fleet.environments.slack.reference_agents import run_ollama_agent

        run_ollama_agent(
            db_path=db_path,
            instruction=args.instruction,
            trajectory_path=Path(args.trajectory),
            transcript_path=Path(args.transcript),
            model=args.model,
            host=args.host,
            timeout_sec=args.timeout_sec,
        )

    handlers: dict[str, Callable[[], dict[str, Any] | None]] = {
        "seed": lambda: seed_database(db_path, snapshot_path),
        "teardown": lambda: teardown_database(db_path, snapshot_path),
        "state": lambda: export_state(db_path),
        "search_messages": lambda: search_messages(db_path, args.query, args.actor_id),
        "get_channel_messages": lambda: get_channel_messages(db_path, args.channel_id, args.actor_id),
        "list_channels": lambda: list_channels(db_path, args.actor_id),
        "post_message": lambda: post_message(db_path, args.channel_id, args.body, args.actor_id),
        "reply_to_thread": lambda: reply_to_thread(db_path, args.thread_parent_id, args.body, args.actor_id),
        "update_message": lambda: update_message(db_path, args.message_id, args.body, args.actor_id),
        "add_reaction": lambda: add_reaction(db_path, args.message_id, args.emoji, args.actor_id),
        "create_channel": lambda: create_channel(db_path, args.name, args.is_private, args.actor_id),
        "create_group": lambda: create_group(
            db_path, args.name, [p.strip() for p in args.participants.split(",") if p.strip()], args.actor_id
        ),
        "change_channel_name": lambda: change_channel_name(db_path, args.channel_id, args.new_name, args.actor_id),
        "change_group_name": lambda: change_group_name(db_path, args.group_id or args.chat_id, args.new_name, args.actor_id),
        "create_dm_message": lambda: create_dm_message(db_path, args.recipient_id, args.actor_id),
        "send_dm_message": lambda: send_dm_message(
            db_path, args.recipient_id or None, args.chat_id or None, args.body, args.actor_id
        ),
        "send_group_message": lambda: send_group_message(db_path, args.group_id or args.chat_id, args.body, args.actor_id),
        "change_user_display_name": lambda: change_user_display_name(
            db_path, args.user_id, args.new_display_name, args.actor_id
        ),
        "run-reference-agent": run_reference,
        "run-ollama-agent": run_ollama,
        "execute_tool": run_execute_tool,
    }
    result = handlers[args.command]()
    if result is not None:
        print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
