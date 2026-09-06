"""Model-facing JSON-schema definitions for the Slack world actions."""

from __future__ import annotations

from typing import Any


def _tool(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


TEXT = {"type": "string"}
BOOLEAN = {"type": "boolean"}
INTEGER = {"type": "integer", "minimum": 1, "maximum": 100, "default": 50}
CHANNEL_HISTORY_LIMIT = {"type": "integer", "minimum": 1, "maximum": 50, "default": 50}
STRING_LIST = {"type": "array", "items": {"type": "string"}}
PAGINATION = {"cursor": TEXT, "limit": INTEGER}
CHANNEL_HISTORY_PAGINATION = {"cursor": TEXT, "limit": CHANNEL_HISTORY_LIMIT}

TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    _tool(
        "list_channels",
        "List visible Slack channel metadata, 50 at a time by default. Pass next_cursor to continue.",
        PAGINATION,
        [],
    ),
    _tool(
        "list_users",
        "List workspace user metadata, 50 at a time by default. Pass next_cursor to continue.",
        PAGINATION,
        [],
    ),
    _tool("list_chats", "List direct and group chats in which the acting user participates.", {}, []),
    _tool("list_user_groups", "List workspace user groups and their members.", {}, []),
    _tool(
        "search_messages",
        "Search visible messages without dumping conversations. Results include channel/thread IDs "
        "and at most one surrounding message on either side; 50 matches are returned by default.",
        {"query": TEXT, **PAGINATION},
        ["query"],
    ),
    _tool(
        "get_channel_messages",
        "Read one channel or chat scrollback page: at most 50 top-level messages, newest first. "
        "Pass the response's next_cursor on the next call to load the next older page. "
        "Omitting cursor reloads the newest page. Thread replies are not included; a message that has a thread carries thread_ts, which get_thread_replies takes. Fields that would be empty are omitted, so a message with no thread_ts has no replies.",
        {"channel_id": TEXT, **CHANNEL_HISTORY_PAGINATION},
        ["channel_id"],
    ),
    _tool(
        "get_thread_replies",
        "Read one complete thread, root first followed by all replies in chronological order.",
        {"channel_id": TEXT, "thread_ts": TEXT},
        ["channel_id", "thread_ts"],
    ),
    _tool(
        "mark_conversation_read",
        "Move the acting user's read cursor to a message ts, or to the latest message when ts is omitted.",
        {"conversation_id": TEXT, "ts": TEXT},
        ["conversation_id"],
    ),
    _tool(
        "post_message",
        "Post a top-level message to a channel or chat.",
        {"channel_id": TEXT, "body": TEXT},
        ["channel_id", "body"],
    ),
    _tool(
        "reply_to_thread",
        "Reply to a message. Pass the thread root to answer the thread, or any "
        "reply inside it to answer that message; either way the reply joins the "
        "same thread and records which message it answered.",
        {"thread_parent_id": TEXT, "body": TEXT},
        ["thread_parent_id", "body"],
    ),
    _tool(
        "edit_message",
        "Edit one of the acting user's existing messages and record an edit timestamp.",
        {"message_id": TEXT, "body": TEXT},
        ["message_id", "body"],
    ),
    _tool(
        "add_reaction",
        "Add an emoji reaction as the acting user.",
        {"message_id": TEXT, "emoji": TEXT},
        ["message_id", "emoji"],
    ),
    _tool(
        "create_channel",
        "Create a public or private Slack channel.",
        {"name": TEXT, "is_private": BOOLEAN},
        ["name"],
    ),
    _tool(
        "create_group",
        "Create a named group chat with the supplied user IDs.",
        {"name": TEXT, "participants": STRING_LIST},
        ["name", "participants"],
    ),
    _tool(
        "edit_channel_name",
        "Rename a Slack channel owned by the acting user.",
        {"channel_id": TEXT, "new_name": TEXT},
        ["channel_id", "new_name"],
    ),
    _tool(
        "edit_group_chat_name",
        "Rename a group chat in which the acting user participates.",
        {"group_id": TEXT, "new_name": TEXT},
        ["group_id", "new_name"],
    ),
    _tool(
        "create_dm_message",
        "Create or locate a direct-message chat with another user, or with "
        "yourself.",
        {"recipient_id": TEXT},
        ["recipient_id"],
    ),
    _tool(
        "send_dm_message",
        "Send a direct message using either an existing chat ID or recipient ID. "
        "Your own user ID is a valid recipient: that is the chat with yourself.",
        {"recipient_id": TEXT, "chat_id": TEXT, "body": TEXT},
        ["body"],
    ),
    _tool(
        "send_group_message",
        "Send a message to an existing group chat.",
        {"group_id": TEXT, "body": TEXT},
        ["group_id", "body"],
    ),
    _tool(
        "edit_display_name",
        "Edit your own display name; workspace admins may edit any user's display name.",
        {"user_id": TEXT, "new_display_name": TEXT},
        ["user_id", "new_display_name"],
    ),
    _tool(
        "tag_people",
        "Post a message with structured person mentions to a channel or participating chat.",
        {"channel_id": TEXT, "user_ids": STRING_LIST, "body": TEXT},
        ["channel_id", "user_ids", "body"],
    ),
    _tool(
        "tag_here",
        "Post a channel message with @here; the acting user must be a channel member.",
        {"channel_id": TEXT, "body": TEXT},
        ["channel_id", "body"],
    ),
    _tool(
        "tag_everyone",
        "Post an @everyone channel message; requires workspace-admin or channel-owner permission.",
        {"channel_id": TEXT, "body": TEXT},
        ["channel_id", "body"],
    ),
    _tool(
        "tag_user_group",
        "Post a message with a structured workspace user-group mention.",
        {"channel_id": TEXT, "user_group_id": TEXT, "body": TEXT},
        ["channel_id", "user_group_id", "body"],
    ),
    _tool(
        "invite_to_channel",
        "Invite workspace users to a channel in which the acting user is already a member.",
        {"channel_id": TEXT, "user_ids": STRING_LIST},
        ["channel_id", "user_ids"],
    ),
    _tool(
        "add_group_chat_participants",
        "Add workspace users to a group chat in which the acting user participates.",
        {"group_id": TEXT, "user_ids": STRING_LIST},
        ["group_id", "user_ids"],
    ),
    # -- messages ----------------------------------------------------------
    _tool(
        "delete_message",
        "Delete one of your own messages. It leaves history, threads, search "
        "and pins; a workspace admin may delete any message.",
        {"message_id": TEXT},
        ["message_id"],
    ),
    # -- threads -----------------------------------------------------------
    _tool(
        "follow_thread",
        "Follow a thread so later replies in it notify you. Pass the thread "
        "root or any reply inside it. Replying to a thread follows it too.",
        {"thread_parent_id": TEXT},
        ["thread_parent_id"],
    ),
    _tool("unfollow_thread", "Stop following a thread.", {"thread_parent_id": TEXT}, ["thread_parent_id"]),
    _tool("list_followed_threads", "List the threads you follow, with reply and unread counts.", {}, []),
    # -- reactions ---------------------------------------------------------
    _tool(
        "remove_reaction",
        "Remove your own reaction from a message; other people's are untouched.",
        {"message_id": TEXT, "emoji": TEXT},
        ["message_id", "emoji"],
    ),
    # -- channels ----------------------------------------------------------
    _tool("join_channel", "Join a public channel, which makes it readable.", {"channel_id": TEXT}, ["channel_id"]),
    _tool("leave_channel", "Leave a channel; you lose read access to it.", {"channel_id": TEXT}, ["channel_id"]),
    _tool(
        "archive_channel",
        "Archive a channel you own, or any channel if you are a workspace "
        "admin. History stays readable; new messages are refused.",
        {"channel_id": TEXT},
        ["channel_id"],
    ),
    # -- membership --------------------------------------------------------
    _tool("list_channel_members", "List a channel's members with their roles and presence.", {"channel_id": TEXT}, ["channel_id"]),
    _tool(
        "remove_from_channel",
        "Remove members from a channel you own, or any channel if you are a "
        "workspace admin. The channel owner cannot be removed.",
        {"channel_id": TEXT, "user_ids": STRING_LIST},
        ["channel_id", "user_ids"],
    ),
    # -- your own presence and status --------------------------------------
    _tool("set_presence", "Set your presence to 'active' or 'away'.", {"presence": TEXT}, ["presence"]),
    _tool(
        "set_status",
        "Set your own status text; workspace admins may set anyone's.",
        {"status_text": TEXT, "user_id": TEXT},
        ["status_text"],
    ),
    # -- pins and saved items ----------------------------------------------
    _tool("pin_message", "Pin a message for everyone in the conversation.", {"message_id": TEXT}, ["message_id"]),
    _tool("unpin_message", "Remove a pin from a message.", {"message_id": TEXT}, ["message_id"]),
    _tool("list_pins", "List the pinned messages in a channel or chat.", {"conversation_id": TEXT}, ["conversation_id"]),
    _tool("save_message", "Save a message to your own saved items; nobody else sees this.", {"message_id": TEXT}, ["message_id"]),
    _tool("unsave_message", "Remove a message from your saved items.", {"message_id": TEXT}, ["message_id"]),
    _tool("list_saved_items", "List the messages you have saved.", {}, []),
    # -- notifications -----------------------------------------------------
    _tool(
        "list_notifications",
        "Your notifications, newest first: mentions, direct messages, and "
        "replies in threads you follow. Pass only_unread to filter.",
        {"only_unread": BOOLEAN},
        [],
    ),
    _tool("mark_notification_read", "Mark one of your notifications read.", {"notification_id": TEXT}, ["notification_id"]),
    # -- search ------------------------------------------------------------
    _tool(
        "search_users",
        "Search people by name, handle, email, team or title.",
        {"query": TEXT},
        ["query"],
    ),
    _tool(
        "search_channels",
        "Search channels by name or topic. Private channels you are not in are not listed.",
        {"query": TEXT},
        ["query"],
    ),
)


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return independent dictionaries safe for callers to modify."""
    import copy

    return copy.deepcopy(list(TOOL_DEFINITIONS))
