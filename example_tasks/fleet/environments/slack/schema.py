"""Tool schema for agents operating the Slack service."""

from __future__ import annotations

from typing import Any

ToolSchema = dict[str, Any]

SLACK_TOOL_SCHEMA: list[ToolSchema] = [
    {
        "name": "list_channels",
        "description": "List all public channels and private channels the acting user is a member of.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_channel_messages",
        "description": "Return messages for a Slack channel by channel id.",
        "input_schema": {
            "type": "object",
            "properties": {"channel_id": {"type": "string", "description": "Slack channel id, for example C003."}},
            "required": ["channel_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "post_message",
        "description": "Post a message to a channel by channel_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Channel id to post to, for example C001."},
                "body": {"type": "string", "description": "Body of the message."},
            },
            "required": ["channel_id", "body"],
            "additionalProperties": False,
        },
    },
    {
        "name": "update_message",
        "description": "Update/edit an existing message body. Only the author of the message can edit it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message_id": {"type": "string", "description": "Message id to update."},
                "body": {"type": "string", "description": "New body of the message."},
            },
            "required": ["message_id", "body"],
            "additionalProperties": False,
        },
    },
    {
        "name": "add_reaction",
        "description": "Add an emoji reaction to a message.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message_id": {"type": "string", "description": "Message id to react to."},
                "emoji": {"type": "string", "description": "Emoji to react with, for example eyes."},
            },
            "required": ["message_id", "emoji"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_channel",
        "description": "Create a new public or private channel.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the new channel."},
                "is_private": {"type": "boolean", "description": "Whether the channel is private. Defaults to false."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_messages",
        "description": "Search visible Slack messages by terms, author handle, and channel name.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query, for example '@alice incidents'."}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_group",
        "description": "Create a new group chat with a name and participants.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the group chat."},
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of user ids to add to the group chat.",
                },
            },
            "required": ["name", "participants"],
            "additionalProperties": False,
        },
    },
    {
        "name": "change_channel_name",
        "description": "Rename a channel. Only the channel owner can rename it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Channel id to rename."},
                "new_name": {"type": "string", "description": "New name of the channel."},
            },
            "required": ["channel_id", "new_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "change_group_name",
        "description": "Rename a group chat. Any participant in the group can rename it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "group_id": {"type": "string", "description": "Group chat id (or chat_id) to rename."},
                "new_name": {"type": "string", "description": "New name of the group chat."},
            },
            "required": ["group_id", "new_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_dm_message",
        "description": "Create a one-on-one DM chat with another user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient_id": {"type": "string", "description": "User id of the DM recipient."},
            },
            "required": ["recipient_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "send_dm_message",
        "description": "Send a message in a DM chat. Finds/creates the DM chat if recipient_id is provided, or posts directly if chat_id is provided.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient_id": {"type": "string", "description": "User id of the DM recipient."},
                "chat_id": {"type": "string", "description": "Chat id of the DM chat."},
                "body": {"type": "string", "description": "Body of the message."},
            },
            "required": ["body"],
            "additionalProperties": False,
        },
    },
    {
        "name": "send_group_message",
        "description": "Send a message in a group chat by group_id/chat_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "group_id": {"type": "string", "description": "Group chat id to post to."},
                "body": {"type": "string", "description": "Body of the message."},
            },
            "required": ["group_id", "body"],
            "additionalProperties": False,
        },
    },
    {
        "name": "change_user_display_name",
        "description": "Change a user's display name. Only the user themselves or an admin can perform this.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User id to change the display name of."},
                "new_display_name": {"type": "string", "description": "New display name."},
            },
            "required": ["user_id", "new_display_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "slack.reply_to_thread",
        "description": "Reply to an existing message thread by parent message id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thread_parent_id": {"type": "string", "description": "Message id of the parent message to reply to."},
                "body": {"type": "string", "description": "Body of the thread reply message."},
            },
            "required": ["thread_parent_id", "body"],
            "additionalProperties": False,
        },
    },
]


def available_tools() -> list[ToolSchema]:
    return SLACK_TOOL_SCHEMA
