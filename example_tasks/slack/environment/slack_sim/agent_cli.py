"""The agent's entire interface to the Slack workspace.

Every verb here is one of the model-facing Slack tools. There is no database flag, no
state dump, and no lifecycle command, because this module never imports them:
requests travel over ``agent.sock`` to the environment's own container, and the
server's agent surface has no dispatch entry for anything else.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from slack_sim.protocol import AGENT_SOCKET, request
from slack_sim.tool_definitions import get_tool_definitions

# argparse flag -> tool input key. Kept explicit so the agent's surface and the
# tool schemas cannot drift apart silently; a test asserts they agree.
_FLAGS: dict[str, tuple[str, str]] = {
    "--query": ("query", "Search query."),
    "--channel-id": ("channel_id", "Channel or chat ID."),
    "--conversation-id": ("conversation_id", "Conversation ID."),
    "--ts": ("ts", "Slack timestamp."),
    "--thread-ts": ("thread_ts", "Thread root Slack timestamp."),
    "--cursor": ("cursor", "Opaque cursor returned by the previous page."),
    "--message-id": ("message_id", "Message ID."),
    "--thread-parent-id": ("thread_parent_id", "Thread parent message ID."),
    "--body": ("body", "Message body."),
    "--emoji": ("emoji", "Emoji name."),
    "--name": ("name", "Name."),
    "--new-name": ("new_name", "New name."),
    "--recipient-id": ("recipient_id", "DM recipient user ID."),
    "--chat-id": ("chat_id", "Chat ID."),
    "--group-id": ("group_id", "Group chat ID."),
    "--user-id": ("user_id", "User ID."),
    "--user-group-id": ("user_group_id", "User group ID."),
    "--new-display-name": ("new_display_name", "New display name."),
}
_LIST_FLAGS: dict[str, str] = {"--user-ids": "user_ids", "--participants": "participants"}
_INTEGER_FLAGS: dict[str, str] = {"--limit": "limit"}


def _build_parser() -> argparse.ArgumentParser:
    tool_names = sorted(tool["name"] for tool in get_tool_definitions())
    parser = argparse.ArgumentParser(
        prog="slack",
        description="Slack workspace tools. The workspace runs in the environment; "
        "this command is a client and holds no state of its own.",
    )
    parser.add_argument("tool", choices=[*tool_names, "list_tools"], help="Tool to invoke.")
    for flag, (_, helptext) in _FLAGS.items():
        parser.add_argument(flag, default=None, help=helptext)
    for flag in _LIST_FLAGS:
        parser.add_argument(flag, default=None, help="Comma-separated IDs.")
    for flag in _INTEGER_FLAGS:
        parser.add_argument(
            flag,
            default=None,
            type=int,
            help="Page size (default 50; channel history is capped at 50, other collections at 100).",
        )
    parser.add_argument("--is-private", action="store_true", help="Create the channel private.")
    parser.add_argument(
        "--input-payload",
        default=None,
        help="Raw JSON tool input. Overrides the individual flags when given.",
    )
    parser.add_argument("--socket", default=AGENT_SOCKET, help=argparse.SUPPRESS)
    return parser


def _payload_from(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_payload:
        payload = json.loads(args.input_payload)
        if not isinstance(payload, dict):
            raise SystemExit("--input-payload must be a JSON object.")
        return payload
    payload: dict[str, Any] = {}
    for flag, (key, _) in _FLAGS.items():
        value = getattr(args, flag.lstrip("-").replace("-", "_"))
        if value is not None:
            payload[key] = value
    for flag, key in _LIST_FLAGS.items():
        value = getattr(args, flag.lstrip("-").replace("-", "_"))
        if value is not None:
            payload[key] = [item.strip() for item in value.split(",") if item.strip()]
    for flag, key in _INTEGER_FLAGS.items():
        value = getattr(args, flag.lstrip("-").replace("-", "_"))
        if value is not None:
            payload[key] = value
    if args.is_private:
        payload["is_private"] = True
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    body: dict[str, Any] = (
        {"op": "list_tools"}
        if args.tool == "list_tools"
        else {"op": "call_tool", "tool": args.tool, "input": _payload_from(args)}
    )
    try:
        response = request(args.socket, body)
    except OSError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "workspace_unavailable",
                        "type": "unavailable",
                        "message": f"Cannot reach the Slack workspace: {exc}",
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
