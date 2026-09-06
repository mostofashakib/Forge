"""The agent's entire interface to the task tracker.

Every verb here is one of the model-facing tools. There is no database flag, no
state dump and no lifecycle command, because this module never imports them:
requests travel over ``agent.sock`` to the environment's own container, and the
server's agent surface has no dispatch entry for anything else.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from task_sim.protocol import AGENT_SOCKET, request
from task_sim.tool_definitions import get_tool_definitions

# argparse flag -> tool input key. Kept explicit so the agent's surface and the
# tool schemas cannot drift apart silently; a test asserts they agree.
_FLAGS: dict[str, tuple[str, str]] = {
    "--task-id": ("task_id", "Task id."),
    "--original-task-id": ("original_task_id", "The canonical task a duplicate points at."),
    "--depends-on-task-id": ("depends_on_task_id", "The upstream (blocking) task."),
    "--project-id": ("project_id", "Project id."),
    "--milestone-id": ("milestone_id", "Milestone id."),
    "--title": ("title", "Title."),
    "--name": ("name", "Name."),
    "--description": ("description", "Description."),
    "--summary": ("summary", "One-line summary of what you did."),
    "--assignee": ("assignee", "Assignee user id."),
    "--status": ("status", "Task status."),
    "--priority": ("priority", "Task priority."),
}
_LIST_FLAGS: dict[str, str] = {"--labels": "labels", "--task-ids": "task_ids"}
_INTEGER_FLAGS: dict[str, str] = {"--due-at-ms": "due_at_ms"}
_BOOLEAN_FLAGS: dict[str, str] = {
    "--include-archived": "include_archived",
    "--include-deleted": "include_deleted",
}


def _attr(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def _build_parser() -> argparse.ArgumentParser:
    names = sorted(tool["name"] for tool in get_tool_definitions())
    parser = argparse.ArgumentParser(
        prog="tasks",
        description="Task-tracker tools. The workspace runs in the environment; "
        "this command is a client and holds no state of its own.",
    )
    parser.add_argument("tool", choices=[*names, "list_tools"], help="Tool to invoke.")
    for flag, (_, helptext) in _FLAGS.items():
        parser.add_argument(flag, default=None, help=helptext)
    for flag in _LIST_FLAGS:
        parser.add_argument(flag, default=None, help="Comma-separated values.")
    for flag in _INTEGER_FLAGS:
        parser.add_argument(flag, default=None, type=int, help="Unix timestamp in milliseconds.")
    for flag in _BOOLEAN_FLAGS:
        parser.add_argument(flag, action="store_true", help="Include these records.")
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
        value = getattr(args, _attr(flag))
        if value is not None:
            payload[key] = value
    for flag, key in _LIST_FLAGS.items():
        value = getattr(args, _attr(flag))
        if value is not None:
            payload[key] = [item.strip() for item in value.split(",") if item.strip()]
    for flag, key in _INTEGER_FLAGS.items():
        value = getattr(args, _attr(flag))
        if value is not None:
            payload[key] = value
    for flag, key in _BOOLEAN_FLAGS.items():
        if getattr(args, _attr(flag)):
            payload[key] = True
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
                        "message": f"Cannot reach the task tracker: {exc}",
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
