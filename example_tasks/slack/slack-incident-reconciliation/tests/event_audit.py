#!/usr/bin/env python3
"""Investigate why a latent event did or did not fire, by hand.

Two ways in:

    # Audit one phrasing against one event.
    python3 tests/event_audit.py --event timing_confirmed \
        --body "Is 9:30 still the plan?"

    # Replay whatever a real agent did, from a Harbor trajectory.
    python3 tests/event_audit.py --event timing_confirmed \
        --trajectory ../jobs/harbor/<job>/<trial>/agent/trajectory.json

Each run reports the state before, the action, every predicate the rule weighed
with its candidates and what matched, whether the event moved, and what became
observable afterwards -- so a failed episode can be attributed to a cause
rather than to "the workspace never changed".

Run from the task root with `PYTHONPATH=environment:tests`.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any

from event_diagnostics import BEN, Action, Workspace, audit

#: Everything the auditor needs to know about one event to check it end to end:
#: the rule that gates it, the message it releases, where that message lands,
#: and a query that should find it once it exists.
EVENTS: dict[str, dict[str, Any]] = {
    "timing_confirmed": {
        "rule_id": "r040_timing_confirmed",
        "message_id": "LAT021",
        "channel": "C019",
        "thread_parent": "CUT001",
        "search_terms": "Confirmed directly with Acme",
        "notification_id": "NTF-LAT008",
        "canonical_body": "Nina, can you confirm whether the 9:30 request is real?",
        "canonical_thread": "CUT001",
    },
    "saml_revision_2": {
        "rule_id": "r010_saml_critique",
        "message_id": "LAT001",
        "channel": "C023",
        "thread_parent": "MSG194",
        "search_terms": "metadata loader supplies a canonical Audience",
        "notification_id": "NTF-LAT001",
        "canonical_body": (
            "Critique: `includes` performs substring matching, but the canonical SAML "
            "Audience must match by exact equality."
        ),
        "canonical_thread": "MSG194",
    },
    "saml_deployed": {
        "rule_id": "r011_saml_approved",
        "message_id": "LAT002",
        "channel": "C023",
        "thread_parent": "MSG194",
        "search_terms": "CI passed for the revised",
        # NTF-LAT002 announces LAT005 in #identity-eng, not this message.
        "notification_id": None,
        "canonical_body": "Looks good to me",
        "canonical_thread": "MSG194",
        "requires": ("saml_revision_2",),
    },
    "export_backfill_results": {
        "rule_id": "r030_export_approved",
        "message_id": "LAT006",
        "channel": "C021",
        "thread_parent": None,
        "search_terms": "corrective backfill finished",
        # NTF-LAT003 announces LAT007, the follow-up in the same channel.
        "notification_id": None,
        "canonical_body": "Looks good to me",
        "canonical_thread": "MSG200",
    },
    "permissions_revision_2": {
        "rule_id": "r020_permissions_critique",
        "message_id": "LAT012",
        "channel": "C023",
        "thread_parent": "MSG214",
        "search_terms": "Revised after Ben's review",
        "notification_id": "NTF-LAT005",
        "canonical_body": (
            "Critique: `some` implements any-match semantics, but the policy requires "
            "every required entitlement."
        ),
        "canonical_thread": "MSG214",
    },
    "permissions_checker_rerun": {
        "rule_id": "r021_permissions_approved",
        "message_id": "LAT013",
        "channel": "C022",
        "thread_parent": None,
        "search_terms": "Corrected checker rerun",
        "notification_id": None,
        "canonical_body": "Looks good to me",
        "canonical_thread": "MSG214",
        "requires": ("permissions_revision_2",),
    },
    "rollback_initially_verified": {
        "rule_id": "r050_rollback_checked",
        "message_id": "LAT024",
        "channel": "C024",
        "thread_parent": "LAT022",
        "search_terms": "Rollback worker owner check",
        "notification_id": None,
        "canonical_body": (
            "Sam, please verify the rollback worker by running the recovery invocation "
            "and report its status."
        ),
        "canonical_thread": "LAT022",
        "requires": ("timing_confirmed",),
    },
}


# ---------------------------------------------------------------------------
# Reading actions out of a Harbor trajectory
# ---------------------------------------------------------------------------

_CLI_FLAG_TO_KEY = {
    "--query": "query", "--channel-id": "channel_id", "--conversation-id": "conversation_id",
    "--ts": "ts", "--thread-ts": "thread_ts", "--cursor": "cursor", "--message-id": "message_id",
    "--thread-parent-id": "thread_parent_id", "--body": "body", "--emoji": "emoji",
    "--name": "name", "--new-name": "new_name", "--recipient-id": "recipient_id",
    "--chat-id": "chat_id", "--group-id": "group_id", "--user-id": "user_id",
    "--user-group-id": "user_group_id", "--new-display-name": "new_display_name",
    "--limit": "limit",
}


def _actions_from_command(command: str) -> list[Action]:
    """Pull `slack <tool> --flag value` invocations out of a shell command."""
    found: list[Action] = []
    for fragment in re.split(r"[;&|\n]+", command):
        fragment = fragment.strip()
        if not re.match(r"^(sudo\s+)?slack\s+[a-z_]+", fragment):
            continue
        try:
            parts = shlex.split(fragment)
        except ValueError:
            continue
        parts = parts[1:] if parts[0] == "sudo" else parts
        if len(parts) < 2:
            continue
        tool = parts[1]
        payload: dict[str, Any] = {}
        index = 2
        while index < len(parts):
            flag = parts[index]
            key = _CLI_FLAG_TO_KEY.get(flag)
            if key and index + 1 < len(parts):
                payload[key] = parts[index + 1]
                index += 2
                continue
            if flag == "--input-payload" and index + 1 < len(parts):
                try:
                    payload.update(json.loads(parts[index + 1]))
                except json.JSONDecodeError:
                    pass
                index += 2
                continue
            index += 1
        found.append(Action(tool, payload, BEN, label="cli"))
    return found


def extract_slack_actions(trajectory: Path) -> list[Action]:
    """Every Slack action a trajectory contains, MCP calls and CLI alike.

    Harbor records MCP tool calls by name (`mcp__slack__reply_to_thread`) and
    shell work as a Bash command, so both spellings are read here. An agent
    that reached the workspace some third way would show up as nothing, which
    is itself a finding worth printing rather than guessing around.
    """
    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    found: list[Action] = []
    for step in payload.get("steps", []):
        for call in step.get("tool_calls") or []:
            name = call.get("name") or ""
            arguments = call.get("arguments") or {}
            if name.startswith("mcp__slack__"):
                found.append(Action(name.split("mcp__slack__", 1)[1], dict(arguments), BEN, "mcp"))
                continue
            if name in {"slack", "Slack"} and isinstance(arguments, dict):
                tool = str(arguments.get("tool") or arguments.get("name") or "")
                if tool:
                    found.append(Action(tool, dict(arguments.get("input") or {}), BEN, "mcp"))
                continue
            command = arguments.get("command") if isinstance(arguments, dict) else None
            if isinstance(command, str):
                found.extend(_actions_from_command(command))
    return found


def relevant_to(actions: list[Action], event_id: str) -> list[Action]:
    """Actions that could plausibly bear on this event, by thread or subject."""
    spec = EVENTS[event_id]
    thread = spec.get("canonical_thread")
    keywords = {word for group in _keyword_candidates(event_id) for word in group}
    chosen: list[Action] = []
    for action in actions:
        body = str(action.payload.get("body", "")).lower()
        if action.payload.get("thread_parent_id") == thread:
            chosen.append(action)
        elif body and any(word in body for word in keywords):
            chosen.append(action)
    return chosen


def _keyword_candidates(event_id: str) -> list[list[str]]:
    from slack_sim.seed import SCENARIO_RULES

    rule_id = EVENTS[event_id]["rule_id"]
    for rule in SCENARIO_RULES:
        if rule.rule_id == rule_id:
            return [list(group) for group in rule.keyword_groups]
    return []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _prepare(workspace: Workspace, event_id: str) -> None:
    """Drive the prerequisites of `event_id` so it is the only thing under test."""
    spec = EVENTS[event_id]
    for required in spec.get("requires", ()):  # type: ignore[union-attr]
        needed = EVENTS[required]
        workspace.reply(str(needed["canonical_thread"]), str(needed["canonical_body"]))
        if workspace.event_status(required) != "activated":
            raise SystemExit(f"could not satisfy prerequisite {required} for {event_id}")
    if event_id == "rollback_initially_verified":
        workspace.join("C024")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="timing_confirmed", choices=sorted(EVENTS))
    parser.add_argument("--body", help="Audit this exact reply body.")
    parser.add_argument("--thread", help="Reply in this thread instead of the canonical one.")
    parser.add_argument("--dm", help="Audit a DM to this user id instead of a thread reply.")
    parser.add_argument("--trajectory", type=Path, help="Replay Slack actions from this file.")
    parser.add_argument("--all-actions", action="store_true",
                        help="With --trajectory, replay every Slack action, not just relevant ones.")
    args = parser.parse_args(argv)

    spec = EVENTS[args.event]
    probe = {
        "message_id": str(spec["message_id"]),
        "channel": str(spec["channel"]),
        "thread_parent": spec["thread_parent"],
        "search_terms": str(spec["search_terms"]),
        "notification_id": spec["notification_id"],
    }

    if args.trajectory:
        actions = extract_slack_actions(args.trajectory)
        print(f"{len(actions)} Slack action(s) found in {args.trajectory}\n")
        if not actions:
            print("The agent issued no Slack tool call at all in this trajectory.")
            print("Nothing can be attributed to the trigger: the workspace was never touched.")
            return 0
        candidates = actions if args.all_actions else relevant_to(actions, args.event)
        if not candidates:
            print(f"None of them plausibly targets {args.event}.")
            print("Re-run with --all-actions to replay everything.")
            return 0
        for action in candidates:
            workspace = Workspace()
            _prepare(workspace, args.event)
            audit(workspace, args.event, str(spec["rule_id"]), action, **probe)
        return 0

    if args.dm:
        action = Action("send_dm_message", {"recipient_id": args.dm, "body": args.body or ""})
    else:
        action = Action(
            "reply_to_thread",
            {
                "thread_parent_id": args.thread or str(spec["canonical_thread"]),
                "body": args.body or str(spec["canonical_body"]),
            },
        )
    workspace = Workspace()
    _prepare(workspace, args.event)
    audit(workspace, args.event, str(spec["rule_id"]), action, **probe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
