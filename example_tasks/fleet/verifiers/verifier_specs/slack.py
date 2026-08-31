"""Slack verifier specifications built on the shared layered verifier."""

from __future__ import annotations

from typing import Any


def slack_seed_shape():
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        state = trajectory.get("extra", {}).get("initial_state_snapshot", {})
        details = {
            "users": len(state.get("users", [])),
            "channels": len(state.get("channels", [])),
            "messages": len(state.get("messages", [])),
        }
        passed = details == {"users": 6, "channels": 7, "messages": 26}
        return passed, "Expected seeded Slack workspace shape.", details

    return check


def agent_used_atif():
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        schema_version = trajectory.get("schema_version")
        first_source = trajectory.get("steps", [{}])[0].get("source")
        passed = schema_version == "ATIF-v1.7" and first_source == "user"
        return passed, "Expected ATIF-v1.7 trajectory with user first step.", {
            "schema_version": schema_version,
            "first_source": first_source,
        }

    return check


def rl_determinism_check():
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        reset_check = trajectory.get("extra", {}).get("reset_determinism_check", {})
        passed = bool(reset_check.get("passed", False))
        return passed, "Expected RL environment to be deterministic on reset.", reset_check

    return check


def check_most_recent_user_message_references_channel(channel_id: str, author_id: str, referenced_channel_id: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        messages = [
            message
            for message in state.get("messages", [])
            if message.get("channel_id") == channel_id
            and message.get("author_id") == author_id
            and not message.get("deleted", False)
        ]
        if not messages:
            return False, f"Expected messages by user {author_id} in channel {channel_id}.", {
                "channel_id": channel_id,
                "author_id": author_id,
            }
        most_recent = max(messages, key=lambda message: int(message.get("created_at_ms", 0)))
        referenced_channel = next(
            (channel for channel in state.get("channels", []) if channel.get("channel_id") == referenced_channel_id),
            None,
        )
        if referenced_channel is None:
            return False, f"Expected referenced channel {referenced_channel_id} to exist.", {
                "referenced_channel_id": referenced_channel_id,
            }
        expected_reference = f"#{referenced_channel['name']}"
        passed = expected_reference in str(most_recent.get("body", ""))
        return passed, "Expected most recent message by stable user ID to reference stable channel ID.", {
            "message_id": most_recent.get("message_id"),
            "channel_id": channel_id,
            "author_id": author_id,
            "referenced_channel_id": referenced_channel_id,
            "expected_reference": expected_reference,
            "body": most_recent.get("body"),
        }

    return check


def check_final_answer_is_channel_name(channel_id: str):
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        final_state = trajectory.get("extra", {}).get("final_state_snapshot", {})
        channel = next(
            (item for item in final_state.get("channels", []) if item.get("channel_id") == channel_id),
            None,
        )
        if channel is None:
            return False, f"Expected channel {channel_id} to exist in final state.", {"channel_id": channel_id}
        expected = f"#{channel['name']}"
        actual = str(trajectory.get("extra", {}).get("final_answer", ""))
        if not actual:
            agent_steps = [step for step in trajectory.get("steps", []) if step.get("source") == "agent"]
            actual = str(agent_steps[-1].get("message", "")) if agent_steps else ""
        cleaned = actual.strip().strip("'\"`").strip()
        return cleaned == expected, "Expected final answer to match channel identified by stable channel ID.", {
            "channel_id": channel_id,
            "expected": expected,
            "actual": actual,
        }

    return check


def check_message_posted(channel_id: str, body: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        messages = state.get("messages", [])
        found = any(m["channel_id"] == channel_id and m["body"] == body and not m.get("deleted", False) for m in messages)
        return found, f"Expected message '{body}' in channel {channel_id}.", {"channel_id": channel_id}
    return check


def check_thread_reply(channel_id: str, parent_msg_id: str, reply_body: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        messages = state.get("messages", [])
        found = any(
            m["channel_id"] == channel_id 
            and m["thread_parent_id"] == parent_msg_id 
            and m["body"] == reply_body 
            and not m.get("deleted", False) 
            for m in messages
        )
        return found, f"Expected threaded reply '{reply_body}' to message {parent_msg_id} in {channel_id}.", {}
    return check


def check_parent_message_unchanged(message_id: str, expected_body: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        messages = state.get("messages", [])
        parent = next((m for m in messages if m["message_id"] == message_id), None)
        if not parent:
            return False, f"Expected parent message {message_id} to exist.", {}
        passed = parent["body"] == expected_body and not parent.get("deleted", False)
        return passed, f"Expected parent message {message_id} to remain unchanged.", {
            "expected_body": expected_body,
            "actual_body": parent.get("body"),
        }
    return check


def check_thread_reaction(channel_id: str, parent_msg_id: str, reply_body: str, emoji: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        messages = state.get("messages", [])
        reply_msg = next((
            m for m in messages 
            if m["channel_id"] == channel_id 
            and m["thread_parent_id"] == parent_msg_id 
            and m["body"] == reply_body 
            and not m.get("deleted", False)
        ), None)
        if not reply_msg:
            return False, f"Thread reply '{reply_body}' not found to check reaction.", {}
        
        reactions = state.get("reactions", [])
        emoji_variants = {emoji, f":{emoji}:", emoji.strip(":")}
        found = any(
            r["message_id"] == reply_msg["message_id"] 
            and r["emoji"] in emoji_variants 
            for r in reactions
        )
        return found, f"Expected reaction '{emoji}' on threaded reply message.", {"message_id": reply_msg["message_id"]}
    return check


def _seed_chat_ids() -> frozenset[str]:
    from fleet.environments.slack.seed import SLACK_CHATS

    return frozenset(chat_id for chat_id, *_ in SLACK_CHATS)


def _find_created_group(chats: list[dict[str, Any]], seed_chat_ids: frozenset[str]) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    """Locate the group the agent created by its unique chat_id (any group chat
    not present in the deterministic seed). Names are not used: they change on
    rename, and name correctness is judged separately by check_group_renamed."""
    created = [c for c in chats if c["type"] == "group" and c["chat_id"] not in seed_chat_ids]
    if not created:
        return None, "No group chat was created beyond the seeded workspace.", {"seed_chat_ids": sorted(seed_chat_ids)}
    if len(created) > 1:
        return None, "Expected exactly one created group chat.", {
            "created_groups": [{"chat_id": c["chat_id"], "name": c["name"]} for c in created],
        }
    return created[0], "", {}


def check_group_created(participants: list[str]):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        group, error, details = _find_created_group(state.get("chats", []), _seed_chat_ids())
        if group is None:
            return False, error, details
        p_ids = {p["user_id"] for p in group.get("participants", [])}
        expected_participants = set(participants)
        passed = p_ids == expected_participants
        return passed, f"Expected created group chat '{group['chat_id']}' with exactly participants {participants}.", {
            "group_chat_id": group["chat_id"],
            "group_name": group["name"],
            "expected_participants": sorted(expected_participants),
            "actual_participants": sorted(p_ids),
        }
    return check


def check_group_renamed(initial_name: str, final_name: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        group, error, details = _find_created_group(state.get("chats", []), _seed_chat_ids())
        if group is None:
            return False, error, details
        passed = group.get("name") == final_name
        return passed, f"Expected group '{initial_name}' to be renamed to '{final_name}'.", {
            "group_chat_id": group.get("chat_id"),
            "expected_name": final_name,
            "actual_name": group.get("name"),
        }
    return check


def check_group_message(body: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        group, error, details = _find_created_group(state.get("chats", []), _seed_chat_ids())
        if group is None:
            return False, error, details
        messages = state.get("messages", [])
        found = any(m["channel_id"] == group["chat_id"] and m["body"] == body and not m.get("deleted", False) for m in messages)
        return found, f"Expected message '{body}' in created group chat '{group['chat_id']}'.", {
            "group_chat_id": group["chat_id"],
            "group_name": group["name"],
        }
    return check


def check_final_answer_is_group_id():
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        extra = trajectory.get("extra", {})
        final_state = extra.get("final_state_snapshot", {})
        initial_state = extra.get("initial_state_snapshot", {})
        initial_chat_ids = frozenset(c["chat_id"] for c in initial_state.get("chats", [])) or _seed_chat_ids()
        group, error, details = _find_created_group(final_state.get("chats", []), initial_chat_ids)
        if group is None:
            return False, f"{error} Cannot resolve the created group ID for the final answer.", details
        expected_id = group["chat_id"]
        
        actual_answer = str(trajectory.get("extra", {}).get("final_answer", "")).strip()
        if not actual_answer:
            agent_steps = [step for step in trajectory.get("steps", []) if step.get("source") == "agent"]
            actual_answer = str(agent_steps[-1].get("message", "")).strip() if agent_steps else ""
            
        cleaned_answer = actual_answer.strip().strip("'\"`").strip()
        passed = cleaned_answer == expected_id
        return passed, f"Expected final answer to be the group ID '{expected_id}'.", {"expected": expected_id, "actual": actual_answer}
    return check


def check_message_in_chat(chat_id: str, body: str, author_id: str):
    """A message with this exact body and author exists in the given chat/DM."""
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        matches = [
            m for m in state.get("messages", [])
            if m.get("channel_id") == chat_id
            and m.get("body") == body
            and m.get("author_id") == author_id
            and not m.get("deleted", False)
        ]
        return bool(matches), f"Expected message '{body}' by {author_id} in chat {chat_id}.", {
            "chat_id": chat_id,
            "matches": len(matches),
        }
    return check


def check_chat_message_count(chat_id: str, expected_count: int):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        count = sum(
            1 for m in state.get("messages", [])
            if m.get("channel_id") == chat_id and not m.get("deleted", False)
        )
        return count == expected_count, f"Expected chat {chat_id} to have exactly {expected_count} messages.", {
            "actual_count": count
        }
    return check


def check_message_edited(message_id: str, expected_body: str):
    """The message was edited to the exact expected body."""
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        message = next((m for m in state.get("messages", []) if m.get("message_id") == message_id), None)
        if message is None:
            return False, f"Expected message {message_id} to exist.", {}
        passed = message.get("body") == expected_body and message.get("edited_at_ms") is not None
        return passed, f"Expected message {message_id} to be edited to '{expected_body}'.", {
            "actual_body": message.get("body"),
            "edited_at_ms": message.get("edited_at_ms"),
        }
    return check


def check_body_prefix_only_in_threads(body_prefix: str, allowed: dict[str, str]):
    """Every non-seed message starting with `body_prefix` must be a thread reply
    under one of the allowed parents with that parent's exact expected body.
    Placement police for replies whose content is parent-specific."""
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        offenders = []
        for m in state.get("messages", []):
            if m.get("deleted", False) or not str(m.get("body", "")).startswith(body_prefix):
                continue
            parent = m.get("thread_parent_id")
            if parent not in allowed or m.get("body") != allowed[parent]:
                offenders.append({"message_id": m.get("message_id"), "parent": parent, "body": m.get("body")})
        return not offenders, f"Expected '{body_prefix}…' replies only under {sorted(allowed)} with exact bodies.", {
            "offenders": offenders
        }
    return check


def _snapshot_message(trajectory: dict[str, Any], snapshot_key: str, message_id: str) -> dict[str, Any] | None:
    snapshot = trajectory.get("extra", {}).get(snapshot_key, {})
    return next(
        (m for m in snapshot.get("messages", []) if m.get("message_id") == message_id),
        None,
    )


def check_message_edit_timestamp_changed(message_id: str):
    """The message's edited_at_ms in the final snapshot must differ from its
    initial-snapshot value — proof the edit happened during this run. None to
    timestamp counts as changed, and so does one virtual-clock value to a
    later one; the comparison is purely against the original value."""
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        initial = _snapshot_message(trajectory, "initial_state_snapshot", message_id)
        final = _snapshot_message(trajectory, "final_state_snapshot", message_id)
        if initial is None or final is None:
            return False, f"Expected message {message_id} in both snapshots.", {}
        passed = final.get("edited_at_ms") != initial.get("edited_at_ms")
        return passed, f"Expected edit timestamp of {message_id} to change from its initial value.", {
            "initial_edited_at_ms": initial.get("edited_at_ms"),
            "final_edited_at_ms": final.get("edited_at_ms"),
        }
    return check


def check_messages_unmodified_since_initial(message_ids: list[str]):
    """Body, created_at_ms, and edited_at_ms must all equal the initial
    snapshot — catches edits that restore the body but leave timestamps."""
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        offenders = []
        for message_id in message_ids:
            initial = _snapshot_message(trajectory, "initial_state_snapshot", message_id)
            final = _snapshot_message(trajectory, "final_state_snapshot", message_id)
            if initial is None or final is None:
                offenders.append({"message_id": message_id, "reason": "missing from snapshot"})
                continue
            for field in ("body", "created_at_ms", "edited_at_ms"):
                if initial.get(field) != final.get(field):
                    offenders.append({"message_id": message_id, "field": field,
                                      "initial": initial.get(field), "final": final.get(field)})
        return not offenders, f"Expected messages {message_ids} to be unmodified since the initial snapshot.", {
            "offenders": offenders
        }
    return check


def check_final_answer_equals(expected: str):
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        actual = str(trajectory.get("extra", {}).get("final_answer", "")).strip()
        if not actual:
            agent_steps = [step for step in trajectory.get("steps", []) if step.get("source") == "agent"]
            actual = str(agent_steps[-1].get("message", "")).strip() if agent_steps else ""
        cleaned = actual.strip().strip("'\"`").strip()
        return cleaned == expected, f"Expected final answer to be exactly '{expected}'.", {"actual": actual}
    return check


def check_no_new_messages_in_channels(channel_ids: list[str], allowed_count: int):
    """Decoy channels must not gain messages beyond their seeded count."""
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        counts = {
            channel_id: sum(
                1
                for message in state.get("messages", [])
                if message.get("channel_id") == channel_id and not message.get("deleted", False)
            )
            for channel_id in channel_ids
        }
        passed = all(count <= allowed_count for count in counts.values())
        return passed, f"Expected decoy channels {channel_ids} to gain no new messages.", {"counts": counts}
    return check


def check_no_thread_replies(parent_id: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        replies = [
            message
            for message in state.get("messages", [])
            if message.get("thread_parent_id") == parent_id and not message.get("deleted", False)
        ]
        return not replies, f"Expected no thread replies under decoy message {parent_id}.", {"replies": len(replies)}
    return check


def check_group_message_count(group_name: str, expected_count: int):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        group = next(
            (c for c in state.get("chats", []) if c.get("type") == "group" and c.get("name") == group_name),
            None,
        )
        if group is None:
            return False, f"Expected group chat '{group_name}' to exist.", {}
        group_id = group.get("chat_id") or group.get("id")
        count = sum(
            1
            for message in state.get("messages", [])
            if message.get("channel_id") == group_id and not message.get("deleted", False)
        )
        return count == expected_count, f"Expected group '{group_name}' to have exactly {expected_count} messages.", {
            "actual_count": count
        }
    return check


def check_call_order(first_tool: str, first_args: dict[str, Any], second_tool: str, second_args: dict[str, Any]):
    """The first matching call of `first_tool` must precede the first matching
    call of `second_tool` in the trajectory."""
    def matching_indexes(trajectory: dict[str, Any], tool_name: str, expected_args: dict[str, Any]) -> list[int]:
        from fleet.verifiers import trajectory_tool_calls

        indexes = []
        for index, call in enumerate(trajectory_tool_calls(trajectory)):
            name = call.get("tool_name", "")
            if name.startswith("slack."):
                name = name[6:]
            target = tool_name[6:] if tool_name.startswith("slack.") else tool_name
            if name != target:
                continue
            payload = call.get("input_payload", {})
            if all(str(payload.get(k)).strip() == str(v).strip() for k, v in expected_args.items()):
                indexes.append(index)
        return indexes

    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        first_indexes = matching_indexes(trajectory, first_tool, first_args)
        second_indexes = matching_indexes(trajectory, second_tool, second_args)
        if not first_indexes or not second_indexes:
            return False, f"Expected both {first_tool} and {second_tool} to be called.", {
                "first_indexes": first_indexes,
                "second_indexes": second_indexes,
            }
        passed = min(first_indexes) < min(second_indexes)
        return passed, f"Expected {first_tool} to be called before {second_tool}.", {
            "first_indexes": first_indexes,
            "second_indexes": second_indexes,
        }
    return check


def check_final_answer_is_group_id_and_channel(referenced_channel_name: str):
    """Final answer must be '<created group id>:<referenced channel>', with the
    group id resolved from the created (non-seed) group in the final state."""
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        extra = trajectory.get("extra", {})
        final_state = extra.get("final_state_snapshot", {})
        initial_state = extra.get("initial_state_snapshot", {})
        initial_chat_ids = frozenset(c["chat_id"] for c in initial_state.get("chats", [])) or _seed_chat_ids()
        group, error, details = _find_created_group(final_state.get("chats", []), initial_chat_ids)
        if group is None:
            return False, f"{error} Cannot resolve the created group ID for the final answer.", details
        expected = f"{group['chat_id']}:{referenced_channel_name}"

        actual = str(extra.get("final_answer", "")).strip()
        if not actual:
            agent_steps = [step for step in trajectory.get("steps", []) if step.get("source") == "agent"]
            actual = str(agent_steps[-1].get("message", "")).strip() if agent_steps else ""
        cleaned = actual.strip().strip("'\"`").strip()
        return cleaned == expected, "Expected final answer formatted as <group_id>:<referenced channel>.", {
            "expected": expected,
            "actual": actual,
        }
    return check


def check_tool_called_with_args(tool_name: str, **expected_args: Any):
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        from fleet.verifiers import trajectory_tool_calls
        calls = []
        for call in trajectory_tool_calls(trajectory):
            name = call.get("tool_name", "")
            if name.startswith("slack."):
                name = name[6:]
            norm_target = tool_name
            if norm_target.startswith("slack."):
                norm_target = norm_target[6:]
            if name == norm_target:
                calls.append(call)
        
        for call in calls:
            payload = call.get("input_payload", {})
            match = True
            for k, val in expected_args.items():
                actual_val = payload.get(k)
                if k == "participants" and isinstance(actual_val, list) and isinstance(val, list):
                    if sorted(actual_val) != sorted(val):
                        match = False
                        break
                else:
                    if str(actual_val).strip().lower() != str(val).strip().lower():
                        match = False
                        break
            if match:
                return True, f"Found call to {tool_name} with expected arguments.", {"tool_name": tool_name, "call": call}
                
        return False, f"Expected call to {tool_name} with arguments {expected_args} was not found.", {
            "tool_name": tool_name,
            "expected_arguments": expected_args,
            "calls_found": [c.get("input_payload") for c in calls]
        }
    return check
