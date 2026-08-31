"""Root-test-only Harbor verifier checks.

These intentionally do not import task-local ``slack_task_*/tests/check.py`` so
the repository tests verify Harbor plumbing independently from task verifier
implementations. Generic checks come from the shared verifier library
(:mod:`fleet.verifiers.verifier_specs.slack`); only checks with no shared
equivalent are defined here.
"""

from __future__ import annotations

from typing import Any

from fleet.verifiers import (
    LayeredVerifier,
    VerificationSpec,
    forbidden_tool_not_called,
    trajectory_tool_calls,
)
from fleet.verifiers.verifier_specs.slack import (
    agent_used_atif,
    check_body_prefix_only_in_threads,
    check_call_order,
    check_chat_message_count,
    check_final_answer_equals,
    check_message_edit_timestamp_changed,
    check_message_edited,
    check_message_posted,
    check_messages_unmodified_since_initial,
    check_no_new_messages_in_channels,
    check_parent_message_unchanged,
    check_tool_called_with_args,
    rl_determinism_check,
)


SLACK_TASK_1_SPEC = {
    "dm_chat_id": "D001",
    "group_chat_id": "G001",
    "group_name": "Incident Response Core",
    "shared_body": "Incident review starts at noon.",
    "edit_message_id": "MSG023",
    "edit_body": "Graphs reviewed, latency back to baseline.",
    "edit_thread_parent_id": "MSG002",
    "edit_parent_body": "Latency watch is normal.",
    "neighbor_message_id": "MSG024",
    "neighbor_body": "Support is standing by.",
    "welcome_replies": {"MSG004": "Welcome Alice Nguyen", "MSG010": "Welcome Ben Ortíz"},
    "decoy_user_id": "U006",
    "decoy_user_display_name": "Ben Ortíz",
    "decoy_channel_ids": ["C006", "C007"],
    "final_answer": "Alice Nguyen;Ben Ortíz",
}

SLACK_TASK_2_SPEC = {
    "target_channel_id": "C005",
    "target_body": "Incident analysis started",
    "parent_id": "MSG004",
    "reply_body": "On it.",
    "group_name": "security-alerts",
    "final_group_name": "Design Ops Sync",
    "participants": ["U005", "U002"],
    "welcome_body": "Welcome @eva and @ben!",
}


def slack_task_1_test_verifier() -> LayeredVerifier:
    spec = SLACK_TASK_1_SPEC
    return LayeredVerifier(
        VerificationSpec(
            state_checks=[
                check_message_posted(spec["dm_chat_id"], spec["shared_body"]),
                check_message_posted(spec["group_chat_id"], spec["shared_body"]),
                check_chat_message_count(spec["dm_chat_id"], 3),
                check_chat_message_count(spec["group_chat_id"], 3),
                check_message_edited(spec["edit_message_id"], spec["edit_body"]),
                check_parent_message_unchanged(spec["neighbor_message_id"], spec["neighbor_body"]),
                check_parent_message_unchanged(spec["edit_thread_parent_id"], spec["edit_parent_body"]),
                thread_reply_posted("MSG004", spec["welcome_replies"]["MSG004"]),
                thread_reply_posted("MSG010", spec["welcome_replies"]["MSG010"]),
                check_body_prefix_only_in_threads("Welcome", spec["welcome_replies"]),
                check_no_new_messages_in_channels(spec["decoy_channel_ids"], allowed_count=1),
                user_display_name_equals(spec["decoy_user_id"], spec["decoy_user_display_name"]),
                chat_name_equals(spec["group_chat_id"], spec["group_name"]),
            ],
            invariant_checks=[
                agent_used_atif(),
                rl_determinism_check(),
            ],
            trajectory_checks=[
                check_tool_called_with_args("send_dm_message", body=spec["shared_body"]),
                check_tool_called_with_args("send_group_message", body=spec["shared_body"]),
                check_call_order("send_dm_message", {"body": spec["shared_body"]}, "send_group_message", {"body": spec["shared_body"]}),
                check_tool_called_with_args("update_message", message_id=spec["edit_message_id"], body=spec["edit_body"]),
                check_message_edit_timestamp_changed(spec["edit_message_id"]),
                check_messages_unmodified_since_initial([spec["neighbor_message_id"], spec["edit_thread_parent_id"], "MSG004", "MSG010"]),
                check_tool_called_with_args("slack.reply_to_thread", thread_parent_id="MSG004", body=spec["welcome_replies"]["MSG004"]),
                check_tool_called_with_args("slack.reply_to_thread", thread_parent_id="MSG010", body=spec["welcome_replies"]["MSG010"]),
                check_final_answer_equals(spec["final_answer"]),
            ],
            negative_checks=[
                forbidden_tool_not_called("create_channel"),
                forbidden_tool_not_called("create_group"),
                forbidden_tool_not_called("create_dm_message"),
                forbidden_tool_not_called("post_message"),
                forbidden_tool_not_called("change_channel_name"),
                forbidden_tool_not_called("change_group_name"),
                forbidden_tool_not_called("change_user_display_name"),
                forbidden_tool_not_called("add_reaction"),
                forbidden_tool_not_called("delete_message"),
            ],
        )
    )


def slack_task_2_test_verifier() -> LayeredVerifier:
    spec = SLACK_TASK_2_SPEC
    return LayeredVerifier(
        VerificationSpec(
            state_checks=[
                check_message_posted(spec["target_channel_id"], spec["target_body"]),
                thread_reply_posted(spec["parent_id"], spec["reply_body"]),
                heart_reaction_exists(),
                group_has_participants(spec["final_group_name"], spec["participants"]),
                group_message_posted(spec["final_group_name"], spec["welcome_body"]),
                final_answer_is_existing_group_id(),
            ],
            invariant_checks=[
                agent_used_atif(),
                rl_determinism_check(),
            ],
            trajectory_checks=[
                check_tool_called_with_args("post_message", channel_id=spec["target_channel_id"], body=spec["target_body"]),
                check_tool_called_with_args("slack.reply_to_thread", thread_parent_id=spec["parent_id"], body=spec["reply_body"]),
                check_tool_called_with_args("add_reaction", emoji="heart"),
                check_tool_called_with_args("create_group", name=spec["group_name"], participants=spec["participants"]),
                check_tool_called_with_args("send_group_message", body=spec["welcome_body"]),
                check_tool_called_with_args("change_group_name", new_name=spec["final_group_name"]),
                final_answer_equals_existing_group_id(),
            ],
            negative_checks=[
                forbidden_tool_not_called("create_channel"),
                forbidden_tool_not_called("change_channel_name"),
                forbidden_tool_not_called("create_dm_message"),
                forbidden_tool_not_called("send_dm_message"),
                forbidden_tool_not_called("change_user_display_name"),
                forbidden_tool_not_called("update_message"),
            ],
        )
    )


def slack_task_1_test_reward_checks() -> list[str]:
    return [
        "state: shared message exists in the DM chat",
        "state: shared message exists in the group chat",
        "state: DM chat has exactly one new message",
        "state: group chat has exactly one new message",
        "state: own thread reply edited to the new body",
        "state: neighbor reply by the decoy user untouched",
        "state: edited reply's thread parent untouched",
        "state: welcome reply under MSG004 names Alice Nguyen",
        "state: welcome reply under MSG010 names Ben Ortíz",
        "state: welcome replies appear only under the two threads",
        "state: decoy channels have no new messages",
        "state: decoy user display name untouched",
        "state: group chat keeps its original name",
        "invariant: trajectory is ATIF-v1.7 and starts with user",
        "invariant: reset determinism passed",
        "trajectory: send_dm_message called with shared body",
        "trajectory: send_group_message called with shared body",
        "trajectory: DM message sent before group message",
        "trajectory: update_message called on own reply with new body",
        "trajectory: edited reply's timestamp changed from initial",
        "trajectory: untouched thread messages match the initial snapshot",
        "trajectory: reply_to_thread under MSG004 with welcome body",
        "trajectory: reply_to_thread under MSG010 with welcome body",
        "trajectory: final answer is the two display names",
        "negative: no channel creation",
        "negative: no group creation",
        "negative: no DM creation",
        "negative: no channel posts",
        "negative: no channel renaming",
        "negative: no group renaming",
        "negative: no display name changes",
        "negative: no reactions",
        "negative: no message deletion",
    ]


# ---------------------------------------------------------------------------
# Checks with no shared-library equivalent
# ---------------------------------------------------------------------------

def user_display_name_equals(user_id: str, expected_display_name: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        users = _collection_values(state, "users")
        user = next((u for u in users if u.get("user_id") == user_id or u.get("id") == user_id), None)
        if not user:
            return False, f"Expected user {user_id} to exist.", {}
        passed = user.get("display_name") == expected_display_name
        return passed, f"Expected user {user_id} display name to be '{expected_display_name}'.", {
            "actual_display_name": user.get("display_name")
        }
    return check


def thread_reply_posted(parent_id: str, body: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        matches = [
            message
            for message in _collection_values(state, "messages")
            if message.get("thread_parent_id") == parent_id
            and message.get("body") == body
            and not message.get("deleted", False)
        ]
        return bool(matches), "Expected thread reply.", {"parent_id": parent_id, "matches": len(matches)}

    return check


def heart_reaction_exists():
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        matches = [
            reaction
            for reaction in _collection_values(state, "reactions")
            if reaction.get("emoji") == "heart"
        ]
        return bool(matches), "Expected heart reaction exists.", {"matches": len(matches)}

    return check


def group_has_participants(final_group_name: str, expected_user_ids: list[str]):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        group = _group_by_name(state, final_group_name)
        actual_user_ids = sorted(participant.get("user_id") for participant in group.get("participants", [])) if group else []
        expected = sorted(expected_user_ids)
        passed = bool(group) and set(expected).issubset(set(actual_user_ids))
        return passed, "Expected group participants by user ID.", {
            "group_name": final_group_name,
            "expected_user_ids": expected,
            "actual_user_ids": actual_user_ids,
        }

    return check


def group_message_posted(final_group_name: str, body: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        group = _group_by_name(state, final_group_name)
        group_id = _chat_id(group) if group else None
        matches = [
            message
            for message in _collection_values(state, "messages")
            if message.get("channel_id") == group_id
            and message.get("body") == body
            and not message.get("deleted", False)
        ]
        return bool(matches), "Expected group welcome message.", {"group_id": group_id, "matches": len(matches)}

    return check


def final_answer_is_existing_group_id():
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        group_ids = {_chat_id(chat) for chat in _collection_values(state, "chats") if chat.get("type") == "group"}
        return bool(group_ids), "Expected at least one group chat.", {"group_ids": sorted(group_ids)}

    return check


def chat_name_equals(chat_id: str, expected_name: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        chat = next((c for c in _collection_values(state, "chats") if _chat_id(c) == chat_id), None)
        if chat is None:
            return False, f"Expected chat {chat_id} to exist.", {}
        passed = chat.get("name") == expected_name
        return passed, f"Expected chat {chat_id} to still be named '{expected_name}'.", {
            "actual_name": chat.get("name")
        }

    return check


def final_answer_equals_existing_group_id():
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        actual = str(trajectory.get("extra", {}).get("final_answer", "")).strip()
        if not actual:
            agent_steps = [step for step in trajectory.get("steps", []) if step.get("source") == "agent"]
            actual = str(agent_steps[-1].get("message", "")).strip() if agent_steps else ""
        created_or_renamed_group_ids = [
            str(call.get("input_payload", {}).get("group_id") or call.get("input_payload", {}).get("chat_id") or "")
            for call in trajectory_tool_calls(trajectory)
            if call.get("tool_name") == "change_group_name"
        ]
        passed = bool(actual) and (not created_or_renamed_group_ids or actual in created_or_renamed_group_ids)
        return passed, "Expected final answer to be the changed group ID.", {
            "actual": actual,
            "changed_group_ids": created_or_renamed_group_ids,
        }

    return check


def _group_by_name(state: dict[str, Any], name: str) -> dict[str, Any] | None:
    for chat in _collection_values(state, "chats"):
        if chat.get("type") == "group" and chat.get("name") == name:
            return chat
    return None


def _collection_values(state: dict[str, Any], key: str) -> list[dict[str, Any]]:
    collection = state.get(key, {})
    if isinstance(collection, dict):
        return list(collection.values())
    if isinstance(collection, list):
        return [item for item in collection if isinstance(item, dict)]
    return []


def _chat_id(chat: dict[str, Any]) -> str:
    return str(chat.get("id") or chat.get("chat_id") or "")
