from fleet.verifiers import (
    LayeredVerifier,
    VerificationSpec,
    forbidden_tool_not_called,
)
from fleet.verifiers.rewardkit_checks import register_harbor_verifier
from fleet.verifiers.verifier_specs.slack import (
    slack_seed_shape,
    agent_used_atif,
    rl_determinism_check,
    check_message_in_chat,
    check_chat_message_count,
    check_message_edited,
    check_parent_message_unchanged,
    check_thread_reply,
    check_body_prefix_only_in_threads,
    check_no_new_messages_in_channels,
    check_message_edit_timestamp_changed,
    check_messages_unmodified_since_initial,
    check_final_answer_equals,
    check_call_order,
    check_tool_called_with_args,
)

# Trap map (all decoys are seeded):
#   - The shared body must land in the DM (D001) and the group (G001) — not in
#     any channel, and the DM/group must be the existing ones, not new chats.
#   - DM first, then group: enforced as a trajectory call-order check.
#   - In the MSG002 thread, Ben's own reply (MSG023) sits next to a reply by
#     the look-alike user "Ben Ortíz" (MSG024); only MSG023 may be edited.
#   - The last commenter in the MSG004 thread is Alice Nguyen (MSG014); in the
#     MSG010 thread it is the decoy user "Ben Ortíz" (MSG026) — the welcome
#     replies and the final answer must transcribe the accented name exactly.

def check_user_display_name(user_id: str, expected_display_name: str):
    def check(state: dict) -> tuple[bool, str, dict]:
        users = state.get("users", [])
        user = next((u for u in users if u.get("user_id") == user_id), None)
        if not user:
            return False, f"Expected user {user_id} to exist.", {}
        passed = user.get("display_name") == expected_display_name
        return passed, f"Expected user {user_id} display name to be '{expected_display_name}'.", {
            "actual_display_name": user.get("display_name")
        }
    return check

def check_chat_name(chat_id: str, expected_name: str):
    def check(state: dict) -> tuple[bool, str, dict]:
        chat = next((c for c in state.get("chats", []) if c.get("chat_id") == chat_id), None)
        if not chat:
            return False, f"Expected chat {chat_id} to exist.", {}
        passed = chat.get("name") == expected_name
        return passed, f"Expected chat {chat_id} to still be named '{expected_name}'.", {
            "actual_name": chat.get("name")
        }
    return check

SHARED_BODY = "Incident review starts at noon."
EDIT_BODY = "Graphs reviewed, latency back to baseline."
WELCOME_REPLIES = {"MSG004": "Welcome Alice Nguyen", "MSG010": "Welcome Ben Ortíz"}
FINAL_ANSWER = "Alice Nguyen;Ben Ortíz"


def slack_task_1_verifier() -> LayeredVerifier:
    return LayeredVerifier(
        VerificationSpec(
            state_checks=[
                check_message_in_chat("D001", SHARED_BODY, "U002"),
                check_message_in_chat("G001", SHARED_BODY, "U002"),
                check_chat_message_count("D001", 3),
                check_chat_message_count("G001", 3),
                check_message_edited("MSG023", EDIT_BODY),
                check_parent_message_unchanged("MSG024", "Support is standing by."),
                check_parent_message_unchanged("MSG002", "Latency watch is normal."),
                check_thread_reply("C003", "MSG004", WELCOME_REPLIES["MSG004"]),
                check_thread_reply("C001", "MSG010", WELCOME_REPLIES["MSG010"]),
                check_body_prefix_only_in_threads("Welcome", WELCOME_REPLIES),
                check_no_new_messages_in_channels(["C006", "C007"], allowed_count=1),
                check_user_display_name("U006", "Ben Ortíz"),
                check_chat_name("G001", "Incident Response Core"),
            ],
            invariant_checks=[slack_seed_shape(), agent_used_atif(), rl_determinism_check()],
            trajectory_checks=[
                check_tool_called_with_args("send_dm_message", body=SHARED_BODY),
                check_tool_called_with_args("send_group_message", body=SHARED_BODY),
                check_call_order(
                    "send_dm_message", {"body": SHARED_BODY},
                    "send_group_message", {"body": SHARED_BODY},
                ),
                check_tool_called_with_args("update_message", message_id="MSG023", body=EDIT_BODY),
                check_message_edit_timestamp_changed("MSG023"),
                check_messages_unmodified_since_initial(["MSG024", "MSG002", "MSG004", "MSG010"]),
                check_tool_called_with_args("slack.reply_to_thread", thread_parent_id="MSG004", body=WELCOME_REPLIES["MSG004"]),
                check_tool_called_with_args("slack.reply_to_thread", thread_parent_id="MSG010", body=WELCOME_REPLIES["MSG010"]),
                check_final_answer_equals(FINAL_ANSWER),
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

def slack_task_1_reward_checks() -> list[str]:
    return [
        "state: shared message exists in the DM chat D001",
        "state: shared message exists in the group chat G001",
        "state: DM chat has exactly one new message",
        "state: group chat has exactly one new message",
        "state: own reply MSG023 edited to the new body",
        "state: decoy user's neighbor reply MSG024 untouched",
        "state: thread parent MSG002 untouched",
        "state: welcome reply under MSG004 names Alice Nguyen",
        "state: welcome reply under MSG010 names Ben Ortíz",
        "state: welcome replies appear only under the two threads",
        "state: decoy channels C006 and C007 gained no messages",
        "state: decoy user U006 display name untouched",
        "state: group chat G001 keeps its original name",
        "invariant: seeded Slack workspace has expected table counts",
        "invariant: trajectory is ATIF-v1.7 and starts with the user instruction",
        "invariant: environment reset determinism passed",
        "trajectory: agent called send_dm_message with the shared body",
        "trajectory: agent called send_group_message with the shared body",
        "trajectory: DM message sent before the group message",
        "trajectory: agent called update_message on MSG023 with the new body",
        "trajectory: MSG023 edit timestamp changed from its initial value",
        "trajectory: untouched thread messages match the initial snapshot",
        "trajectory: agent replied under MSG004 with the Alice welcome body",
        "trajectory: agent replied under MSG010 with the Ortíz welcome body",
        "trajectory: final answer is the two display names",
        "negative: agent did not create a channel",
        "negative: agent did not create a group",
        "negative: agent did not create a DM",
        "negative: agent did not post to a channel",
        "negative: agent did not rename a channel",
        "negative: agent did not rename a group",
        "negative: agent did not change display names",
        "negative: agent did not add reactions",
        "negative: agent did not delete messages",
    ]

if not __name__.endswith("_check"):
    register_harbor_verifier(
        "check:slack_task_1_verifier",
        "slack_task_1",
        slack_task_1_reward_checks(),
    )
