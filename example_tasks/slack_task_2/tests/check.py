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
    check_message_posted,
    check_parent_message_unchanged,
    check_thread_reply,
    check_thread_reaction,
    check_group_created,
    check_group_message,
    check_group_renamed,
    check_final_answer_is_group_id,
    check_call_order,
    check_tool_called_with_args,
)


def slack_task_2_verifier() -> LayeredVerifier:
    return LayeredVerifier(
        VerificationSpec(
            state_checks=[
                check_message_posted("C005", "Incident analysis started"),
                check_parent_message_unchanged("MSG004", "Incident bridge opened with SRE."),
                check_thread_reply("C003", "MSG004", "On it."),
                check_thread_reaction("C003", "MSG004", "On it.", "heart"),
                check_group_created(["U005", "U002"]),
                check_group_message("Welcome @eva and @ben!"),
                check_group_renamed("security-alerts", "Design Ops Sync"),
            ],
            invariant_checks=[slack_seed_shape(), agent_used_atif(), rl_determinism_check()],
            trajectory_checks=[
                check_tool_called_with_args("post_message", channel_id="C005", body="Incident analysis started"),
                check_tool_called_with_args("slack.reply_to_thread", thread_parent_id="MSG004", body="On it."),
                check_tool_called_with_args("add_reaction", emoji="heart"),
                check_tool_called_with_args("create_group", name="security-alerts", participants=["U005", "U002"]),
                check_tool_called_with_args("send_group_message", body="Welcome @eva and @ben!"),
                check_tool_called_with_args("change_group_name", new_name="Design Ops Sync"),
                check_call_order(
                    "create_group", {"name": "security-alerts"},
                    "post_message", {"channel_id": "C005"},
                ),
                check_call_order(
                    "slack.reply_to_thread", {"thread_parent_id": "MSG004", "body": "On it."},
                    "add_reaction", {"emoji": "heart"},
                ),
                check_call_order(
                    "send_group_message", {"body": "Welcome @eva and @ben!"},
                    "change_group_name", {"new_name": "Design Ops Sync"},
                ),
                check_final_answer_is_group_id(),
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


def slack_task_2_reward_checks() -> list[str]:
    return [
        "state: message exists in channel C005 with expected body",
        "state: parent message MSG004 remains unchanged",
        "state: threaded reply exists under MSG004 in channel C003",
        "state: heart reaction exists on the threaded reply",
        "state: group contains expected user IDs U002, U005",
        "state: welcome message exists in the created group",
        "state: group was renamed without retaining the initial group name",
        "invariant: seeded Slack workspace has expected table counts",
        "invariant: trajectory is ATIF-v1.7 and starts with the user instruction",
        "invariant: environment reset determinism passed",
        "trajectory: agent called post_message with incident analysis args",
        "trajectory: agent called slack.reply_to_thread with bridge reply args",
        "trajectory: agent called add_reaction with heart emoji",
        "trajectory: agent called create_group with security alerts participants",
        "trajectory: agent called send_group_message with welcome args",
        "trajectory: agent called change_group_name to design ops sync",
        "trajectory: group was created before the channel message was posted",
        "trajectory: thread reply was created before the heart reaction",
        "trajectory: welcome message was sent before the group was renamed",
        "trajectory: final answer is the created group chat ID",
        "negative: agent did not create a channel",
        "negative: agent did not rename a channel",
        "negative: agent did not create a DM message",
        "negative: agent did not send a DM message",
        "negative: agent did not change user display name",
        "negative: agent did not edit messages",
    ]


if not __name__.endswith("_check"):
    register_harbor_verifier(
        "check:slack_task_2_verifier",
        "slack_task_2",
        slack_task_2_reward_checks(),
    )
