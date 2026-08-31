from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fleet.environments.slack.environment import SlackEnvironment
from fleet.verifiers import LayeredVerifier, VerificationSpec, forbidden_tool_not_called, required_tool_called
from tests.simulation_driver import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_AGENT_PROVIDER,
    builtin_slack_incident_task,
    legacy_artifacts,
    reset_determinism_check,
    run_task_list,
)
from tests.smoke_utils import SMOKE_TRANSCRIPT_PATH, initialize_smoke_transcript, print_task_report


class SlackSmokeTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        initialize_smoke_transcript()

    def test_slack_environment_task_and_verifier_are_functional(self) -> None:
        task = builtin_slack_incident_task()
        bundle = run_task_list(
            tasks=[task],
            seed=1,
            output_path="/tmp/fleet_smoke/slack.json",
            verbose=True,
            agent_provider=DEFAULT_AGENT_PROVIDER,
            agent_model=DEFAULT_AGENT_MODEL,
            real_agent=True,
            transcript_path=SMOKE_TRANSCRIPT_PATH,
            transcript_append=True,
        )
        trajectory = bundle["runs"][0]["trajectory"]
        artifacts = legacy_artifacts(trajectory)

        verifier = LayeredVerifier(
            VerificationSpec(
                state_checks=[most_recent_alice_incident_references("#platform-outages")],
                invariant_checks=[],
                trajectory_checks=[
                    required_tool_called("search_messages"),
                    required_tool_called("get_channel_messages"),
                ],
                negative_checks=[forbidden_tool_not_called("post_message")],
            )
        )
        results = verifier.verify(artifacts["final_state_snapshot"], artifacts)
        print_task_report(task, trajectory, results)

        self.assertEqual("slack", task.environment_name)
        self.assertIn("@alice", task.instruction)
        self.assertTrue(all(result.passed for result in results), results)
        self.assertTrue(reset_determinism_check(trajectory)["passed"])




class SlackToolSmokeTests(unittest.TestCase):
    """Detailed Slack tool calls, permutations, determinism and negative scenarios."""

    def setUp(self) -> None:
        self.env = SlackEnvironment(seed=42)

    def test_slack_environment_determinism_check(self) -> None:
        # 1. Capture initial state
        initial_state = self.env.export_state()
        initial_state_json = json.dumps(initial_state, sort_keys=True)

        # 2. Make some mutations
        self.env.execute_tool("post_message", {"channel_id": "C001", "body": "test message"}, "U002")
        self.env.execute_tool("create_channel", {"name": "temp-channel", "is_private": False}, "U001")

        # Verify state has changed
        mutated_state = self.env.export_state()
        self.assertNotEqual(initial_state, mutated_state)

        # 3. Perform Reset
        reset_state = self.env.reset()
        reset_state_json = json.dumps(reset_state, sort_keys=True)

        # 4. Assert byte-stable match
        self.assertEqual(initial_state_json, reset_state_json)

    def test_all_slack_tools_functional(self) -> None:
        """Basic verification of all 15 tool interfaces."""
        # 1. list_channels
        res = self.env.execute_tool("list_channels", {}, "U002")
        self.assertFalse(res.error)
        self.assertTrue(any(c["channel_id"] == "C001" for c in res.payload["output"]["channels"]))

        # 2. create_channel
        res = self.env.execute_tool("create_channel", {"name": "marketing", "is_private": False}, "U001")
        self.assertFalse(res.error)
        mkt_channel_id = res.payload["output"]["id"]

        # 3. change_channel_name
        res = self.env.execute_tool("change_channel_name", {"channel_id": mkt_channel_id, "new_name": "marketing-campaigns"}, "U001")
        self.assertFalse(res.error)
        self.assertEqual(res.payload["output"]["channel"]["name"], "marketing-campaigns")

        # 4. post_message
        res = self.env.execute_tool("post_message", {"channel_id": "C001", "body": "Hello everyone!"}, "U002")
        self.assertFalse(res.error)
        msg_id = res.payload["output"]["id"]

        # 5. update_message
        res = self.env.execute_tool("update_message", {"message_id": msg_id, "body": "Hello everyone! Edited."}, "U002")
        self.assertFalse(res.error)
        self.assertEqual(res.payload["output"]["message"]["body"], "Hello everyone! Edited.")

        # 6. reply_to_thread
        res = self.env.execute_tool("slack.reply_to_thread", {"thread_parent_id": msg_id, "body": "Thread reply here"}, "U003")
        self.assertFalse(res.error)
        reply_id = res.payload["output"]["id"]

        # 7. add_reaction
        res = self.env.execute_tool("add_reaction", {"message_id": msg_id, "emoji": "thumbsup"}, "U004")
        self.assertFalse(res.error)

        # 8. get_channel_messages
        res = self.env.execute_tool("get_channel_messages", {"channel_id": "C001"}, "U002")
        self.assertFalse(res.error)
        self.assertTrue(res.payload["output"]["count"] > 0)

        # 9. create_group
        res = self.env.execute_tool("create_group", {"name": "design-review", "participants": ["U002", "U003"]}, "U001")
        self.assertFalse(res.error)
        group_id = res.payload["output"]["id"]

        # 10. change_group_name
        res = self.env.execute_tool("change_group_name", {"group_id": group_id, "new_name": "design-review-weekly"}, "U001")
        self.assertFalse(res.error)
        self.assertEqual(res.payload["output"]["chat"]["name"], "design-review-weekly")

        # 11. send_group_message
        res = self.env.execute_tool("send_group_message", {"group_id": group_id, "body": "Let's review designs"}, "U001")
        self.assertFalse(res.error)

        # 12. create_dm_message
        res = self.env.execute_tool("create_dm_message", {"recipient_id": "U003"}, "U001")
        self.assertFalse(res.error)
        dm_id = res.payload["output"]["id"]

        # 13. send_dm_message
        res = self.env.execute_tool("send_dm_message", {"chat_id": dm_id, "body": "DM text"}, "U001")
        self.assertFalse(res.error)

        # 14. change_user_display_name
        res = self.env.execute_tool("change_user_display_name", {"user_id": "U001", "new_display_name": "Alice N."}, "U001")
        self.assertFalse(res.error)

        # 15. search_messages
        res = self.env.execute_tool("search_messages", {"query": "Edited"}, "U002")
        self.assertFalse(res.error)
        self.assertEqual(res.payload["output"]["count"], 1)

    def test_permutations_and_combinations(self) -> None:
        """Complex scenario flows."""
        # Flow: Create private channel, verify list_channels ignores for non-members
        res = self.env.execute_tool("create_channel", {"name": "confidential", "is_private": True}, "U003")
        channel_id = res.payload["output"]["id"]

        # U002 lists channels -> should not see "confidential"
        res_list = self.env.execute_tool("list_channels", {}, "U002")
        channels = [c["name"] for c in res_list.payload["output"]["channels"]]
        self.assertNotIn("confidential", channels)

        # U003 (member/owner) lists channels -> should see "confidential"
        res_list = self.env.execute_tool("list_channels", {}, "U003")
        channels = [c["name"] for c in res_list.payload["output"]["channels"]]
        self.assertIn("confidential", channels)

        # U003 posts message in "confidential"
        res_post = self.env.execute_tool("post_message", {"channel_id": channel_id, "body": "private leak: password is test123"}, "U003")
        self.assertFalse(res_post.error)

        # U002 searches for "password" -> should return 0 results (private)
        res_search = self.env.execute_tool("search_messages", {"query": "password"}, "U002")
        self.assertEqual(res_search.payload["output"]["count"], 0)

        # U003 searches for "password" -> should return 1 result
        res_search = self.env.execute_tool("search_messages", {"query": "password"}, "U003")
        self.assertEqual(res_search.payload["output"]["count"], 1)

    def test_negative_cases(self) -> None:
        """Error handling and unauthorized access scenarios."""
        # 1. Access private channel without membership
        # C004 is design-confidential (owner: U005)
        res = self.env.execute_tool("get_channel_messages", {"channel_id": "C004"}, "U001")
        self.assertTrue(res.error)
        self.assertEqual(res.error.error_code, "permission_denied")

        # 2. Post to private channel without membership
        res = self.env.execute_tool("post_message", {"channel_id": "C004", "body": "spam"}, "U001")
        self.assertTrue(res.error)

        # 3. Update message of another user
        # MSG001 is authored by U002
        res = self.env.execute_tool("update_message", {"message_id": "MSG001", "body": "hacked"}, "U003")
        self.assertTrue(res.error)
        self.assertEqual(res.error.error_code, "permission_denied")

        # 4. Change display name of another user without admin role
        # U002 is member, U003 is member
        res = self.env.execute_tool("change_user_display_name", {"user_id": "U003", "new_display_name": "Hacked Cara"}, "U002")
        self.assertTrue(res.error)
        self.assertEqual(res.error.error_code, "permission_denied")

        # 5. Duplicate reaction check
        # Add reaction
        self.env.execute_tool("add_reaction", {"message_id": "MSG001", "emoji": "smile"}, "U002")
        # Add same reaction again
        res = self.env.execute_tool("add_reaction", {"message_id": "MSG001", "emoji": "smile"}, "U002")
        self.assertTrue(res.error)
        self.assertEqual(res.error.error_code, "duplicate_reaction")

        # 6. Post message with empty body
        res = self.env.execute_tool("post_message", {"channel_id": "C001", "body": ""}, "U002")
        self.assertTrue(res.error)
        self.assertEqual(res.error.error_code, "invalid_arguments")





def most_recent_alice_incident_references(expected_reference: str):
    def check(state):
        incidents_id = next(
            channel["channel_id"] for channel in state["channels"] if channel["name"] == "incidents"
        )
        alice_id = next(user["user_id"] for user in state["users"] if user["handle"] == "alice")
        messages = [
            message
            for message in state["messages"]
            if message["channel_id"] == incidents_id and message["author_id"] == alice_id
        ]
        most_recent = max(messages, key=lambda message: message["created_at_ms"])
        passed = expected_reference in most_recent["body"]
        return (
            passed,
            "Most recent Alice incident message references expected channel.",
            {"message_id": most_recent["message_id"], "expected_reference": expected_reference},
        )

    return check




if __name__ == "__main__":
    unittest.main(verbosity=2)
