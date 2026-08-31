from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tempfile

from fleet.core.serialization import canonical_json
from fleet.environments.slack.environment import SlackEnvironment
from fleet.environments.slack.sqlite_service import seed_database, update_message
from tests.deterministic_utils import print_deterministic_report


class SlackDeterministicTests(unittest.TestCase):
    def test_sqlite_edit_timestamps_advance_per_edit(self) -> None:
        # The edit clock must advance on every edit so a verifier can compare
        # edited_at_ms against the initial snapshot, including repeat edits of
        # the same message.
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "slack.db"
            seed_database(db_path, Path(temp_dir) / "snapshot.sql")
            first = update_message(db_path, "MSG001", "edit one", "U002")["message"]["edited_at_ms"]
            second = update_message(db_path, "MSG001", "edit two", "U002")["message"]["edited_at_ms"]
        self.assertIsNotNone(first)
        self.assertGreater(second, first)

    def test_harbor_tasks_use_local_service_environments(self) -> None:
        task_root = Path("slack_task_1")
        environment_root = task_root / "environment"
        task_config = (task_root / "task.toml").read_text(encoding="utf-8")
        slack_wrapper = (environment_root / "slack_env.py").read_text(encoding="utf-8")

        self.assertTrue((task_root / "instruction.md").exists())
        self.assertTrue((task_root / "tests" / "test.sh").exists())
        self.assertIn("[environment]", task_config)
        self.assertIn("[verifier]", task_config)
        self.assertIn("fleet.environments.slack.sqlite_service", slack_wrapper)
        self.assertTrue((environment_root / "Dockerfile").exists())
        self.assertTrue((environment_root / "docker-compose.yaml").exists())
        self.assertTrue((task_root / "tests" / "Dockerfile").exists())
        self.assertTrue((task_root / "tests" / "docker-compose.yaml").exists())
        self.assertFalse((environment_root / "fleet").exists())
        self.assertFalse((task_root / "tests" / "fleet").exists())
        self.assertNotIn("docker_image =", task_config)
        self.assertTrue(Path("fleet/environments/slack/sqlite_service.py").exists())
        self.assertTrue(Path("fleet/environments/task_manager/sqlite_service.py").exists())

    def test_slack_reset_restores_byte_identical_state(self) -> None:
        environment = SlackEnvironment(seed=42)
        initial = canonical_json(environment.export_state())
        general_id = next(
            channel["channel_id"]
            for channel in environment.export_state()["channels"]
            if channel["name"] == "general"
        )
        user_id = environment.export_state()["users"][0]["user_id"]

        environment.execute_tool(
            "post_message",
            {"channel_id": general_id, "body": "Deterministic hello"},
            user_id,
        )
        mutated_state = environment.export_state()

        self.assertNotEqual(initial, canonical_json(environment.export_state()))
        environment.reset()
        print_deterministic_report(
            "Slack reset restores byte-identical state",
            "Post a Slack message, then reset the environment.",
            json.loads(initial),
            [{"tool_name": "post_message", "input_payload": {"channel_id": general_id, "body": "Deterministic hello"}}],
            mutated_state,
            environment.export_state(),
            {"expected": "reset state equals initial state", "actual": canonical_json(environment.export_state()) == initial},
        )
        self.assertEqual(initial, canonical_json(environment.export_state()))

    def test_slack_duplicate_reaction_returns_error_state(self) -> None:
        environment = SlackEnvironment(seed=7)
        state = environment.export_state()
        message_id = state["messages"][0]["message_id"]
        user_id = state["users"][0]["user_id"]

        environment.execute_tool("add_reaction", {"message_id": message_id, "emoji": "eyes"}, user_id)
        observation = environment.execute_tool("add_reaction", {"message_id": message_id, "emoji": "eyes"}, user_id)
        print_deterministic_report(
            "Slack duplicate reaction returns deterministic ErrorState",
            "Add the same reaction twice.",
            state,
            [
                {"tool_name": "add_reaction", "input_payload": {"message_id": message_id, "emoji": "eyes"}},
                {"tool_name": "add_reaction", "input_payload": {"message_id": message_id, "emoji": "eyes"}},
            ],
            environment.export_state(),
            environment.export_state(),
            {"expected": "duplicate_reaction", "actual": observation.error.error_code if observation.error else None},
        )

        self.assertIsNotNone(observation.error)
        self.assertEqual("duplicate_reaction", observation.error.error_code)
        self.assertFalse(observation.error.state_changed)

    def test_slack_incident_fixture_matches_requested_shape(self) -> None:
        environment = SlackEnvironment(seed=1)
        state = environment.export_state()
        user_id = "U002"

        search = environment.execute_tool("search_messages", {"query": "alice incidents"}, user_id)
        channel = environment.execute_tool("get_channel_messages", {"channel_id": "C003"}, user_id)
        print_deterministic_report(
            "Slack incident fixture and read tools are deterministic",
            "Search Alice incident messages and read #incidents.",
            state,
            [
                {"tool_name": "search_messages", "input_payload": {"query": "alice incidents"}},
                {"tool_name": "get_channel_messages", "input_payload": {"channel_id": "C003"}},
            ],
            environment.export_state(),
            environment.export_state(),
            {
                "expected": {"search_count": 4, "channel_count": 9, "reference": "#platform-outages"},
                "actual": {
                    "search_count": search.payload["output"]["count"],
                    "channel_count": channel.payload["output"]["count"],
                    "top_message": search.payload["output"]["messages"][0]["body"],
                },
            },
        )

        self.assertEqual(6, len(state["users"]))
        self.assertEqual(7, len(state["channels"]))
        self.assertEqual(26, len(state["messages"]))
        self.assertEqual(4, search.payload["output"]["count"])
        self.assertEqual(9, channel.payload["output"]["count"])
        self.assertIn("#platform-outages", search.payload["output"]["messages"][0]["body"])
    def test_post_message_to_dm_or_group_returns_error(self) -> None:
        environment = SlackEnvironment(seed=42)
        state = environment.export_state()
        user_id = state["users"][0]["user_id"]

        # Test DM chat ID
        dm_id = "D001"
        res_dm = environment.execute_tool("post_message", {"channel_id": dm_id, "body": "Hello DM"}, user_id)
        self.assertTrue(res_dm.error)
        self.assertEqual("channel_not_found", res_dm.error.error_code)

        # Test Group chat ID
        group_id = "G001"
        res_group = environment.execute_tool("post_message", {"channel_id": group_id, "body": "Hello Group"}, user_id)
        self.assertTrue(res_group.error)
        self.assertEqual("channel_not_found", res_group.error.error_code)


if __name__ == "__main__":
    unittest.main(verbosity=2)

