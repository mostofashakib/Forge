"""Dependency-free tests for the example task packaging contracts."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TASK_NAMES = {
    "slack_task_1": "forge/slack-task-1",
    "slack_task_2": "forge/slack-task-2",
    "task_manager_task_1": "forge/task-manager-task-1",
}


class TaskContractTests(unittest.TestCase):
    def test_task_metadata_uses_forge_namespace(self) -> None:
        for directory, expected_name in EXPECTED_TASK_NAMES.items():
            with self.subTest(task=directory):
                config_path = ROOT / directory / "task.toml"
                config = tomllib.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(expected_name, config["task"]["name"])

    def test_every_task_documents_the_root_runner(self) -> None:
        for directory in EXPECTED_TASK_NAMES:
            with self.subTest(task=directory):
                readme = (ROOT / directory / "README.md").read_text(encoding="utf-8")
                self.assertIn("./run.sh validate", readme)
                self.assertIn("./run.sh setup", readme)

    def test_slack_task_2_solution_implements_its_full_contract(self) -> None:
        solution = (ROOT / "slack_task_2" / "solution" / "solve.sh").read_text(
            encoding="utf-8"
        )
        required_commands = {
            "create_group",
            "post_message",
            "reply_to_thread",
            "add_reaction",
            "send_group_message",
            "change_group_name",
        }
        for command in required_commands:
            with self.subTest(command=command):
                self.assertIn(f"slack_env.py {command}", solution)
        self.assertIn('--channel-id C005', solution)
        self.assertIn('printf \'%s\\n\' "$GROUP_ID"', solution)


if __name__ == "__main__":
    unittest.main()
