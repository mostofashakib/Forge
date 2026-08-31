"""Tool payload validation against the declared schema in the SQLite service.

Regression context: the agent called bulk_update_tasks with `assignee_id`
(undeclared; the schema says `assignee`) and the dispatcher silently no-opped
while returning a success-shaped payload. The tool surface now has a single
mutation path (update_task) and every execute_tool payload is validated
against the declared input_schema, so contract violations fail loudly.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fleet.environments.task_manager import sqlite_service
from fleet.environments.task_manager.environment import TaskManagerEnvironment
from fleet.environments.task_manager.schema import TASK_MANAGER_TOOL_SCHEMA


class BulkUpdateTasksRemovalTests(unittest.TestCase):
    def test_bulk_update_tasks_is_not_in_schema(self) -> None:
        tool_names = {tool["name"] for tool in TASK_MANAGER_TOOL_SCHEMA}
        self.assertNotIn("bulk_update_tasks", tool_names)

    def test_bulk_update_tasks_is_not_an_environment_tool(self) -> None:
        self.assertNotIn("bulk_update_tasks", set(TaskManagerEnvironment(seed=19).tools))


class SqlitePayloadValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.db_path = root / "task_manager.db"
        self.snapshot_path = root / "snapshot.sql"
        sqlite_service.seed_database(self.db_path, self.snapshot_path)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def execute(self, tool_name: str, payload: dict) -> dict:
        return sqlite_service.execute_tool(self.db_path, tool_name, payload, "U001")

    def test_bulk_update_tasks_is_an_unknown_tool(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "[Uu]nknown tool"):
            self.execute("bulk_update_tasks", {"task_ids": ["TASK008"], "assignee": "U002"})

    def test_unknown_parameter_is_rejected_and_state_unchanged(self) -> None:
        # The exact misuse from the failed eval run: assignee_id is not in the
        # update_task schema, so it must fail loudly instead of no-opping.
        with self.assertRaisesRegex(RuntimeError, "assignee_id"):
            self.execute("update_task", {"task_id": "TASK008", "assignee_id": "U002"})
        task = self.execute("get_task", {"task_id": "TASK008"})
        self.assertEqual("U004", task["task"]["assignee_id"])

    def test_unknown_parameter_error_names_the_allowed_parameters(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "assignee"):
            self.execute("update_task", {"task_id": "TASK008", "assignee_id": "U002"})

    def test_missing_required_parameter_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "task_id"):
            self.execute("get_task", {})

    def test_invalid_enum_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "status"):
            self.execute("list_tasks", {"status": "NOT_A_STATUS"})

    def test_enum_values_match_case_insensitively(self) -> None:
        # normalize_status uppercases statuses, so lowercase enum spellings
        # remain valid inputs.
        result = self.execute("list_tasks", {"status": "pending"})
        self.assertGreater(len(result["tasks"]), 0)

    def test_null_for_optional_parameter_is_not_a_validation_error(self) -> None:
        # The SQLite service treats null as "leave unchanged"; validation must
        # not reject it on type grounds.
        result = self.execute("update_task", {"task_id": "TASK008", "milestone_id": None})
        self.assertIn("task", result)

    def test_valid_reassignment_still_succeeds(self) -> None:
        result = self.execute("update_task", {"task_id": "TASK008", "assignee": "U002"})
        self.assertEqual("U002", result["task"]["assignee_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
