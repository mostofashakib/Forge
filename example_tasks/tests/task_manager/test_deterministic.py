from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fleet.core.serialization import canonical_json
from fleet.environments.task_manager.environment import TaskManagerEnvironment
from fleet.environments.task_manager.schema import TASK_MANAGER_TOOL_SCHEMA
from fleet.environments.task_manager.sqlite_service import (
    create_task as sqlite_create_task,
    delete_task as sqlite_delete_task,
    export_state as sqlite_export_state,
    get_task as sqlite_get_task,
    seed_database as sqlite_seed_database,
    update_task as sqlite_update_task,
)
from fleet.verifiers import LayeredVerifier, VerificationSpec, forbidden_tool_not_called, required_tool_called, task_has_status
from tests.deterministic_utils import print_deterministic_report


class TaskManagerDeterministicTests(unittest.TestCase):
    def test_task_status_transition_records_audit_and_trajectory(self) -> None:
        environment = TaskManagerEnvironment(seed=11)
        state = environment.export_state()
        task_id = next(task["task_id"] for task in state["tasks"] if task["status"] == "PENDING")
        admin_id = next(user["user_id"] for user in state["users"] if user["role"] == "admin")

        environment.execute_tool("update_task", {"task_id": task_id, "status": "IN_PROGRESS"}, admin_id)
        final_state = environment.export_state()
        print_deterministic_report(
            "Task status transition records audit and trajectory",
            "Move a PENDING task to IN_PROGRESS.",
            state,
            [{"tool_name": "update_task", "input_payload": {"task_id": task_id, "status": "IN_PROGRESS"}}],
            final_state,
            final_state,
            {
                "expected": "task status IN_PROGRESS with one audit event",
                "actual": {
                    "status": next(t for t in final_state["tasks"] if t["task_id"] == task_id)["status"],
                    "audit_events": len(final_state["audit_events"]),
                },
            },
        )

        self.assertEqual("IN_PROGRESS", next(t for t in final_state["tasks"] if t["task_id"] == task_id)["status"])
        self.assertEqual(1, len(final_state["audit_events"]))
        self.assertEqual("update_task", environment.export_trajectory()["tool_calls"][0]["tool_name"])

    def test_task_manager_crud_tools(self) -> None:
        environment = TaskManagerEnvironment(seed=17)
        admin_id = next(user["user_id"] for user in environment.export_state()["users"] if user["role"] == "admin")

        create_result = environment.execute_tool(
            "create_task",
            {
                "task_id": "TASK999",
                "title": "Prepare launch checklist",
                "description": "Verify rollout owners and gates.",
                "assignee": "U003",
                "status": "PENDING",
            },
            admin_id,
        )
        self.assertIsNone(create_result.error)
        self.assertEqual("TASK999", create_result.payload["output"]["id"])

        get_result = environment.execute_tool("get_task", {"task_id": "TASK999"}, admin_id)
        self.assertIsNone(get_result.error)
        self.assertEqual("U003", get_result.payload["output"]["task"]["assignee_id"])

        update_result = environment.execute_tool(
            "update_task",
            {"task_id": "TASK999", "status": "IN_PROGRESS", "description": "Updated rollout checklist."},
            admin_id,
        )
        self.assertIsNone(update_result.error)
        self.assertEqual("IN_PROGRESS", update_result.payload["output"]["task"]["status"])

        delete_result = environment.execute_tool("delete_task", {"task_id": "TASK999"}, admin_id)
        self.assertIsNone(delete_result.error)
        self.assertEqual("DELETED", delete_result.payload["output"]["task"]["status"])

        list_result = environment.execute_tool("list_tasks", {}, admin_id)
        self.assertNotIn("TASK999", {task["task_id"] for task in list_result.payload["output"]["tasks"]})
        list_deleted_result = environment.execute_tool("list_tasks", {"include_deleted": True}, admin_id)
        self.assertIn("TASK999", {task["task_id"] for task in list_deleted_result.payload["output"]["tasks"]})

    def test_task_manager_schema_uses_canonical_tool_names(self) -> None:
        schema_tool_names = {tool["name"] for tool in TASK_MANAGER_TOOL_SCHEMA}
        environment_tool_names = set(TaskManagerEnvironment(seed=19).tools)
        expected_tools = {
            "list_tasks", "get_task", "create_task", "update_task", "delete_task",
            "archive_task", "mark_task_duplicate",
            "create_project", "list_projects", "get_project",
            "create_milestone",
            "move_task_to_project", "link_tasks", "unlink_tasks",
        }

        self.assertEqual(expected_tools, schema_tool_names)
        self.assertEqual(expected_tools, environment_tool_names)
        self.assertFalse(any(name.startswith("task.") for name in schema_tool_names))
        self.assertFalse(any(name.startswith("task.") for name in environment_tool_names))

    def test_task_manager_sqlite_crud_parity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "task_manager.db"
            snapshot_path = Path(temp_dir) / "task_manager_seed_snapshot.sql"
            sqlite_seed_database(db_path, snapshot_path)

            created = sqlite_create_task(
                db_path,
                title="Prepare launch checklist",
                description="Verify rollout owners and gates.",
                actor_id="U001",
                task_id="TASK999",
                assignee_id="U003",
                status="PENDING",
            )
            fetched = sqlite_get_task(db_path, "TASK999")
            updated = sqlite_update_task(db_path, "TASK999", actor_id="U001", status="IN_PROGRESS")
            deleted = sqlite_delete_task(db_path, "TASK999", actor_id="U001")
            final_state = sqlite_export_state(db_path)

        self.assertEqual("TASK999", created["id"])
        self.assertEqual("U003", fetched["task"]["assignee_id"])
        self.assertEqual("IN_PROGRESS", updated["task"]["status"])
        self.assertEqual("DELETED", deleted["task"]["status"])
        self.assertTrue(next(task for task in final_state["tasks"] if task["task_id"] == "TASK999")["deleted"])

    def test_invalid_task_transition_is_deterministic_error(self) -> None:
        first = TaskManagerEnvironment(seed=3)
        second = TaskManagerEnvironment(seed=3)
        task_id = next(task["task_id"] for task in first.export_state()["tasks"] if task["status"] == "PENDING")
        admin_id = next(user["user_id"] for user in first.export_state()["users"] if user["role"] == "admin")

        first.execute_tool("update_task", {"task_id": task_id, "status": "COMPLETED"}, admin_id)
        second.execute_tool("update_task", {"task_id": task_id, "status": "COMPLETED"}, admin_id)
        print_deterministic_report(
            "Invalid task transition is reproducible",
            "Attempt PENDING -> COMPLETED twice with the same seed.",
            first.export_state(),
            [{"tool_name": "update_task", "input_payload": {"task_id": task_id, "status": "COMPLETED"}}],
            first.export_state(),
            second.export_state(),
            {
                "expected": "invalid_status_transition and identical trajectories",
                "actual": first.export_trajectory()["errors"][0]["error_code"],
            },
        )

        self.assertEqual(canonical_json(first.export_trajectory()), canonical_json(second.export_trajectory()))
        self.assertEqual("invalid_status_transition", first.export_trajectory()["errors"][0]["error_code"])

    def test_layered_verifier_scores_properties(self) -> None:
        environment = TaskManagerEnvironment(seed=5)
        state = environment.export_state()
        task_id = next(task["task_id"] for task in state["tasks"] if task["title"] == "Draft benchmark plan")
        admin_id = next(user["user_id"] for user in state["users"] if user["role"] == "admin")
        environment.execute_tool("update_task", {"task_id": task_id, "status": "IN_PROGRESS"}, admin_id)

        verifier = LayeredVerifier(
            VerificationSpec(
                state_checks=[task_has_status("Draft benchmark plan", "IN_PROGRESS")],
                invariant_checks=[],
                trajectory_checks=[required_tool_called("update_task")],
                negative_checks=[forbidden_tool_not_called("delete_task")],
            )
        )
        results = verifier.verify(environment.export_state(), environment.export_trajectory())
        print_deterministic_report(
            "Layered verifier scores state and trajectory",
            "Verify task status and required tool usage.",
            state,
            [{"tool_name": "update_task", "input_payload": {"task_id": task_id, "status": "IN_PROGRESS"}}],
            environment.export_state(),
            environment.export_state(),
            {"expected": [1, 1, 1], "actual": [result.score for result in results]},
        )

        self.assertTrue(all(result.passed for result in results))
        self.assertEqual([1, 1, 1], [result.score for result in results])


class TaskManagerFinalAnswerCheckTests(unittest.TestCase):
    EXPECTED_IDS = ["TASK006", "TASK008", "TASK009", "TASK031", "TASK032"]

    def _run(self, final_answer: str | None, steps: list[dict] | None = None):
        from fleet.verifiers.verifier_specs.task_manager import tm_final_answer_is_task_ids

        trajectory: dict = {"extra": {}, "steps": steps or []}
        if final_answer is not None:
            trajectory["extra"]["final_answer"] = final_answer
        passed, _message, details = tm_final_answer_is_task_ids(self.EXPECTED_IDS)(trajectory)
        return passed, details

    def test_exact_oracle_answer_passes(self) -> None:
        passed, _ = self._run("TASK006,TASK008,TASK009,TASK031,TASK032")
        self.assertTrue(passed)

    def test_order_insensitive_and_spaces_tolerated(self) -> None:
        passed, _ = self._run("TASK032, TASK006, TASK008, TASK031, TASK009")
        self.assertTrue(passed)

    def test_missing_task_id_fails(self) -> None:
        # The gemma4:26b eval run answered with TASK009 omitted; this must fail.
        passed, details = self._run("TASK032,TASK006,TASK008,TASK031")
        self.assertFalse(passed)
        self.assertNotIn("TASK009", details["parsed_ids"])

    def test_empty_answer_fails(self) -> None:
        passed, _ = self._run("")
        self.assertFalse(passed)

    def test_extra_task_id_fails(self) -> None:
        passed, _ = self._run("TASK006,TASK008,TASK009,TASK031,TASK032,TASK001")
        self.assertFalse(passed)

    def test_duplicate_task_id_fails(self) -> None:
        passed, _ = self._run("TASK006,TASK008,TASK009,TASK031,TASK032,TASK032")
        self.assertFalse(passed)

    def test_falls_back_to_last_agent_step_message(self) -> None:
        steps = [
            {"source": "user", "message": "instruction"},
            {"source": "agent", "message": "TASK006,TASK008,TASK009,TASK031,TASK032"},
        ]
        passed, _ = self._run(None, steps=steps)
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
