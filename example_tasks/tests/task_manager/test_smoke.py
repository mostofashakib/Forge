from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fleet.environments.task_manager.environment import TaskManagerEnvironment
from fleet.verifiers import LayeredVerifier, VerificationSpec, forbidden_tool_not_called, required_tool_called, task_has_status
from tests.simulation_driver import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_AGENT_PROVIDER,
    builtin_task_manager_task,
    legacy_artifacts,
    reset_determinism_check,
    run_task_list,
)
from tests.smoke_utils import SMOKE_TRANSCRIPT_PATH, initialize_smoke_transcript, print_task_report


class TaskManagerSmokeTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        initialize_smoke_transcript()

    def test_task_manager_environment_task_and_verifier_are_functional(self) -> None:
        task = builtin_task_manager_task()
        bundle = run_task_list(
            tasks=[task],
            seed=1,
            output_path="/tmp/fleet_smoke/task_manager.json",
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
                state_checks=[task_has_status("Draft benchmark plan", "IN_PROGRESS")],
                invariant_checks=[status_audit_exists("Draft benchmark plan", "PENDING", "IN_PROGRESS")],
                trajectory_checks=[
                    required_tool_called("update_task"),
                ],
                negative_checks=[forbidden_tool_not_called("create_task")],
            )
        )
        results = verifier.verify(artifacts["final_state_snapshot"], artifacts)
        print_task_report(task, trajectory, results)

        self.assertEqual("task_manager", task.environment_name)
        self.assertIn("Draft benchmark plan", task.instruction)
        self.assertTrue(all(result.passed for result in results), results)
        self.assertTrue(reset_determinism_check(trajectory)["passed"])





class TaskManagerToolSmokeTests(unittest.TestCase):
    """Detailed Task Manager tool calls, determinism and negative scenarios."""

    def setUp(self) -> None:
        self.env = TaskManagerEnvironment(seed=42)

    def test_task_manager_environment_determinism_check(self) -> None:
        initial_state = self.env.export_state()
        initial_state_json = json.dumps(initial_state, sort_keys=True)

        self.env.execute_tool(
            "create_task",
            {
                "task_id": "TASK900",
                "title": "Smoke test task",
                "description": "Created during smoke test.",
                "assignee": "U003",
                "status": "PENDING",
            },
            "U001",
        )
        self.env.execute_tool("update_task", {"task_id": "TASK900", "status": "IN_PROGRESS"}, "U001")

        mutated_state = self.env.export_state()
        self.assertNotEqual(initial_state, mutated_state)

        reset_state = self.env.reset()
        reset_state_json = json.dumps(reset_state, sort_keys=True)

        self.assertEqual(initial_state_json, reset_state_json)

    def test_all_task_manager_tools_functional(self) -> None:
        # 1. list_tasks
        res = self.env.execute_tool("list_tasks", {}, "U001")
        self.assertFalse(res.error)
        self.assertEqual(28, len(res.payload["output"]["tasks"]))

        # 2. get_task
        res = self.env.execute_tool("get_task", {"task_id": "TASK001"}, "U001")
        self.assertFalse(res.error)
        self.assertEqual("Draft benchmark plan", res.payload["output"]["task"]["title"])

        # 3. create_task
        res = self.env.execute_tool(
            "create_task",
            {
                "task_id": "TASK900",
                "title": "Prepare rollout checklist",
                "description": "Verify owners, rollback, and launch gates.",
                "assignee": "U003",
                "status": "PENDING",
            },
            "U001",
        )
        self.assertFalse(res.error)
        self.assertEqual("TASK900", res.payload["output"]["id"])
        self.assertEqual("U003", res.payload["output"]["task"]["assignee_id"])

        # 4. update_task metadata and status
        res = self.env.execute_tool(
            "update_task",
            {
                "task_id": "TASK900",
                "title": "Prepare production rollout checklist",
                "description": "Verify owners, rollback, launch gates, and comms.",
                "assignee": "U002",
                "status": "IN_PROGRESS",
            },
            "U001",
        )
        self.assertFalse(res.error)
        updated_task = res.payload["output"]["task"]
        self.assertEqual("Prepare production rollout checklist", updated_task["title"])
        self.assertEqual("U002", updated_task["assignee_id"])
        self.assertEqual("IN_PROGRESS", updated_task["status"])
        self.assertEqual("task_updated", res.payload["output"]["audit_event"]["event_type"])

        # 5. delete_task
        res = self.env.execute_tool("delete_task", {"task_id": "TASK900"}, "U001")
        self.assertFalse(res.error)
        self.assertEqual("DELETED", res.payload["output"]["task"]["status"])

        # 6. list_tasks excludes deleted by default, includes deleted when requested
        res = self.env.execute_tool("list_tasks", {}, "U001")
        self.assertFalse(res.error)
        self.assertNotIn("TASK900", {task["task_id"] for task in res.payload["output"]["tasks"]})

        res = self.env.execute_tool("list_tasks", {"include_deleted": True}, "U001")
        self.assertFalse(res.error)
        self.assertIn("TASK900", {task["task_id"] for task in res.payload["output"]["tasks"]})

    def test_status_permissions_and_transitions(self) -> None:
        # Assignee can update status on assigned task.
        res = self.env.execute_tool("update_task", {"task_id": "TASK002", "status": "COMPLETED"}, "U003")
        self.assertFalse(res.error)
        self.assertEqual("COMPLETED", res.payload["output"]["task"]["status"])

        # Non-creator non-admin cannot update task metadata.
        res = self.env.execute_tool("update_task", {"task_id": "TASK001", "title": "Unauthorized rename"}, "U003")
        self.assertTrue(res.error)
        self.assertEqual("permission_denied", res.error.error_code)

        # Creator can update metadata.
        res = self.env.execute_tool("update_task", {"task_id": "TASK002", "description": "Creator update"}, "U002")
        self.assertFalse(res.error)
        self.assertEqual("Creator update", res.payload["output"]["task"]["description"])

    def test_task_manager_negative_cases(self) -> None:
        # 1. Duplicate task id
        res = self.env.execute_tool(
            "create_task",
            {"task_id": "TASK001", "title": "Duplicate", "description": ""},
            "U001",
        )
        self.assertTrue(res.error)
        self.assertEqual("duplicate_task", res.error.error_code)

        # 2. Missing task
        res = self.env.execute_tool("get_task", {"task_id": "TASK404"}, "U001")
        self.assertTrue(res.error)
        self.assertEqual("task_not_found", res.error.error_code)

        # 3. Invalid status transition: PENDING -> COMPLETED is not allowed.
        res = self.env.execute_tool("update_task", {"task_id": "TASK001", "status": "COMPLETED"}, "U001")
        self.assertTrue(res.error)
        self.assertEqual("invalid_status_transition", res.error.error_code)

        # 4. Unknown status
        res = self.env.execute_tool("update_task", {"task_id": "TASK001", "status": "SHIPPED"}, "U001")
        self.assertTrue(res.error)
        self.assertEqual("invalid_arguments", res.error.error_code)

        # 5. Non-creator non-admin cannot delete.
        res = self.env.execute_tool("delete_task", {"task_id": "TASK001"}, "U002")
        self.assertTrue(res.error)
        self.assertEqual("permission_denied", res.error.error_code)

        # 6. Empty title is invalid.
        res = self.env.execute_tool("create_task", {"title": "", "description": "No title"}, "U001")
        self.assertTrue(res.error)
        self.assertEqual("invalid_arguments", res.error.error_code)





def status_audit_exists(title: str, before_status: str, after_status: str):
    def check(trajectory):
        final_state = trajectory["final_state_snapshot"]
        task_id = next(
            task["task_id"] for task in final_state["tasks"] if task["title"] == title
        )
        matches = [
            event
            for event in final_state["audit_events"]
            if event["task_id"] == task_id
            and event["event_type"] == "status_changed"
            and json.loads(event["before_json"]).get("status") == before_status
            and json.loads(event["after_json"]).get("status") == after_status
        ]
        return (
            bool(matches),
            "Expected deterministic status-change audit event.",
            {"matches": len(matches), "task_id": task_id},
        )

    return check




if __name__ == "__main__":
    unittest.main(verbosity=2)
