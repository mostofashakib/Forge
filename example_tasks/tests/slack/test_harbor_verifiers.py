from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rewardkit as rk

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fleet.agents.action_parsing import normalize_ollama_action
from fleet.environments.slack.schema import SLACK_TOOL_SCHEMA
from fleet.environments.slack.sqlite_service import (
    add_reaction,
    change_group_name,
    change_user_display_name,
    create_group,
    export_state,
    post_message,
    reply_to_thread,
    seed_database,
    send_dm_message,
    send_group_message,
    update_message,
)
from tests.slack.isolated_harbor_checks import (
    SLACK_TASK_1_SPEC,
    SLACK_TASK_2_SPEC,
    slack_task_1_test_reward_checks,
    slack_task_1_test_verifier,
    slack_task_2_test_verifier,
)
from tests.verify_task import main as verify_task_main


class HarborSlackVerifierTests(unittest.TestCase):
    def test_external_agent_accepts_harbor_action_shape(self) -> None:
        tool_names = {tool["name"] for tool in SLACK_TOOL_SCHEMA}

        action = normalize_ollama_action(
            {
                "thought": "I should search first.",
                "tool_name": "search_messages",
                "input_payload": {"query": "alice incidents"},
            },
            tool_names,
        )

        self.assertEqual(
            {"type": "tool_call", "tool_name": "search_messages", "input": {"query": "alice incidents"}},
            action,
        )
        self.assertEqual(
            {"type": "final_answer", "answer": "#platform-outages"},
            normalize_ollama_action({"thought": "Done.", "final_answer": "#platform-outages"}, tool_names),
        )

    def test_new_slack_operation_tasks_score_all_checks(self) -> None:
        with self.subTest(verifier="root_slack_task_1_test_verifier"), tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self._build_dm_thread_trajectory(Path(temp_dir), SLACK_TASK_1_SPEC)
            results = slack_task_1_test_verifier().verify(trajectory["extra"]["final_state_snapshot"], trajectory)
            self.assertTrue(all(result.passed for result in results), [result.message for result in results if not result.passed])
            self.assertEqual(len(results), sum(result.score for result in results))

        with self.subTest(verifier="root_slack_task_2_test_verifier"), tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self._build_operations_trajectory(Path(temp_dir), SLACK_TASK_2_SPEC, include_reaction=True)
            results = slack_task_2_test_verifier().verify(trajectory["extra"]["final_state_snapshot"], trajectory)
            self.assertTrue(all(result.passed for result in results), [result.message for result in results if not result.passed])
            self.assertEqual(len(results), sum(result.score for result in results))

    def _assert_task_1_fails(self, trajectory: dict, expected_fragment: str) -> None:
        results = slack_task_1_test_verifier().verify(trajectory["extra"]["final_state_snapshot"], trajectory)
        failed = [result.message for result in results if not result.passed]
        self.assertTrue(failed, "Expected at least one failed check.")
        self.assertTrue(
            any(expected_fragment in message for message in failed),
            f"Expected a failure mentioning {expected_fragment!r}, got: {failed}",
        )

    def test_verifier_fails_on_incorrect_shared_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = {**SLACK_TASK_1_SPEC, "shared_body": "Hack campaign sync"}
            trajectory = self._build_dm_thread_trajectory(Path(temp_dir), spec)
            trajectory["extra"]["final_state_snapshot"] = trajectory["extra"]["final_state_snapshot"]
        results = slack_task_1_test_verifier().verify(trajectory["extra"]["final_state_snapshot"], trajectory)
        failed = [r.message for r in results if not r.passed]
        self.assertTrue(any("Incident review starts at noon." in msg or "send_dm_message" in msg for msg in failed), failed)

    def test_verifier_fails_when_group_message_precedes_dm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self._build_dm_thread_trajectory(Path(temp_dir), SLACK_TASK_1_SPEC, dm_first=False)
        self._assert_task_1_fails(trajectory, "before")

    def test_verifier_fails_when_wrong_own_message_edited(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = {**SLACK_TASK_1_SPEC, "edit_message_id": "MSG001"}
            trajectory = self._build_dm_thread_trajectory(Path(temp_dir), spec)
        self._assert_task_1_fails(trajectory, "MSG023")

    def test_verifier_fails_when_welcome_names_wrong_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = {
                **SLACK_TASK_1_SPEC,
                "welcome_replies": {"MSG004": "Welcome Alice Nguyen", "MSG010": "Welcome Cara Singh"},
            }
            trajectory = self._build_dm_thread_trajectory(Path(temp_dir), spec)
        self._assert_task_1_fails(trajectory, "Welcome Ben Ortíz")

    def test_verifier_fails_when_welcome_reply_under_wrong_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = {
                **SLACK_TASK_1_SPEC,
                "welcome_replies": {"MSG004": "Welcome Alice Nguyen", "MSG019": "Welcome Ben Ortíz"},
            }
            trajectory = self._build_dm_thread_trajectory(Path(temp_dir), spec)
        self._assert_task_1_fails(trajectory, "only under")

    def test_verifier_fails_when_edit_timestamp_did_not_change(self) -> None:
        # Final body looks right but the edit timestamp still matches the
        # initial snapshot: the message was never actually edited in-run.
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self._build_dm_thread_trajectory(Path(temp_dir), SLACK_TASK_1_SPEC)
        final = trajectory["extra"]["final_state_snapshot"]
        initial = trajectory["extra"]["initial_state_snapshot"]
        initial_msg = next(m for m in initial["messages"] if m["message_id"] == "MSG023")
        for message in final["messages"]:
            if message["message_id"] == "MSG023":
                message["edited_at_ms"] = initial_msg["edited_at_ms"]
        self._assert_task_1_fails(trajectory, "edit timestamp")

    def test_verifier_fails_when_neighbor_reply_was_edited(self) -> None:
        # The decoy user's adjacent reply was edited (as its own author);
        # the timestamp comparison against the initial snapshot must catch it
        # even though the body is later restored.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trajectory = self._build_dm_thread_trajectory(root, SLACK_TASK_1_SPEC)
            db_path = root / "slack.db"
            update_message(db_path, "MSG024", "tampered", "U006")
            update_message(db_path, "MSG024", "Support is standing by.", "U006")
            trajectory["extra"]["final_state_snapshot"] = export_state(db_path)
        self._assert_task_1_fails(trajectory, "unmodified")

    def test_verifier_fails_when_answer_drops_the_accent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = {**SLACK_TASK_1_SPEC, "final_answer": "Alice Nguyen;Ben Ortiz"}
            trajectory = self._build_dm_thread_trajectory(Path(temp_dir), spec)
        self._assert_task_1_fails(trajectory, "exactly")

    def test_verify_task_writes_zero_reward_without_failing_when_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trajectory = self._build_dm_thread_trajectory(temp_path, SLACK_TASK_1_SPEC, skip_edit=True)
            trajectory_path = temp_path / "trajectory.json"
            reward_path = temp_path / "reward.txt"
            report_path = temp_path / "report.json"
            trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")

            with patch.object(
                sys,
                "argv",
                [
                    "verify_task",
                    "--spec",
                    "tests.slack.isolated_harbor_checks:slack_task_1_test_verifier",
                    "--trajectory",
                    str(trajectory_path),
                    "--reward",
                    str(reward_path),
                    "--report",
                    str(report_path),
                ],
            ):
                verify_task_main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            reward = float(reward_path.read_text(encoding="utf-8"))

        self.assertFalse(report["passed"])
        self.assertEqual(0.0, reward)
        self.assertEqual(0, report["reward"])
        self.assertEqual(0, report["score"])
        self.assertEqual(1, report["max_score"])

    def test_verify_task_writes_zero_reward_without_failing_on_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trajectory_path = temp_path / "missing.json"
            reward_path = temp_path / "reward.txt"
            report_path = temp_path / "report.json"

            with patch.object(
                sys,
                "argv",
                [
                    "verify_task",
                    "--spec",
                    "tests.slack.isolated_harbor_checks:slack_task_1_test_verifier",
                    "--trajectory",
                    str(trajectory_path),
                    "--reward",
                    str(reward_path),
                    "--report",
                    str(report_path),
                ],
            ):
                verify_task_main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            reward = float(reward_path.read_text(encoding="utf-8"))

        self.assertFalse(report["passed"])
        self.assertEqual(0.0, reward)
        self.assertEqual("verifier_exception", report["results"][0]["layer"])
        self.assertIn("exception", report)

    def test_rewardkit_writes_reward_json_for_shared_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trajectory = self._build_dm_thread_trajectory(temp_path, SLACK_TASK_1_SPEC)
            trajectory_path = temp_path / "trajectory.json"
            report_path = temp_path / "report.json"
            reward_path = temp_path / "reward.json"
            tests_path = temp_path / "tests"
            tests_path.mkdir()
            trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
            (tests_path / "check.py").write_text(
                "\n".join(
                    [
                        "import rewardkit as rk",
                        "from fleet.verifiers.rewardkit_checks import register_harbor_verifier",
                        "from tests.slack.isolated_harbor_checks import slack_task_1_test_reward_checks",
                        "register_harbor_verifier(",
                        "    'tests.slack.isolated_harbor_checks:slack_task_1_test_verifier',",
                        "    'root_slack_task_1',",
                        "    slack_task_1_test_reward_checks(),",
                        f"    trajectory_path={str(trajectory_path)!r},",
                        f"    report_path={str(report_path)!r},",
                        ")",
                    ]
                ),
                encoding="utf-8",
            )

            sys.modules.pop("check", None)
            rewards = rk.run(tests_path, workspace=temp_path, output=reward_path)
            reward_json = json.loads(reward_path.read_text(encoding="utf-8"))
            reward_details = json.loads(reward_path.with_name("reward-details.json").read_text(encoding="utf-8"))
            report_json = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual({"reward": 1.0}, rewards)
        self.assertEqual({"reward": 1.0}, reward_json)
        self.assertEqual(1 + len(slack_task_1_test_reward_checks()), len(reward_details["reward"]["criteria"]))
        self.assertTrue(any(item["weight"] == 0 for item in reward_details["reward"]["criteria"]))
        self.assertTrue(report_json["passed"])

    def test_rewardkit_keeps_binary_reward_with_named_failed_check(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trajectory = self._build_dm_thread_trajectory(temp_path, SLACK_TASK_1_SPEC, skip_edit=True)
            trajectory_path = temp_path / "trajectory.json"
            report_path = temp_path / "report.json"
            reward_path = temp_path / "reward.json"
            tests_path = temp_path / "tests"
            tests_path.mkdir()
            trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
            (tests_path / "check.py").write_text(
                "\n".join(
                    [
                        "import rewardkit as rk",
                        "from fleet.verifiers.rewardkit_checks import register_harbor_verifier",
                        "from tests.slack.isolated_harbor_checks import slack_task_1_test_reward_checks",
                        "register_harbor_verifier(",
                        "    'tests.slack.isolated_harbor_checks:slack_task_1_test_verifier',",
                        "    'root_slack_task_1',",
                        "    slack_task_1_test_reward_checks(),",
                        f"    trajectory_path={str(trajectory_path)!r},",
                        f"    report_path={str(report_path)!r},",
                        ")",
                    ]
                ),
                encoding="utf-8",
            )

            sys.modules.pop("check", None)
            rewards = rk.run(tests_path, workspace=temp_path, output=reward_path)
            reward_details = json.loads(reward_path.with_name("reward-details.json").read_text(encoding="utf-8"))

        self.assertEqual({"reward": 0.0}, rewards)
        criteria = reward_details["reward"]["criteria"]
        failed = [item for item in criteria if item["value"] == 0.0]
        self.assertTrue(any("own thread reply edited" in item["description"] for item in failed))
        self.assertEqual(0.0, criteria[0]["value"])
        self.assertEqual(1.0, criteria[0]["weight"])

    def test_verifier_reads_final_state_from_workspace_database(self) -> None:
        # SQLite is the source of truth: when the workspace database is
        # available, the verdict comes from it, so a forged
        # final_state_snapshot in the trajectory cannot change the result.
        from fleet.verifiers.rewardkit_checks import evaluate_verifier

        spec = "tests.slack.isolated_harbor_checks:slack_task_1_test_verifier"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trajectory = self._build_dm_thread_trajectory(root, SLACK_TASK_1_SPEC)
            trajectory["extra"]["final_state_snapshot"] = trajectory["extra"]["initial_state_snapshot"]
            trajectory_path = root / "trajectory.json"
            trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")

            snapshot_only = evaluate_verifier(trajectory_path, spec)
            db_direct = evaluate_verifier(trajectory_path, spec, root)

        self.assertFalse(snapshot_only["passed"])
        self.assertTrue(db_direct["passed"])

    def _build_dm_thread_trajectory(
        self,
        root: Path,
        task_spec: dict,
        dm_first: bool = True,
        skip_edit: bool = False,
    ) -> dict:
        """Golden run for the DM/group/thread task shape (slack_task_1 v3)."""
        db_path = root / "slack.db"
        snapshot_path = root / "slack_seed_snapshot.sql"
        seed_database(db_path, snapshot_path)
        initial_state = export_state(db_path)

        steps = [{"step_id": 1, "source": "user", "message": "task"}]
        step_id = 2

        sends = [
            ("send_dm_message",
             lambda: send_dm_message(db_path, None, task_spec["dm_chat_id"], task_spec["shared_body"], "U002"),
             {"chat_id": task_spec["dm_chat_id"], "body": task_spec["shared_body"]}),
            ("send_group_message",
             lambda: send_group_message(db_path, task_spec["group_chat_id"], task_spec["shared_body"], "U002"),
             {"group_id": task_spec["group_chat_id"], "body": task_spec["shared_body"]}),
        ]
        if not dm_first:
            sends.reverse()
        for tool_name, perform, arguments in sends:
            perform()
            steps.append(self._tool_step(step_id, tool_name, arguments))
            step_id += 1

        if not skip_edit:
            update_message(db_path, task_spec["edit_message_id"], task_spec["edit_body"], "U002")
            steps.append(
                self._tool_step(step_id, "update_message", {"message_id": task_spec["edit_message_id"], "body": task_spec["edit_body"]})
            )
            step_id += 1

        for parent_id, body in task_spec["welcome_replies"].items():
            reply_to_thread(db_path, parent_id, body, "U002")
            steps.append(
                self._tool_step(step_id, "slack.reply_to_thread", {"thread_parent_id": parent_id, "body": body})
            )
            step_id += 1

        steps.append({"step_id": step_id, "source": "agent", "message": task_spec["final_answer"]})

        return {
            "schema_version": "ATIF-v1.7",
            "steps": steps,
            "extra": {
                "initial_state_snapshot": initial_state,
                "final_state_snapshot": export_state(db_path),
                "final_answer": task_spec["final_answer"],
                "reset_determinism_check": {"passed": True},
            },
        }

    def _build_operations_trajectory(
        self,
        root: Path,
        task_spec: dict,
        include_reaction: bool,
    ) -> dict:
        db_path = root / "slack.db"
        snapshot_path = root / "slack_seed_snapshot.sql"
        seed_database(db_path, snapshot_path)
        initial_state = export_state(db_path)

        steps = [{"step_id": 1, "source": "user", "message": "task"}]
        step_id = 2

        if "new_display_name" in task_spec:
            change_user_display_name(db_path, "U002", task_spec["new_display_name"], "U002")
            steps.append(
                self._tool_step(step_id, "change_user_display_name", {"user_id": "U002", "new_display_name": task_spec["new_display_name"]})
            )
            step_id += 1

        post_message(db_path, task_spec["target_channel_id"], task_spec["target_body"], "U002")
        steps.append(
            self._tool_step(step_id, "post_message", {"channel_id": task_spec["target_channel_id"], "body": task_spec["target_body"]})
        )
        step_id += 1

        reply = reply_to_thread(db_path, task_spec["parent_id"], task_spec["reply_body"], "U002")
        steps.append(
            self._tool_step(step_id, "slack.reply_to_thread", {"thread_parent_id": task_spec["parent_id"], "body": task_spec["reply_body"]})
        )
        step_id += 1

        if include_reaction:
            add_reaction(db_path, reply["id"], "heart", "U002")
            steps.append(
                self._tool_step(step_id, "add_reaction", {"message_id": reply["id"], "emoji": "heart"})
            )
            step_id += 1

        group = create_group(db_path, task_spec["group_name"], task_spec["participants"], "U002")
        steps.append(
            self._tool_step(step_id, "create_group", {"name": task_spec["group_name"], "participants": task_spec["participants"]})
        )
        step_id += 1

        group_actions = [
            ("send_group_message", lambda: send_group_message(db_path, group["id"], task_spec["welcome_body"], "U002"),
             {"group_id": group["id"], "body": task_spec["welcome_body"]}),
            ("change_group_name", lambda: change_group_name(db_path, group["id"], task_spec["final_group_name"], "U002"),
             {"group_id": group["id"], "new_name": task_spec["final_group_name"]}),
        ]
        for tool_name, perform, arguments in group_actions:
            perform()
            steps.append(self._tool_step(step_id, tool_name, arguments))
            step_id += 1

        final_answer = group["id"]
        if "referenced_channel_name" in task_spec:
            final_answer = f"{group['id']}:{task_spec['referenced_channel_name']}"
        steps.append({"step_id": step_id, "source": "agent", "message": final_answer})

        return {
            "schema_version": "ATIF-v1.7",
            "steps": steps,
            "extra": {
                "initial_state_snapshot": initial_state,
                "final_state_snapshot": export_state(db_path),
                "final_answer": final_answer,
                "reset_determinism_check": {"passed": True},
            },
        }

    @staticmethod
    def _tool_step(step: int, tool_name: str, arguments: dict) -> dict:
        return {
            "step_id": step,
            "source": "agent",
            "message": f"Calling {tool_name}.",
            "tool_calls": [{"tool_call_id": f"call_{step}", "function_name": tool_name, "arguments": arguments}],
        }


if __name__ == "__main__":
    unittest.main()
