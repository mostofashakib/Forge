"""Task Manager verifier specifications built on the shared layered verifier."""

from __future__ import annotations

from typing import Any


def task_manager_seed_shape(expected_counts: dict[str, int]):
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        state = trajectory.get("extra", {}).get("initial_state_snapshot", {})
        details = {k: len(state.get(k, [])) for k in expected_counts}
        passed = details == expected_counts
        return passed, "Expected seeded task manager state shape.", details
    return check


def tm_agent_used_atif():
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        schema_version = trajectory.get("schema_version")
        first_source = trajectory.get("steps", [{}])[0].get("source")
        passed = schema_version == "ATIF-v1.7" and first_source == "user"
        return passed, "Expected ATIF-v1.7 trajectory with user first step.", {
            "schema_version": schema_version,
            "first_source": first_source,
        }
    return check


def tm_rl_determinism_check():
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        reset_check = trajectory.get("extra", {}).get("reset_determinism_check", {})
        passed = bool(reset_check.get("passed", False))
        return passed, "Expected RL environment to be deterministic on reset.", reset_check
    return check


def check_task_has_status(task_id: str, expected_status: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        task = next((t for t in state.get("tasks", []) if t["task_id"] == task_id), None)
        if task is None:
            return False, f"Task {task_id} not found in final state.", {"task_id": task_id}
        actual = task.get("status")
        return actual == expected_status, f"Expected task {task_id} status={expected_status}.", {
            "task_id": task_id, "expected": expected_status, "actual": actual,
        }
    return check


def check_task_deleted(task_id: str, expected: bool = True):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        task = next((t for t in state.get("tasks", []) if t["task_id"] == task_id), None)
        if task is None:
            return False, f"Task {task_id} not found in final state.", {"task_id": task_id}
        actual = bool(task.get("deleted", False))
        return actual == expected, f"Expected task {task_id} deleted={expected}.", {
            "task_id": task_id, "expected": expected, "actual": actual,
        }
    return check


def check_task_milestone(task_id: str, expected_milestone_id: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        task = next((t for t in state.get("tasks", []) if t["task_id"] == task_id), None)
        if task is None:
            return False, f"Task {task_id} not found in final state.", {"task_id": task_id}
        actual = task.get("milestone_id")
        return actual == expected_milestone_id, f"Expected task {task_id} milestone_id={expected_milestone_id}.", {
            "task_id": task_id, "expected": expected_milestone_id, "actual": actual,
        }
    return check


def check_task_milestone_unchanged(task_id: str, disallowed_milestone_id: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        task = next((t for t in state.get("tasks", []) if t["task_id"] == task_id), None)
        if task is None:
            return True, f"Task {task_id} not found — assumed untouched.", {}
        actual = task.get("milestone_id")
        passed = actual != disallowed_milestone_id
        return passed, f"Expected task {task_id} NOT in milestone {disallowed_milestone_id}.", {
            "task_id": task_id, "disallowed": disallowed_milestone_id, "actual": actual,
        }
    return check


def check_task_status_unchanged(task_id: str, expected_status: str):
    """Verify a task's status was not changed (same as initial seed)."""
    return check_task_has_status(task_id, expected_status)


def tm_tool_called(tool_name: str):
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        from fleet.verifiers import trajectory_tool_calls
        calls = [c for c in trajectory_tool_calls(trajectory)
                 if c.get("tool_name") == tool_name or c.get("function_name") == tool_name]
        return bool(calls), f"Expected tool {tool_name} to be called.", {
            "tool_name": tool_name, "count": len(calls),
        }
    return check


def tm_tool_called_n_times(tool_name: str, min_count: int):
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        from fleet.verifiers import trajectory_tool_calls
        calls = [c for c in trajectory_tool_calls(trajectory)
                 if c.get("tool_name") == tool_name or c.get("function_name") == tool_name]
        passed = len(calls) >= min_count
        return passed, f"Expected tool {tool_name} to be called at least {min_count} time(s).", {
            "tool_name": tool_name, "count": len(calls), "min_count": min_count,
        }
    return check


def tm_tool_called_with_args(tool_name: str, **expected_args: Any):
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        from fleet.verifiers import trajectory_tool_calls
        calls = [c for c in trajectory_tool_calls(trajectory)
                 if c.get("tool_name") == tool_name or c.get("function_name") == tool_name]
        for call in calls:
            payload = call.get("input_payload") or call.get("arguments") or {}
            if all(str(payload.get(k, "")).strip().lower() == str(v).strip().lower()
                   for k, v in expected_args.items()):
                return True, f"Found call to {tool_name} with expected arguments.", {"call": call}
        return False, f"Expected call to {tool_name} with {expected_args}.", {
            "tool_name": tool_name,
            "expected": expected_args,
            "calls_found": [c.get("input_payload") or c.get("arguments", {}) for c in calls],
        }
    return check


def check_task_assignee(task_id: str, expected_assignee: str | None):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        task = next((t for t in state.get("tasks", []) if t["task_id"] == task_id), None)
        if task is None:
            return False, f"Task {task_id} not found in final state.", {"task_id": task_id}
        actual = task.get("assignee_id")
        return actual == expected_assignee, f"Expected task {task_id} assignee={expected_assignee!r}.", {
            "task_id": task_id, "expected": expected_assignee, "actual": actual,
        }
    return check


def check_task_has_priority(task_id: str, expected_priority: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        task = next((t for t in state.get("tasks", []) if t["task_id"] == task_id), None)
        if task is None:
            return False, f"Task {task_id} not found in final state.", {"task_id": task_id}
        actual = task.get("priority")
        return actual == expected_priority, f"Expected task {task_id} priority={expected_priority!r}.", {
            "task_id": task_id, "expected": expected_priority, "actual": actual,
        }
    return check


def check_task_has_label(task_id: str, label: str):
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        import json as _json
        task = next((t for t in state.get("tasks", []) if t["task_id"] == task_id), None)
        if task is None:
            return False, f"Task {task_id} not found in final state.", {"task_id": task_id}
        labels = task.get("labels", [])
        if isinstance(labels, str):
            try:
                labels = _json.loads(labels)
            except Exception:
                labels = []
        return label in labels, f"Expected task {task_id} to have label '{label}'.", {
            "task_id": task_id, "label": label, "actual_labels": labels,
        }
    return check


def check_task_mutation_scope(
    expected_task_ids: list[str],
    allowed_fields_by_task: dict[str, set[str]],
):
    """Ensure only the intended tasks and fields changed during the episode."""

    def by_id(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {task["task_id"]: task for task in snapshot.get("tasks", [])}

    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        extra = trajectory.get("extra", {})
        initial_tasks = by_id(extra.get("initial_state_snapshot", {}))
        final_tasks = by_id(extra.get("final_state_snapshot", {}))
        expected_ids = set(expected_task_ids)
        violations: list[dict[str, Any]] = []
        changed_ids: set[str] = set()

        for task_id in sorted(set(initial_tasks) | set(final_tasks)):
            initial = initial_tasks.get(task_id)
            final = final_tasks.get(task_id)
            if initial is None or final is None:
                violations.append({"task_id": task_id, "reason": "task added or removed"})
                continue

            changed_fields = {
                field
                for field in set(initial) | set(final)
                if initial.get(field) != final.get(field)
            }
            if not changed_fields:
                continue

            changed_ids.add(task_id)
            allowed_fields = allowed_fields_by_task.get(task_id, set()) | {"updated_at_ms"}
            disallowed_fields = changed_fields - allowed_fields
            if task_id not in expected_ids:
                violations.append({
                    "task_id": task_id,
                    "reason": "unexpected task mutation",
                    "changed_fields": sorted(changed_fields),
                })
            elif disallowed_fields:
                violations.append({
                    "task_id": task_id,
                    "reason": "unexpected field mutation",
                    "changed_fields": sorted(changed_fields),
                    "disallowed_fields": sorted(disallowed_fields),
                })

        missing_changes = sorted(expected_ids - changed_ids)
        if missing_changes:
            violations.append({"reason": "expected tasks were unchanged", "task_ids": missing_changes})

        passed = not violations
        return passed, "Expected mutations to be limited to the requested tasks and fields.", {
            "changed_task_ids": sorted(changed_ids),
            "violations": violations,
        }

    return check


def tm_final_answer_is_task_ids(expected_task_ids: list[str]):
    """Final answer must be a comma-separated list of exactly the expected task
    IDs. Order-insensitive: the instruction asks for a comma-separated list but
    does not fix an order. Duplicates or extra/missing IDs fail."""
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        actual = str(trajectory.get("extra", {}).get("final_answer", "")).strip()
        if not actual:
            agent_steps = [step for step in trajectory.get("steps", []) if step.get("source") == "agent"]
            actual = str(agent_steps[-1].get("message", "")).strip() if agent_steps else ""
        cleaned = actual.strip().strip("'\"`").strip()
        actual_ids = [part.strip().strip("'\"`").upper() for part in cleaned.split(",") if part.strip()]
        expected = sorted(task_id.upper() for task_id in expected_task_ids)
        passed = sorted(actual_ids) == expected
        return passed, f"Expected final answer to list exactly the task IDs {expected}.", {
            "expected": expected, "actual_answer": actual, "parsed_ids": sorted(actual_ids),
        }
    return check


def tm_tool_not_called(tool_name: str):
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        from fleet.verifiers import trajectory_tool_calls
        calls = [c for c in trajectory_tool_calls(trajectory)
                 if c.get("tool_name") == tool_name or c.get("function_name") == tool_name]
        return not calls, f"Expected tool {tool_name} not to be called.", {
            "tool_name": tool_name, "count": len(calls),
        }
    return check
