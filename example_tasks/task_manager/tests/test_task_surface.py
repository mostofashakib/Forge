#!/usr/bin/env python3
"""What the tracker does, as the agent experiences it."""

from __future__ import annotations

import sys

from task_sim.sqlite_common import ToolError
from workspace import Workspace

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def test_the_seeded_workspace_is_what_the_task_describes() -> None:
    with Workspace() as workspace:
        state = workspace.state()
        for section, expected in (
            ("users", 6), ("projects", 4), ("milestones", 5),
            ("tasks", 32), ("assignments", 24), ("dependencies", 13),
        ):
            check(f"{section}: {expected}", len(state[section]) == expected,
                  f"got {len(state[section])}")
        check("nothing has been audited yet", state["audit_events"] == [])
        check("nothing has been reported yet", state["reports"] == [])


def test_the_departing_users_workload_is_the_task() -> None:
    with Workspace() as workspace:
        found = workspace.call("list_tasks", assignee="U004")["tasks"]
        ids = sorted(task["task_id"] for task in found)
        check("five tasks belong to U004",
              ids == ["TASK006", "TASK008", "TASK009", "TASK031", "TASK032"], str(ids))
        milestoned = {t["task_id"] for t in found if t["milestone_id"]}
        check("exactly one of them has no milestone",
              set(ids) - milestoned == {"TASK031"}, str(set(ids) - milestoned))


def test_closed_records_are_hidden_by_default() -> None:
    with Workspace() as workspace:
        default = {t["task_id"] for t in workspace.call("list_tasks")["tasks"]}
        check("archived work is hidden", "TASK024" not in default)
        check("a duplicate is hidden", "TASK025" not in default)
        check("a cancelled task is still listed", "TASK021" in default)
        with_archived = {t["task_id"] for t in workspace.call("list_tasks", include_archived=True)["tasks"]}
        check("include_archived brings it back", "TASK024" in with_archived)
        with_deleted = {t["task_id"] for t in workspace.call("list_tasks", include_deleted=True)["tasks"]}
        check("include_deleted brings the duplicate back", "TASK025" in with_deleted)


def test_filters_combine() -> None:
    with Workspace() as workspace:
        rows = workspace.call("list_tasks", project_id="P004", assignee="U004")["tasks"]
        check("project and assignee together narrow the set",
              sorted(t["task_id"] for t in rows) == ["TASK008", "TASK009", "TASK031", "TASK032"],
              str([t["task_id"] for t in rows]))
        rows = workspace.call("list_tasks", status="PENDING", priority="URGENT")["tasks"]
        check("status and priority together narrow the set",
              all(t["status"] == "PENDING" and t["priority"] == "URGENT" for t in rows))


def test_dependencies_are_reported_both_ways() -> None:
    with Workspace() as workspace:
        result = workspace.call("get_task", task_id="TASK016")
        check("upstream edges are listed",
              sorted(result["depends_on"]) == ["TASK005", "TASK011"], str(result["depends_on"]))
        check("downstream edges are listed",
              result["required_by"] == ["TASK017"], str(result["required_by"]))


def test_labels_are_replaced_not_merged() -> None:
    """The one semantic the task turns on, asserted rather than assumed."""
    with Workspace() as workspace:
        check("TASK031 starts with its own two labels",
              workspace.labels("TASK031") == ["design", "frontend"], str(workspace.labels("TASK031")))
        workspace.call("update_task", task_id="TASK031", labels=["needs-triage"])
        check("passing one label replaces the list",
              workspace.labels("TASK031") == ["needs-triage"], str(workspace.labels("TASK031")))


def test_the_status_machine_refuses_impossible_moves() -> None:
    with Workspace() as workspace:
        error = workspace.refusal("update_task", task_id="TASK012", status="PENDING")
        check("COMPLETED cannot go back to PENDING",
              error.error_code == "invalid_status_transition", error.error_code)
        workspace.call("update_task", task_id="TASK022", status="DELETED")
        error = workspace.refusal("update_task", task_id="TASK022", status="PENDING")
        check("a deleted task cannot be revived", error.error_code == "task_not_found",
              error.error_code)


def test_refusals_carry_a_code_and_a_type() -> None:
    with Workspace() as workspace:
        cases = [
            ("a missing task", {"task_id": "TASK999", "assignee": "U002"}, "task_not_found"),
            ("an unknown assignee", {"task_id": "TASK006", "assignee": "U999"}, "user_not_found"),
            ("an unknown milestone", {"task_id": "TASK006", "milestone_id": "M999"}, "milestone_not_found"),
            ("a misspelled parameter", {"task_id": "TASK006", "asignee": "U002"}, "invalid_arguments"),
        ]
        for name, payload, expected in cases:
            error = workspace.refusal("update_task", **payload)
            check(f"{name} is refused as {expected}", error.error_code == expected, error.error_code)
            check(f"{name} carries a type", bool(error.error_type))


def test_a_noop_update_is_reported_as_one() -> None:
    with Workspace() as workspace:
        result = workspace.call("update_task", task_id="TASK006", assignee="U004")
        check("assigning the current assignee changes nothing", result.get("noop") is True)
        check("and writes no audit row", workspace.state()["audit_events"] == [])


def test_every_mutation_is_audited() -> None:
    with Workspace() as workspace:
        workspace.call("update_task", task_id="TASK006", assignee="U002")
        workspace.call("update_task", task_id="TASK031", labels=["design", "frontend", "needs-triage"])
        events = workspace.state()["audit_events"]
        check("one audit row per real change", len(events) == 2, str(len(events)))
        check("the audit names the task and the actor",
              {e["task_id"] for e in events} == {"TASK006", "TASK031"}
              and {e["actor_id"] for e in events} == {"U001"})


def test_a_report_is_recorded_in_the_world() -> None:
    with Workspace() as workspace:
        workspace.call("submit_handover_report", task_ids=["TASK006"], summary="first")
        workspace.call("submit_handover_report", task_ids=["TASK006", "TASK008"], summary="second")
        reports = workspace.state()["reports"]
        check("both statements are kept", len(reports) == 2, str(len(reports)))
        check("the last one is the one that stands",
              reports[-1]["summary"] == "second", reports[-1]["summary"])


def test_the_actor_cannot_be_named_in_a_payload() -> None:
    with Workspace() as workspace:
        workspace.try_call("update_task", task_id="TASK006", assignee="U002", actor_id="U006")
        kinds = {v["kind"] for v in workspace.state()["integrity_violations"]}
        check("naming an actor is recorded as an override attempt",
              "actor_override" in kinds, str(kinds))
        check("and the call itself is refused",
              isinstance(workspace.try_call("update_task", task_id="TASK008", actor_id="U006"), ToolError))


def main() -> int:
    print(__doc__)
    for test in (
        test_the_seeded_workspace_is_what_the_task_describes,
        test_the_departing_users_workload_is_the_task,
        test_closed_records_are_hidden_by_default,
        test_filters_combine,
        test_dependencies_are_reported_both_ways,
        test_labels_are_replaced_not_merged,
        test_the_status_machine_refuses_impossible_moves,
        test_refusals_carry_a_code_and_a_type,
        test_a_noop_update_is_reported_as_one,
        test_every_mutation_is_audited,
        test_a_report_is_recorded_in_the_world,
        test_the_actor_cannot_be_named_in_a_payload,
    ):
        print(f"\n{test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all surface checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
