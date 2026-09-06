#!/usr/bin/env python3
"""The engine is generic: it is driven here by scenarios it has never seen."""

from __future__ import annotations

import sys

from task_sim.models import ScenarioEvent, ScenarioRule
from task_sim.scenario import Scenario
from task_sim.service import seed_database
from dynamic_scenario import DISCOVERY_SCENARIO, EMPTY_SCENARIO, ORDERED_SCENARIO
from workspace import Workspace

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def test_an_empty_scenario_is_a_plain_tracker() -> None:
    with Workspace(EMPTY_SCENARIO) as workspace:
        workspace.call("update_task", task_id="TASK006", assignee="U002")
        check("no events, no activations", workspace.events() == {})
        check("the tools still work", workspace.task("TASK006")["assignee_id"] == "U002")


def test_observation_fires_on_what_the_result_carried() -> None:
    with Workspace(DISCOVERY_SCENARIO) as workspace:
        workspace.call("list_tasks", project_id="P001")
        check("an unrelated read does not fire it", "audit_opened" not in workspace.activated())
        workspace.call("get_task", task_id="TASK020")
        check("seeing the id fires it", "audit_opened" in workspace.activated())


def test_a_latent_task_is_absent_until_its_event() -> None:
    with Workspace(DISCOVERY_SCENARIO) as workspace:
        check("latent work is invisible at T0", "TASK900" not in workspace.visible_task_ids())
        check("it is invisible to a filtered list too",
              "TASK900" not in {t["task_id"] for t in workspace.call("list_tasks", project_id="P003")["tasks"]})
        check("and to the project rollup",
              "TASK900" not in {t["task_id"] for t in workspace.call("get_project", project_id="P003")["tasks"]})

        workspace.call("get_task", task_id="TASK020")
        workspace.call("update_task", task_id="TASK020", status="IN_PROGRESS")

        check("firing the event publishes it", "TASK900" in workspace.visible_task_ids())
        check("its dependency arrives with it",
              any(d["dep_id"] == "DEP900" for d in workspace.state()["dependencies"]))
        released = workspace.task("TASK900")
        check("it arrives with its authored fields",
              released["assignee_id"] == "U003" and released["priority"] == "URGENT",
              str(released))


def test_the_call_that_fires_a_rule_does_not_see_what_it_released() -> None:
    with Workspace(DISCOVERY_SCENARIO) as workspace:
        workspace.call("get_task", task_id="TASK020")
        result = workspace.call("update_task", task_id="TASK020", status="IN_PROGRESS")
        # Transitions are evaluated after the call has produced its result, so
        # the call that satisfied a rule never returns the rows it released.
        check("the releasing call's own result is unchanged",
              "TASK900" not in str(result), "the released task leaked into the trigger result")


def test_a_closure_resolves_in_the_same_call() -> None:
    with Workspace(DISCOVERY_SCENARIO) as workspace:
        workspace.call("get_task", task_id="TASK020")
        check("the closure waits for its last prerequisite",
              "triage_ready" not in workspace.activated())
        workspace.call("update_task", task_id="TASK020", status="IN_PROGRESS")
        # Evaluation runs to a fixed point, so the closure does not need an
        # unrelated action to come along and nudge it.
        check("it resolves in the call that completed the last one",
              "triage_ready" in workspace.activated())


def test_requires_pending_locks_out_the_wrong_order() -> None:
    with Workspace(ORDERED_SCENARIO) as workspace:
        workspace.call("update_task", task_id="TASK017", labels=["infra", "signed-off"])
        check("early sign-off fires while the deploy is pending",
              "signed_off_early" in workspace.activated())

    with Workspace(ORDERED_SCENARIO) as workspace:
        workspace.call("update_task", task_id="TASK016", status="IN_PROGRESS")
        workspace.call("update_task", task_id="TASK016", status="COMPLETED")
        check("the deploy fired", "shipped" in workspace.activated())
        workspace.call("update_task", task_id="TASK017", labels=["infra", "signed-off"])
        check("the same label after the deploy does not fire it",
              "signed_off_early" not in workspace.activated())


def test_label_rules_require_every_label() -> None:
    scenario = Scenario(
        events=(ScenarioEvent("both_labels"),),
        rules=(ScenarioRule("l10", "both_labels", "label_present",
                            row_id="TASK022", labels=("mobile", "urgent")),),
        observation_only=("both_labels",),
    )
    with Workspace(scenario) as workspace:
        workspace.call("update_task", task_id="TASK022", labels=["mobile"])
        check("one of two is not enough", "both_labels" not in workspace.activated())
        workspace.call("update_task", task_id="TASK022", labels=["mobile", "urgent"])
        check("both fires it", "both_labels" in workspace.activated())


def test_an_event_fires_at_most_once() -> None:
    with Workspace(DISCOVERY_SCENARIO) as workspace:
        workspace.call("get_task", task_id="TASK020")
        first = workspace.state()
        stamp = [e for e in first["scenario_events"] if e["event_id"] == "audit_opened"][0]
        workspace.call("get_task", task_id="TASK020")
        second = workspace.state()
        again = [e for e in second["scenario_events"] if e["event_id"] == "audit_opened"][0]
        check("the second look does not re-fire it",
              stamp["activated_ms"] == again["activated_ms"])
        check("and does not republish its rows",
              len(first["tasks"]) == len(second["tasks"]))


def test_a_malformed_scenario_is_refused_at_seed_time() -> None:
    unknown_trigger = Scenario(
        events=(ScenarioEvent("e"),),
        rules=(ScenarioRule("r", "e", "telepathy"),),
        observation_only=("e",),
    )
    with Workspace() as scratch:
        try:
            seed_database(scratch.db_path, scratch.snapshot_path, unknown_trigger)
            refused = False
        except ValueError:
            refused = True
    check("an unknown trigger fails at seed, not silently at runtime", refused)

    dangling = False
    try:
        Scenario(events=(), rules=(ScenarioRule("r", "nope", "observed", observed_ids=("X",)),))
    except ValueError:
        dangling = True
    check("a rule naming an undeclared event is refused", dangling)


def main() -> int:
    print(__doc__)
    for test in (
        test_an_empty_scenario_is_a_plain_tracker,
        test_observation_fires_on_what_the_result_carried,
        test_a_latent_task_is_absent_until_its_event,
        test_the_call_that_fires_a_rule_does_not_see_what_it_released,
        test_a_closure_resolves_in_the_same_call,
        test_requires_pending_locks_out_the_wrong_order,
        test_label_rules_require_every_label,
        test_an_event_fires_at_most_once,
        test_a_malformed_scenario_is_refused_at_seed_time,
    ):
        print(f"\n{test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all scenario-engine checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
