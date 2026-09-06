#!/usr/bin/env python3
"""The clock is a pure function of world mutations, not of wall time."""

from __future__ import annotations

import sqlite3
import sys

from task_sim.clock import START_MS, STEP_MS, VIRTUAL_CLOCK, moment
from task_sim.service import execute_tool
from task_sim.sqlite_common import connect
from workspace import ORACLE_STEPS, Workspace, run_steps

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def clock(workspace: Workspace) -> int:
    with connect(workspace.db_path) as connection:
        return VIRTUAL_CLOCK.now(connection)


def test_reads_do_not_move_time() -> None:
    with Workspace() as workspace:
        before = clock(workspace)
        workspace.call("list_tasks")
        workspace.call("get_task", task_id="TASK006")
        workspace.call("list_projects")
        workspace.call("get_project", project_id="P004")
        check("reads leave the clock alone", clock(workspace) == before,
              f"{before} -> {clock(workspace)}")


def test_a_mutation_advances_once() -> None:
    with Workspace() as workspace:
        before = clock(workspace)
        workspace.call("update_task", task_id="TASK006", assignee="U002")
        after = clock(workspace)
        # One action is one instant however many rows it touched: the update
        # writes the task, an assignment row and an audit row.
        check("one mutation is one step", after == before + STEP_MS, f"{before} -> {after}")


def test_a_refused_call_rolls_time_back() -> None:
    with Workspace() as workspace:
        before = clock(workspace)
        workspace.refusal("update_task", task_id="NOPE", assignee="U002")
        check("a refused call leaves no trace in time", clock(workspace) == before,
              f"{before} -> {clock(workspace)}")


def test_the_clock_never_runs_backwards() -> None:
    with Workspace() as workspace:
        with connect(workspace.db_path) as connection:
            try:
                connection.execute("UPDATE virtual_clock SET current_ms = 1 WHERE clock_id = 1")
                connection.commit()
                rejected = False
            except sqlite3.IntegrityError:
                rejected = True
            except sqlite3.OperationalError:
                rejected = True
        check("a rewind is rejected by the database itself", rejected,
              "the monotonic trigger did not fire")


def test_the_same_episode_writes_the_same_timestamps() -> None:
    stamps = []
    for _ in range(2):
        with Workspace() as workspace:
            run_steps(workspace, ORACLE_STEPS)
            state = workspace.state()
            stamps.append([
                (task["task_id"], task["updated_at_ms"]) for task in state["tasks"]
            ])
    check("two identical episodes write identical timestamps", stamps[0] == stamps[1],
          "timestamps diverged between runs")


def test_browsing_first_does_not_perturb_later_writes() -> None:
    """An agent that reads more before acting still writes the same instants."""
    plain, browsed = None, None
    with Workspace() as workspace:
        workspace.call("update_task", task_id="TASK006", assignee="U002")
        plain = workspace.task("TASK006")["updated_at_ms"]
    with Workspace() as workspace:
        for _ in range(20):
            workspace.call("list_tasks")
        workspace.call("update_task", task_id="TASK006", assignee="U002")
        browsed = workspace.task("TASK006")["updated_at_ms"]
    check("reading more does not shift the write", plain == browsed, f"{plain} != {browsed}")


def test_moment_maps_to_the_seed_calendar() -> None:
    check("day 0 midnight is step 0", moment(0, 0, 0, 0) == 0)
    check("one hour is 3600 steps", moment(0, 1) == 3600)
    check("day 1 is a day of steps", moment(1) == 86_400)
    check("VIRTUAL_CLOCK.at(0) is the epoch", VIRTUAL_CLOCK.at(0) == START_MS)
    bad = False
    try:
        moment(0, 24)
    except ValueError:
        bad = True
    check("an impossible wall-clock time is rejected", bad)


def main() -> int:
    print(__doc__)
    for test in (
        test_reads_do_not_move_time,
        test_a_mutation_advances_once,
        test_a_refused_call_rolls_time_back,
        test_the_clock_never_runs_backwards,
        test_the_same_episode_writes_the_same_timestamps,
        test_browsing_first_does_not_perturb_later_writes,
        test_moment_maps_to_the_seed_calendar,
    ):
        print(f"\n{test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all clock checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
