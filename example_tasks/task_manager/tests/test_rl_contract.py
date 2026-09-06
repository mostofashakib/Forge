#!/usr/bin/env python3
"""The training loop: lifecycle, shaped reward, and what a policy may see."""

from __future__ import annotations

import sys

from task_sim.environment import CONTRACT_VERSION, TaskHandoverEnvironment
from task_sim.progress import WORKSPACE_MILESTONES, workspace_progress
from task_sim.service import export_state
from workspace import ORACLE_STEPS, Workspace

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def environment(workspace: Workspace) -> TaskHandoverEnvironment:
    return TaskHandoverEnvironment(workspace.db_path, workspace.snapshot_path)


def test_setup_and_reset() -> None:
    with Workspace() as workspace:
        env = environment(workspace)
        info = env.setup()
        check("the contract states its version", info["contract_version"] == CONTRACT_VERSION)
        check("it reports its tool count", info["tool_count"] == 16, str(info["tool_count"]))

        out = env.reset(session_cookie="ep1", user_instruction="Reassign Jordan's work.")
        check("reset returns a ready episode", out["episode"]["state"] == "ready")
        check("with a session at turn zero", out["session"]["turn"] == 0)
        check("and a rendered prompt carrying the tools",
              len(out["prompt"]["tools"]) == 16 and "system" in out["prompt"])

        bad = False
        try:
            env.setup_state(seed=3)
        except ValueError:
            bad = True
        check("an undefined seed variant is refused, not silently ignored", bad)


def test_a_session_cookie_is_validated() -> None:
    with Workspace() as workspace:
        env = environment(workspace)
        env.reset(session_cookie="ep1")
        for cookie in ("", "has space", "x" * 129, "../etc"):
            rejected = False
            try:
                env.state(cookie)
            except (ValueError, KeyError):
                rejected = True
            check(f"{cookie[:12]!r} is refused", rejected)


def test_reward_is_the_change_in_progress() -> None:
    with Workspace() as workspace:
        env = environment(workspace)
        env.reset(session_cookie="ep1")
        total = 0.0
        last = None
        for tool, payload in ORACLE_STEPS:
            out = env.step("ep1", tool, dict(payload))
            total += out["reward"]
            last = out
        check("the rewards sum to the final progress",
              abs(total - last["progress"]) < 1e-6, f"{total} vs {last['progress']}")
        check("a correct episode reaches full progress", abs(last["progress"] - 1.0) < 1e-6,
              str(last["progress"]))
        check("and terminates on the goal", last["terminated"] is True)
        check("rather than on the turn budget", last["truncated"] is False)


def test_a_refused_tool_is_an_observation_not_a_crash() -> None:
    with Workspace() as workspace:
        env = environment(workspace)
        env.reset(session_cookie="ep1")
        out = env.step("ep1", "get_task", {"task_id": "TASK999"})
        check("the step comes back", out["ok"] is False)
        check("carrying the refusal", out["error"]["code"] == "task_not_found")
        check("and costs no progress", out["reward"] == 0.0)


def test_a_rollout_truncates_on_its_budget() -> None:
    with Workspace() as workspace:
        env = environment(workspace)
        env.reset(session_cookie="ep1", max_turns=2)
        env.step("ep1", "list_tasks", {})
        out = env.step("ep1", "list_users", {})
        check("the budget truncates the rollout", out["truncated"] is True)
        check("without claiming the goal was met", out["terminated"] is False)
        after = env.step("ep1", "list_tasks", {})
        check("stepping past the end is reported, not applied",
              after["ok"] is False and after["error"]["code"] == "episode_finished")


def test_state_never_hands_back_the_answer_key() -> None:
    """An RL loop that can read the world is not training on the task."""
    with Workspace() as workspace:
        env = environment(workspace)
        env.reset(session_cookie="ep1")
        env.step("ep1", "list_tasks", {"assignee": "U004"})
        state = env.state("ep1")
        check("only session, history and its total come back",
              set(state) == {"session", "history", "history_total"}, str(sorted(state)))
        blob = repr(state)
        for leak in ("scenario_events", "latent_tasks", "action_log", "integrity_violations"):
            check(f"{leak} is absent", leak not in blob)
        check("history is what the agent already saw", state["history_total"] >= 2)


def test_history_windows_without_lying_about_the_total() -> None:
    with Workspace() as workspace:
        env = environment(workspace)
        env.reset(session_cookie="ep1")
        for _ in range(5):
            env.step("ep1", "list_tasks", {})
        state = env.state("ep1", history_limit=3)
        check("the window is honoured", len(state["history"]) == 3, str(len(state["history"])))
        check("the total is not", state["history_total"] > 3)
        check("and the window is still chronological",
              [row["turn"] for row in state["history"]]
              == sorted(row["turn"] for row in state["history"]))


def test_two_identical_rollouts_are_identical() -> None:
    finals = []
    for _ in range(2):
        with Workspace() as workspace:
            env = environment(workspace)
            env.reset(session_cookie="ep1")
            rewards = [round(env.step("ep1", tool, dict(payload))["reward"], 6)
                       for tool, payload in ORACLE_STEPS]
            finals.append(rewards)
    check("the same episode is shaped the same way twice", finals[0] == finals[1],
          f"{finals[0]} != {finals[1]}")


def test_progress_counts_only_what_the_world_can_see() -> None:
    with Workspace() as workspace:
        scored = workspace_progress(export_state(workspace.db_path))
        check("an untouched workspace has made no progress", scored["progress"] == 0.0)
        check("and is not complete", scored["complete"] is False)
        check("every milestone is declared", set(scored["milestones_met"]) == set(WORKSPACE_MILESTONES))


def test_a_side_effect_costs_progress() -> None:
    with Workspace() as workspace:
        env = environment(workspace)
        env.reset(session_cookie="ep1")
        # Stop short of the report, so the rollout is still live: a terminated
        # episode refuses further steps, which is a different thing to measure.
        for tool, payload in ORACLE_STEPS[:-1]:
            env.step("ep1", tool, dict(payload))
        before = env.state("ep1")["session"]["progress"]
        out = env.step("ep1", "update_task", {"task_id": "TASK022", "assignee": "U005"})
        check("touching a bystander is recorded as a side effect",
              out["side_effects"], str(out["side_effects"]))
        check("and costs progress", out["progress"] < before, f"{out['progress']} vs {before}")
        check("so the step's reward is negative", out["reward"] < 0, str(out["reward"]))


def test_a_finished_episode_still_answers_in_the_same_shape() -> None:
    with Workspace() as workspace:
        env = environment(workspace)
        env.reset(session_cookie="ep1")
        live = None
        for tool, payload in ORACLE_STEPS:
            live = env.step("ep1", tool, dict(payload))
        after = env.step("ep1", "list_tasks", {})
        check("the rollout is over", after["error"]["code"] == "episode_finished")
        # A trainer reads the same keys every turn; the turn that says "stop"
        # must not be the one that breaks it.
        check("the refusal carries every key a live step does",
              set(live) - set(after) - {"ok", "result"} == set(),
              str(set(live) - set(after)))


def main() -> int:
    print(__doc__)
    for test in (
        test_setup_and_reset,
        test_a_session_cookie_is_validated,
        test_reward_is_the_change_in_progress,
        test_a_refused_tool_is_an_observation_not_a_crash,
        test_a_rollout_truncates_on_its_budget,
        test_state_never_hands_back_the_answer_key,
        test_history_windows_without_lying_about_the_total,
        test_two_identical_rollouts_are_identical,
        test_progress_counts_only_what_the_world_can_see,
        test_a_side_effect_costs_progress,
        test_a_finished_episode_still_answers_in_the_same_shape,
    ):
        print(f"\n{test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all RL-contract checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
