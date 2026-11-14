"""Training rows built from a calibration run's graded attempts."""
from __future__ import annotations

import pytest

from forge.contracts import RolloutRecord
from forge.taskfactory.rollout_datasets import FORMATS, Calibrated, dataset_rows
from forge.taskfactory.schemas import (
    Check,
    GoldenStep,
    ReviewVerdict,
    SyntheticTask,
    TaskSeed,
)

RUN = {"env_name": "ops", "run_id": "cr_1", "target_model": "ollama:m"}


def _task(n: int) -> SyntheticTask:
    return SyntheticTask(
        id=f"t-{n:05d}", category="triage", difficulty=2, title=f"task {n}", objective=f"objective {n}",
        seed=TaskSeed(), golden=[GoldenStep(tool="shell", args={"command": "true"})],
        checks=[Check(kind="shell", command="test -f /done", description="done exists")],
        reflection_points=[], step_budget=4, round=1, fingerprint="fp",
        review=ReviewVerdict(slot=n - 1, realistic=True, realistic_reason="r", fair=True, fair_reason="f",
                             sensible=True, sensible_reason="s"),
    )


def _attempt(index: int, passed: bool) -> dict:
    command = f"touch /done{'' if passed else '-wrong'}-{index}"
    return {
        "attempt_index": index, "passed": passed, "termination": "submitted", "error": "",
        "steps": [{"step": {"tool": "shell", "args": {"command": command}}, "ok": True, "output": "", "error": ""}],
        "checks": [{"passed": passed, "detail": "" if passed else "exit 1"}],
        "labels": {"passed": passed, "termination": "submitted", "failed_checks": [] if passed else ["done exists"],
                   "step_errors": 0, "off_golden_tools": []},
    }


def _calibrated(n: int, results: list[bool]) -> Calibrated:
    return Calibrated(task=_task(n), attempts=[_attempt(i, passed) for i, passed in enumerate(results)])


# t-00001 is too easy, t-00002 and t-00003 are on the frontier, t-00004 is unsolved.
RUN_TASKS = [
    _calibrated(1, [True, True, True]),
    _calibrated(2, [True, False, False]),
    _calibrated(3, [False, True, True]),
    _calibrated(4, [False, False, False]),
]


def test_grpo_has_one_group_per_frontier_task_with_every_attempt():
    rows = dataset_rows("grpo", RUN_TASKS, **RUN)

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["task_name"], []).append(row)
    assert sorted(groups) == ["t-00002", "t-00003"]
    assert all(len(group) == 3 for group in groups.values())


def test_grpo_rewards_are_one_for_a_pass_and_zero_for_a_fail():
    rows = dataset_rows("grpo", [_calibrated(2, [True, False, False])], **RUN)

    assert [(row["passed"], row["total_reward"]) for row in rows] == [(True, 1.0), (False, 0.0), (False, 0.0)]


def test_grpo_rows_are_rollout_records_with_the_prompt_the_model_saw_and_its_steps():
    row = dataset_rows("grpo", [_calibrated(2, [True, False])], **RUN)[0]

    record = RolloutRecord.model_validate(row)
    assert record.prompt == "objective 2"
    assert record.completion == "$ touch /done-0"
    assert record.env_name == "ops" and record.behavior_model == "ollama:m"
    assert record.episode_id == "cr_1/t-00002/0"
    assert record.termination_reason == "submitted"


def test_steps_that_are_not_shell_commands_keep_their_tool_and_arguments():
    calibrated = _calibrated(2, [True, False])
    calibrated.attempts[0]["steps"] = [
        {"step": {"tool": "fill", "args": {"selector": "#name", "value": "Ada"}}, "ok": True, "output": "", "error": ""},
    ]

    row = dataset_rows("grpo", [calibrated], **RUN)[0]

    assert row["completion"] == 'fill {"selector": "#name", "value": "Ada"}'


def test_preference_pairs_each_failing_attempt_with_a_passing_one_from_the_same_task():
    rows = dataset_rows("preference", RUN_TASKS, **RUN)

    assert [(row["task"], row["chosen_passed"], row["rejected_passed"]) for row in rows] == [
        ("t-00002", True, False), ("t-00002", True, False), ("t-00003", True, False),
    ]
    first = rows[0]
    assert first["chosen"] == [
        {"role": "user", "content": "objective 2"}, {"role": "assistant", "content": "$ touch /done-0"},
    ]
    assert first["rejected"][1] == {"role": "assistant", "content": "$ touch /done-wrong-1"}
    assert first["chosen_reward"] > first["rejected_reward"]


def test_the_rejected_side_carries_its_failed_checks_as_the_reason():
    row = dataset_rows("preference", [_calibrated(2, [True, False])], **RUN)[0]

    assert row["rejected_failed_checks"] == ["done exists"]


def test_preference_makes_no_pair_for_a_task_with_only_passes_or_only_fails():
    rows = dataset_rows("preference", [_calibrated(1, [True, True]), _calibrated(4, [False, False])], **RUN)

    assert rows == []


def test_sft_keeps_passing_attempts_from_frontier_tasks_only():
    rows = dataset_rows("sft", RUN_TASKS, **RUN)

    assert [(row["task_id"], row["attempt_index"]) for row in rows] == [("t-00002", 0), ("t-00003", 1), ("t-00003", 2)]
    assert rows[0]["messages"] == [
        {"role": "user", "content": "objective 2"}, {"role": "assistant", "content": "$ touch /done-0"},
    ]


def test_failures_keep_every_failing_attempt_with_its_checks_and_labels():
    rows = dataset_rows("failures", RUN_TASKS, **RUN)

    assert [(row["task_id"], row["attempt_index"]) for row in rows] == [
        ("t-00002", 1), ("t-00002", 2), ("t-00003", 0), ("t-00004", 0), ("t-00004", 1), ("t-00004", 2),
    ]
    first = rows[0]
    assert first["failed_checks"] == ["done exists"]
    assert first["checks"] == [{"passed": False, "detail": "exit 1"}]
    assert first["labels"]["termination"] == "submitted"
    assert first["steps"][0]["step"]["args"]["command"] == "touch /done-wrong-1"


def test_every_format_is_known_and_another_is_refused():
    assert FORMATS == ("grpo", "preference", "sft", "failures")
    with pytest.raises(ValueError, match="grpo, preference, sft, failures"):
        dataset_rows("dpo", RUN_TASKS, **RUN)


def test_an_attempt_whose_seed_failed_is_not_training_data_but_is_a_failure():
    calibrated = _calibrated(2, [True, False, False])
    calibrated.attempts[2].update(termination="seed_error", error="seed failed", steps=[])

    grpo = dataset_rows("grpo", [calibrated], **RUN)
    pairs = dataset_rows("preference", [calibrated], **RUN)
    failures = dataset_rows("failures", [calibrated], **RUN)

    assert [row["episode_id"] for row in grpo] == ["cr_1/t-00002/0", "cr_1/t-00002/1"]
    assert len(pairs) == 1
    assert [row["termination"] for row in failures] == ["submitted", "seed_error"]
