"""One model attempt at a stored task, graded by the task's own checks."""
from __future__ import annotations

import copy
import json

import pytest

from forge.contracts import InitialStateProvider
from forge.runtime.env_builder import EnvBuilder
from forge.runtime.errors import AgentError, InvalidActionError
from forge.runtime.transition import TransitionResult
from forge.taskfactory.attempt import (
    Attempt,
    attempt_labels,
    run_attempt,
    summarize_failures,
)
from forge.taskfactory.policy import PolicyError, Submit
from forge.taskfactory.runners.in_process import InProcessRunner
from forge.taskfactory.schemas import (
    Check,
    GoldenStep,
    ReviewVerdict,
    SyntheticTask,
    TaskSeed,
    step_budget,
)
from forge.taskfactory.state_checks import CheckOutcome


class _Inbox(InitialStateProvider):
    def reset(self, ctx, *, seed, options):
        return {"emails": {"e1": {"id": "e1", "folder": "inbox"}}}


def _archive(state, action, ctx):
    email_id = action.get("email_id")
    if email_id not in state["emails"]:
        raise InvalidActionError(f"no email {email_id}")
    new_state = copy.deepcopy(state)
    new_state["emails"][email_id]["folder"] = "archive"
    return TransitionResult(state=new_state, events=[{"type": "archived", "entity_id": email_id}])


def _runner() -> InProcessRunner:
    return InProcessRunner(
        lambda max_steps: EnvBuilder("inbox", domain="mail", max_steps=max_steps)
        .with_initial_state(_Inbox())
        .with_transition("archive", _archive)
        .with_transition("star", lambda state, action, ctx: TransitionResult(state=state, events=[]))
        .build(verify=False)
    )


GOLDEN = [GoldenStep(tool="archive", args={"email_id": "e2"})]
TASK = SyntheticTask(
    id="t-00001", category="triage", difficulty=1, title="Archive the invoice", objective="Archive email e2.",
    seed=TaskSeed(records={"emails": [{"id": "e2", "folder": "inbox"}]}),
    golden=GOLDEN,
    checks=[Check(kind="record_exists", collection="emails", match={"id": "e2"}, expect={"folder": "archive"},
                  description="e2 is archived")],
    reflection_points=[], step_budget=step_budget(GOLDEN), round=1, fingerprint="fp",
    review=ReviewVerdict(slot=0, realistic=True, realistic_reason="r", fair=True, fair_reason="f",
                         sensible=True, sensible_reason="s"),
)


class _Scripted:
    """Plays a fixed list of choices, then keeps choosing the last one."""

    def __init__(self, *choices) -> None:
        self.choices = list(choices)
        self.histories: list[int] = []

    def next_step(self, history):
        self.histories.append(len(history))
        choice = self.choices.pop(0) if len(self.choices) > 1 else self.choices[0]
        if isinstance(choice, Exception):
            raise choice
        return choice


def test_replaying_the_golden_solution_passes():
    attempt = run_attempt(_runner(), TASK, _Scripted(*GOLDEN, Submit()))

    assert attempt.passed
    assert attempt.termination == "submitted"
    assert [turn.step for turn in attempt.turns] == GOLDEN
    assert attempt_labels(attempt, TASK)["passed"] is True
    assert attempt_labels(attempt, TASK)["failed_checks"] == []


def test_submitting_at_once_fails():
    # pass^k proved an outcome check fails before the golden solution runs.
    attempt = run_attempt(_runner(), TASK, _Scripted(Submit()))

    assert not attempt.passed
    assert attempt.termination == "submitted"
    assert attempt.turns == ()
    assert attempt_labels(attempt, TASK)["failed_checks"] == ["e2 is archived"]


def test_a_policy_that_never_submits_stops_at_the_step_budget():
    policy = _Scripted(GoldenStep(tool="star"))

    attempt = run_attempt(_runner(), TASK, policy)

    assert attempt.termination == "budget"
    assert len(attempt.turns) == TASK.step_budget
    assert not attempt.passed


def test_the_budget_ending_after_the_work_is_done_still_passes():
    # The checks grade the final state, so a model that did the work and
    # never submitted solved the task.
    attempt = run_attempt(_runner(), TASK, _Scripted(GOLDEN[0], GoldenStep(tool="star")))

    assert attempt.termination == "budget"
    assert attempt.passed


def test_a_failed_step_is_shown_to_the_model_and_the_attempt_goes_on():
    policy = _Scripted(GoldenStep(tool="archive", args={"email_id": "nope"}), *GOLDEN, Submit())

    attempt = run_attempt(_runner(), TASK, policy)

    assert not attempt.turns[0].result.ok
    assert "nope" in attempt.turns[0].result.error
    assert attempt.passed
    # The budget is 2 steps, so the model is asked twice and the second time
    # it sees the failed step.
    assert policy.histories == [0, 1]
    assert attempt.termination == "budget"
    assert attempt_labels(attempt, TASK)["step_errors"] == 1


@pytest.mark.parametrize("error", [
    AgentError("m answered with no tool call: 'done'"),
    PolicyError("the model called '/nuke', which this environment does not have"),
    json.JSONDecodeError("Expecting value", "{bad", 0),
])
def test_a_model_fault_ends_the_attempt_as_a_policy_error_and_never_passes(error):
    # The golden step ran first, so the checks would pass. The attempt still fails.
    attempt = run_attempt(_runner(), TASK, _Scripted(GOLDEN[0], error))

    assert attempt.termination == "policy_error"
    assert not attempt.passed
    assert attempt.error


def test_an_infrastructure_failure_is_not_blamed_on_the_model():
    with pytest.raises(ConnectionError):
        run_attempt(_runner(), TASK, _Scripted(ConnectionError("ollama is down")))


def test_a_seed_that_cannot_be_applied_is_recorded_as_a_seed_error():
    broken = TASK.model_copy(update={"seed": TaskSeed(records={"no_such_collection": [{"id": "x"}]})})

    attempt = run_attempt(_runner(), broken, _Scripted(Submit()))

    assert attempt.termination == "seed_error"
    assert not attempt.passed


def test_labels_name_tools_the_golden_solution_never_used():
    attempt = run_attempt(_runner(), TASK, _Scripted(GoldenStep(tool="star"), Submit()))

    assert attempt_labels(attempt, TASK)["off_golden_tools"] == ["star"]


# ---------------------------------------------------------------------------
# The summary of why a task's attempts failed (D3)
# ---------------------------------------------------------------------------

def _labels(*policies):
    # Summaries are built from stored labels, so a run's rows are enough to rebuild them.
    return [attempt_labels(run_attempt(_runner(), TASK, policy), TASK) for policy in policies]


def test_the_summary_counts_failures_by_cause_and_names_the_most_common():
    labels = _labels(
        _Scripted(Submit()),
        _Scripted(Submit()),
        _Scripted(GoldenStep(tool="star")),
        _Scripted(*GOLDEN, Submit()),
    )

    summary = summarize_failures(labels)

    assert summary["failed"] == 3
    assert summary["attempts"] == 4
    assert summary["terminations"] == {"submitted": 2, "budget": 1}
    assert summary["failed_checks"] == {"e2 is archived": 3}
    assert "3 of 4 attempts failed" in summary["text"]
    assert "e2 is archived" in summary["text"]


def test_a_task_whose_attempts_all_passed_has_no_failure_summary():
    assert summarize_failures(_labels(_Scripted(*GOLDEN, Submit()))) is None


def test_a_summary_of_no_attempts_is_rejected():
    with pytest.raises(ValueError):
        summarize_failures([])


# ---------------------------------------------------------------------------
# The two anchors on every runner family: replaying the golden solution
# passes, and submitting at once fails.
# ---------------------------------------------------------------------------

def _stored(draft) -> SyntheticTask:
    return SyntheticTask(
        id="t-00002", category="c", difficulty=1, title=draft.title, objective=draft.objective,
        seed=draft.seed, golden=draft.golden, checks=draft.checks, reflection_points=[],
        step_budget=step_budget(draft.golden), round=1, fingerprint="fp", review=TASK.review,
    )


def _cli():
    from tests.taskfactory.test_cli_runner import DRAFT
    from tests.taskfactory.test_cli_runner import _runner as cli_runner

    runner, _runtime, _shell = cli_runner()
    return runner, _stored(DRAFT)


def _browser():
    from tests.taskfactory.test_browser_runner import DRAFT, FakeBrowser
    from tests.taskfactory.test_browser_runner import _runner as browser_runner

    return browser_runner(FakeBrowser()), _stored(DRAFT)


@pytest.mark.parametrize("family", [_cli, _browser], ids=["cli", "browser"])
def test_replaying_the_golden_solution_passes_on_cli_and_browser(family):
    runner, task = family()

    with runner.batch():
        attempt = run_attempt(runner, task, _Scripted(*task.golden, Submit()))

    assert attempt.passed, attempt_labels(attempt, task)


@pytest.mark.parametrize("family", [_cli, _browser], ids=["cli", "browser"])
def test_submitting_at_once_fails_on_cli_and_browser(family):
    runner, task = family()

    with runner.batch():
        attempt = run_attempt(runner, task, _Scripted(Submit()))

    assert not attempt.passed


def test_a_check_without_a_description_is_named_by_what_it_checks():
    # CLI checks rarely carry a description. Naming them all "shell" made
    # different checks count as one cause in the failure summary.
    shell_a = Check(kind="shell", command="test -f /data/a.txt")
    shell_b = Check(kind="shell", command="grep -q done /data/b.txt")
    task = TASK.model_copy(update={"checks": [shell_a, shell_b]})
    attempt = Attempt(turns=(), checks=(CheckOutcome(False), CheckOutcome(False)), termination="submitted")

    names = attempt_labels(attempt, task)["failed_checks"]

    assert names == ["shell: test -f /data/a.txt", "shell: grep -q done /data/b.txt"]


def test_a_long_check_command_is_cut_in_its_name():
    check = Check(kind="shell", command="x" * 300)
    task = TASK.model_copy(update={"checks": [check]})
    attempt = Attempt(turns=(), checks=(CheckOutcome(False),), termination="submitted")

    (name,) = attempt_labels(attempt, task)["failed_checks"]

    assert len(name) < 140
    assert name.endswith("…")
