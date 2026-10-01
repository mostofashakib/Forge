"""pass^k: the golden solution must pass every one of k fresh runs, identically.

Each run starts fresh and seeded, confirms the task is not already solved,
runs every golden step, confirms every check passes, and fingerprints the
final state. Any miss, or two different fingerprints, rejects the task.
"""
from __future__ import annotations

from contextlib import contextmanager

from forge.taskfactory.pass_k import run_pass_k
from forge.taskfactory.runner import SeedError, StepResult
from forge.taskfactory.schemas import Check, GoldenStep, TaskDraft
from forge.taskfactory.state_checks import CheckOutcome

DRAFT = TaskDraft(
    slot=0, title="t", objective="o",
    golden=[GoldenStep(tool="a"), GoldenStep(tool="b")],
    checks=[
        Check(kind="record_absent", collection="inbox", match={"id": "e1"}),
        Check(kind="actions_not_called", tools=["delete"]),
    ],
)


class _Session:
    def __init__(self, runner: "_FakeRunner", run: int) -> None:
        self._runner = runner
        self._run = run
        self.steps = 0

    def step(self, step):
        self.steps += 1
        error = self._runner.step_errors.get((self._run, step.tool))
        return StepResult(ok=error is None, error=error or "")

    def evaluate(self, checks):
        solved = self.steps == len(DRAFT.golden)
        if solved and self._run in self._runner.fail_after:
            return [CheckOutcome(False, "inbox still has e1")] + [CheckOutcome(True)] * (len(checks) - 1)
        if not solved and self._runner.solved_at_start:
            return [CheckOutcome(True) for _ in checks]
        return [CheckOutcome(solved or c.kind not in ("record_absent",), "") for c in checks]

    def fingerprint(self):
        return self._runner.fingerprints[self._run % len(self._runner.fingerprints)]


class _FakeRunner:
    def __init__(self, **options) -> None:
        self.step_errors = options.get("step_errors", {})
        self.fail_after = options.get("fail_after", set())
        self.solved_at_start = options.get("solved_at_start", False)
        self.fingerprints = options.get("fingerprints", ["abc"])
        self.seed_error = options.get("seed_error")
        self.sessions = 0
        self.seen_checks: list[list[str]] = []

    @contextmanager
    def session(self, draft):
        run = self.sessions
        self.sessions += 1
        if self.seed_error:
            raise SeedError(self.seed_error)
        session = _Session(self, run)
        original = session.evaluate

        def recording(checks):
            self.seen_checks.append([c.kind for c in checks])
            return original(checks)

        session.evaluate = recording
        yield session


def test_a_golden_solution_that_passes_k_times_identically_is_accepted():
    runner = _FakeRunner()

    result = run_pass_k(runner, DRAFT, k=3)

    assert result.passed
    assert result.fingerprint == "abc"
    assert runner.sessions == 3


def test_only_outcome_checks_are_evaluated_before_the_golden_solution():
    runner = _FakeRunner()

    run_pass_k(runner, DRAFT, k=1)

    assert runner.seen_checks[0] == ["record_absent"]
    assert runner.seen_checks[1] == ["record_absent", "actions_not_called"]


def test_a_task_already_solved_at_the_start_is_rejected():
    result = run_pass_k(_FakeRunner(solved_at_start=True), DRAFT, k=3)

    assert not result.passed
    assert "already" in result.reason


def test_a_failing_golden_step_names_the_run_and_step():
    result = run_pass_k(_FakeRunner(step_errors={(1, "b"): "404 no such email"}), DRAFT, k=3)

    assert not result.passed
    assert "run 2" in result.reason
    assert "step 2" in result.reason
    assert "404 no such email" in result.reason


def test_a_check_failing_on_one_run_rejects_the_task():
    # Flaky: passes on runs 1 and 2, fails on run 3. pass^k means every run.
    result = run_pass_k(_FakeRunner(fail_after={2}), DRAFT, k=3)

    assert not result.passed
    assert "run 3" in result.reason
    assert "inbox still has e1" in result.reason


def test_different_final_states_across_runs_are_rejected():
    result = run_pass_k(_FakeRunner(fingerprints=["abc", "abd"]), DRAFT, k=2)

    assert not result.passed
    assert "differ" in result.reason


def test_a_seed_that_cannot_be_applied_is_rejected():
    result = run_pass_k(_FakeRunner(seed_error="setup command 2 exited 1"), DRAFT, k=3)

    assert not result.passed
    assert "setup command 2 exited 1" in result.reason
