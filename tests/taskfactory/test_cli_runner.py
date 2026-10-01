"""The CLI runner: snapshot once per round, fork per session, grade in the sandbox."""
from __future__ import annotations

import subprocess
from contextlib import contextmanager

import pytest

from forge.taskfactory.pass_k import run_pass_k
from forge.taskfactory.runner import SeedError
from forge.taskfactory.runners.cli import CliRunner
from forge.taskfactory.schemas import Check, GoldenStep, TaskDraft, TaskSeed


class FakeRuntime:
    def __init__(self) -> None:
        self.snapshots: list[str] = []
        self.discarded: list[str] = []
        self.forks: list[str] = []

    def snapshot_cli(self, env_name, container_id, run_id):
        tag = f"forge-cli-snap:{env_name}-{run_id}"
        self.snapshots.append(tag)
        return tag

    def discard_cli_snapshot(self, snapshot):
        self.discarded.append(snapshot)

    @contextmanager
    def cli_episode(self, env_name, snapshot, *, refill):
        assert refill is False
        cid = f"fork-{len(self.forks)}"
        self.forks.append(snapshot)
        yield cid


class FakeShell:
    """Plays each fork as a tiny filesystem of written files."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, str]] = {}
        self.graded: list[str] = []

    def run(self, argv, **kwargs):
        if argv[0] == "sandbox":
            cid, command = argv[1], argv[-1]
            self.graded.append(command)
            ok = command.startswith("has ") and command[4:] in self.files.get(cid, {})
            return subprocess.CompletedProcess(argv, 0 if ok else 1, "", "" if ok else "missing")
        cid, command = argv[2], argv[-1]
        if command.startswith("quiet-fail"):
            return subprocess.CompletedProcess(argv, 1, "matched 0 of 3 rows\n", "")
        if command.startswith("fail"):
            return subprocess.CompletedProcess(argv, 2, "", "boom")
        if command.startswith("write "):
            self.files.setdefault(cid, {})[command[6:]] = "x"
        return subprocess.CompletedProcess(argv, 0, f"ran {command}\n", "")


def _sandbox_factory(shell: FakeShell):
    @contextmanager
    def sandbox(container_id):
        yield ["sandbox", container_id, "sh", "-c"]
    return sandbox


def _runner(runtime=None, shell=None) -> tuple[CliRunner, FakeRuntime, FakeShell]:
    runtime = runtime or FakeRuntime()
    shell = shell or FakeShell()
    runner = CliRunner(
        runtime, "shell-env", "base-cid", batch_id="b1",
        sandbox=_sandbox_factory(shell), run=shell.run,
    )
    return runner, runtime, shell


DRAFT = TaskDraft(
    slot=0, title="Write the report", objective="Create /tmp/report.",
    seed=TaskSeed(setup=["write /data/input"]),
    golden=[GoldenStep(tool="shell", args={"command": "write /tmp/report"})],
    checks=[Check(kind="shell", command="has /tmp/report")],
)


def test_a_sound_task_passes_k_runs_on_separate_forks():
    runner, runtime, _shell = _runner()

    with runner.batch():
        result = run_pass_k(runner, DRAFT, k=3)

    assert result.passed, result.reason
    assert len(runtime.forks) == 3
    assert runtime.snapshots == runtime.discarded


def test_the_snapshot_is_discarded_even_when_a_session_raises():
    runner, runtime, _shell = _runner()

    with pytest.raises(RuntimeError):
        with runner.batch():
            raise RuntimeError("worker died")

    assert runtime.discarded == runtime.snapshots


def test_a_failing_setup_command_raises_seed_error():
    runner, _runtime, _shell = _runner()
    draft = DRAFT.model_copy(update={"seed": TaskSeed(setup=["write a", "fail now"])})

    with runner.batch(), pytest.raises(SeedError, match="setup command 2"):
        with runner.session(draft):
            pass


def test_a_golden_command_that_exits_non_zero_is_a_failed_step():
    runner, _runtime, _shell = _runner()

    with runner.batch(), runner.session(DRAFT) as session:
        result = session.step(GoldenStep(tool="shell", args={"command": "fail please"}))

    assert not result.ok
    assert "boom" in result.error


def test_a_quiet_failure_reports_its_output_when_stderr_is_empty():
    runner, _runtime, _shell = _runner()

    with runner.batch(), runner.session(DRAFT) as session:
        result = session.step(GoldenStep(tool="shell", args={"command": "quiet-fail"}))

    assert not result.ok
    assert "matched 0 of 3 rows" in result.error


def test_checks_run_in_the_grading_sandbox_not_the_agents_shell():
    runner, _runtime, shell = _runner()

    with runner.batch():
        run_pass_k(runner, DRAFT, k=1)

    assert shell.graded == ["has /tmp/report", "has /tmp/report"]


def test_a_session_outside_a_batch_is_refused():
    runner, _runtime, _shell = _runner()

    with pytest.raises(RuntimeError, match="batch"):
        with runner.session(DRAFT):
            pass
