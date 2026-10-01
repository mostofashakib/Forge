"""Run tasks in a CLI environment: one snapshot per round, a fresh fork per session.

Setup and golden commands run in the fork's shell, like an agent's. Checks
run in the grading sandbox, out of the shell's reach, like the real grader.
"""
from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol

from forge.envgen.grading_sandbox import grading_sandbox
from forge.taskfactory.runner import SeedError, StepResult, fingerprint_of
from forge.taskfactory.schemas import Check, GoldenStep, TaskDraft
from forge.taskfactory.state_checks import CheckOutcome

COMMAND_TIMEOUT_S = 120


class _CliRuntime(Protocol):
    def snapshot_cli(self, env_name: str, container_id: str, run_id: str) -> str: ...

    def discard_cli_snapshot(self, snapshot: str) -> None: ...

    def cli_episode(self, env_name: str, snapshot: str, *, refill: bool) -> AbstractContextManager[str]: ...


class CliRunner:
    def __init__(
        self,
        runtime: _CliRuntime,
        env_name: str,
        container_id: str,
        *,
        batch_id: str,
        sandbox: Callable[[str], AbstractContextManager[list[str]]] = grading_sandbox,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self._runtime = runtime
        self._env_name = env_name
        self._container_id = container_id
        self._batch_id = batch_id
        self._sandbox = sandbox
        self._run = run
        self._snapshot: str | None = None
        self._rounds = 0

    @contextmanager
    def batch(self) -> Iterator[None]:
        self._rounds += 1
        snapshot = self._runtime.snapshot_cli(
            self._env_name, self._container_id, f"tf-{self._batch_id}-{self._rounds}"
        )
        self._snapshot = snapshot
        try:
            yield
        finally:
            self._snapshot = None
            self._runtime.discard_cli_snapshot(snapshot)

    @contextmanager
    def session(self, draft: TaskDraft) -> Iterator["_CliSession"]:
        if self._snapshot is None:
            raise RuntimeError("a CLI session needs an open batch, which holds the snapshot")
        with self._runtime.cli_episode(self._env_name, self._snapshot, refill=False) as container_id:
            for number, command in enumerate(draft.seed.setup, start=1):
                result = self.exec(container_id, command)
                if not result.ok:
                    raise SeedError(f"setup command {number} failed: {result.error}")
            yield _CliSession(self, container_id)

    def exec(self, container_id: str, command: str) -> StepResult:
        try:
            proc = self._run(
                ["docker", "exec", container_id, "bash", "-c", command],
                capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return StepResult(ok=False, error=f"timed out after {COMMAND_TIMEOUT_S}s")
        if proc.returncode != 0:
            # A command that fails quietly says why on stdout, if anywhere.
            said = proc.stderr.strip() or proc.stdout.strip()[-300:]
            return StepResult(ok=False, error=f"exited {proc.returncode}: {said[:300]}", output=proc.stdout)
        return StepResult(ok=True, output=proc.stdout)

    def grade(self, container_id: str, checks: Sequence[Check]) -> list[CheckOutcome]:
        with self._sandbox(container_id) as exec_argv:
            return [self._grade_one(exec_argv, check) for check in checks]

    def _grade_one(self, exec_argv: list[str], check: Check) -> CheckOutcome:
        try:
            proc = self._run([*exec_argv, check.command or ""], capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return CheckOutcome(False, f"timed out after {COMMAND_TIMEOUT_S}s")
        if proc.returncode == 0:
            return CheckOutcome(True)
        return CheckOutcome(False, f"`{check.command}` exited {proc.returncode} {proc.stderr.strip()[:200]}".strip())


class _CliSession:
    def __init__(self, runner: CliRunner, container_id: str) -> None:
        self._runner = runner
        self._container_id = container_id
        self._outputs: list[str] = []

    def step(self, step: GoldenStep) -> StepResult:
        if step.tool != "shell":
            return StepResult(ok=False, error=f"{step.tool!r} is not a shell step")
        result = self._runner.exec(self._container_id, str(step.args.get("command", "")))
        if result.ok:
            self._outputs.append(result.output)
        return result

    def evaluate(self, checks: Sequence[Check]) -> list[CheckOutcome]:
        return self._runner.grade(self._container_id, checks)

    def fingerprint(self) -> str:
        return fingerprint_of(self._outputs)
