"""Run tasks in an in-process ForgeEnv, built fresh for every session."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext

from forge.runtime.env import ForgeEnv
from forge.taskfactory.runner import SeedError, StepResult, fingerprint_of
from forge.taskfactory.schemas import Check, GoldenStep, TaskDraft
from forge.taskfactory.state_checks import CheckOutcome, apply_seed, evaluate_state_check

# Headroom past the golden solution, so the step limit never ends a session early.
_STEP_HEADROOM = 10
# Reset with a fixed seed so every session starts from the same universe.
_RESET_SEED = 0


class InProcessRunner:
    def __init__(self, build_env: Callable[[int], ForgeEnv]) -> None:
        """`build_env(max_steps)` returns a fresh environment."""
        self._build_env = build_env

    def batch(self):
        return nullcontext()

    @contextmanager
    def session(self, draft: TaskDraft) -> Iterator["_InProcessSession"]:
        env = self._build_env(len(draft.golden) + _STEP_HEADROOM)
        env.reset(seed=_RESET_SEED)
        try:
            seeded = apply_seed(env.state.get(), draft.seed.records)
        except ValueError as exc:
            raise SeedError(str(exc)) from exc
        env.state.apply(seeded)
        yield _InProcessSession(env)


class _InProcessSession:
    def __init__(self, env: ForgeEnv) -> None:
        self._env = env
        self._called: list[str] = []

    def step(self, step: GoldenStep) -> StepResult:
        try:
            _obs, _reward, _terminated, _truncated, info = self._env.step({**step.args, "type": step.tool})
        except Exception as exc:  # noqa: BLE001 — any refusal is a failed golden step
            return StepResult(ok=False, error=str(exc))
        error = info.get("error") or info.get("policy_violations")
        if error:
            return StepResult(ok=False, error=str(error))
        self._called.append(step.tool)
        return StepResult(ok=True)

    def evaluate(self, checks: Sequence[Check]) -> list[CheckOutcome]:
        state = self._env.state.get()
        return [evaluate_state_check(check, state, self._called) for check in checks]

    def fingerprint(self) -> str:
        return fingerprint_of(self._env.state.get())
