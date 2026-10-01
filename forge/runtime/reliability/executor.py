"""Run an episode with capped retries, snapshot replay, divergence flagging and quarantine."""
from __future__ import annotations

import inspect
import json
import logging
import secrets
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from forge.contracts.termination import BUDGET_REASONS
from forge.runtime.reliability.failures import (
    FAILURE_TYPE_AGENT,
    FAILURE_TYPE_INFRASTRUCTURE,
    DivergenceError,
    classify_failure,
    log_infrastructure_failure,
)
from forge.runtime.reliability.quarantine import (
    record_flagged_version_if_needed,
    record_task_quarantine_if_needed,
)
from forge.runtime.reliability.settings import ReliabilitySettings, get_reliability_settings
from forge.runtime.reliability.versioning import compute_environment_version


@dataclass
class AttemptRecord:
    attempt: int
    mode: str  # "fresh" or "replay"
    resume_step: int
    replayed_steps: int
    error: str | None = None
    diverged: bool = False

    def to_dict(self) -> dict:
        return {
            "attempt": self.attempt,
            "mode": self.mode,
            "resume_step": self.resume_step,
            "replayed_steps": self.replayed_steps,
            "error": self.error,
            "diverged": self.diverged,
        }


class ReliableEpisodeExecution(tuple):
    """Result tuple (result, attempts) that also delegates attribute access to result."""

    def __new__(cls, result: Any, attempts: list[AttemptRecord]):
        return super().__new__(cls, (result, attempts))

    @property
    def result(self) -> Any:
        return self[0]

    @property
    def attempts(self) -> list[AttemptRecord]:
        return self[1]

    def __getattr__(self, name: str) -> Any:
        return getattr(self[0], name)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str) and hasattr(self[0], "__getitem__"):
            return self[0][item]
        return super().__getitem__(item)


@dataclass(frozen=True)
class _AttemptPlan:
    attempt: int
    mode: str  # "fresh" or "replay"
    resume_step: int
    replay_steps: list[dict] | None

    @property
    def replayed(self) -> int:
        return len(self.replay_steps) if self.replay_steps else 0


def _plan_attempt(
    attempt: int, last_crash_step: int, settings: ReliabilitySettings, model_outputs: list[dict],
) -> _AttemptPlan:
    """Restart short episodes from scratch; replay long ones from the last snapshot."""
    if attempt == 0 or last_crash_step < settings.short_episode_threshold:
        return _AttemptPlan(attempt, "fresh", 0, None)
    resume = (last_crash_step // settings.snapshot_interval) * settings.snapshot_interval
    return _AttemptPlan(attempt, "replay", resume, model_outputs[:last_crash_step])


def _attempt_invoker(fn: Callable, *, seed: int, env_version: str, model_outputs: list[dict]):
    """Call `fn` the way its signature asks, decided once.

    A runner takes `(attempt, seed)`, takes nothing, or takes the replay
    keywords (`attempt`, `mode`, ...). Deciding from the signature, rather than
    retrying on TypeError, means a TypeError raised inside an episode surfaces
    instead of silently running the episode again.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return lambda plan: fn(plan.attempt, seed)
    if "attempt" in params and "mode" in params:
        return lambda plan: fn(
            attempt=plan.attempt,
            mode=plan.mode,
            replay_steps=plan.replay_steps,
            resume_from_step=plan.resume_step,
            model_outputs=model_outputs,
            env_version=env_version,
        )
    if not params:
        return lambda plan: fn()
    return lambda plan: fn(plan.attempt, seed)


class _EpisodeRecorder:
    """Writes the outcome to the episode row and records flags and quarantines.

    A bookkeeping failure is logged, never raised: it must not turn a finished
    episode into a crash.
    """

    def __init__(
        self, *, db_session, db_session_factory, episode_model, episode_id: str, env_name: str,
        task_name: str, env_version: str, log: logging.Logger,
    ) -> None:
        self._session = db_session
        self._factory = db_session_factory
        self._model = episode_model
        self._episode_id = episode_id
        self._env_name = env_name
        self._task_name = task_name
        self._env_version = env_version
        self._log = log

    def _db(self):
        if self._session is not None:
            return nullcontext(self._session)
        if self._factory is not None:
            return self._factory()
        return None

    def _with_db(self, what: str, action: Callable) -> None:
        ctx = self._db()
        if ctx is None:
            return
        try:
            with ctx as db:
                action(db)
        except Exception as exc:
            self._log.warning("[reliability] could not %s for %s: %s", what, self._episode_id, exc)

    def finish(self, attempts: list[AttemptRecord], status: str, failure_type=None, failure_reason=None) -> None:
        if self._model is None:
            return

        def write(db) -> None:
            episode = db.get(self._model, self._episode_id)
            if episode is None:
                return
            episode.status = status
            episode.failure_type = failure_type
            episode.failure_reason = failure_reason
            episode.environment_version = self._env_version
            episode.attempts_json = json.dumps([a.to_dict() for a in attempts])
            episode.completed_at = datetime.now(timezone.utc)
            db.commit()

        self._with_db("record the episode outcome", write)

    def flag_version(self, reason: str) -> None:
        self._with_db(
            "flag the environment version",
            lambda db: record_flagged_version_if_needed(db, self._env_name, self._env_version, reason),
        )

    def quarantine(self, reason: str) -> None:
        self._with_db(
            "quarantine the task",
            lambda db: record_task_quarantine_if_needed(
                db, self._env_name, self._task_name, self._env_version, reason
            ),
        )


def _is_truncated(result: Any) -> bool:
    return bool(getattr(result, "is_truncated", False)) or (
        getattr(result, "termination_reason", "") in BUDGET_REASONS
    )


def execute_reliable_episode(
    *,
    episode_id: str | None = None,
    env_name: str,
    task_name: str | None = None,
    task_id: str | None = None,
    env_type: str = "general",
    seed: int = 0,
    run_fn: Any = None,
    task_runner: Any = None,
    environment_version: str | None = None,
    db_session: Any = None,
    db_session_factory: Any = None,
    episode_model: Any = None,
    container_id: str | None = None,
    image_id: str | None = None,
    log: logging.Logger | None = None,
    logger: logging.Logger | None = None,
) -> ReliableEpisodeExecution:
    """Run an episode with capped retries, snapshot restoration, replay, and quarantine.

    Agent failures fail at once. Infrastructure crashes retry up to the cap,
    then quarantine the task and flag the environment version.
    """
    fn = run_fn or task_runner
    if fn is None:
        raise ValueError("run_fn or task_runner must be provided")

    resolved_task = task_name or task_id or "default_task"
    resolved_episode_id = episode_id or f"ep_{seed:08x}_{secrets.token_hex(4)}"
    target_log = log or logger or logging.getLogger("forge.reliability")
    settings = get_reliability_settings()
    env_ver = environment_version or compute_environment_version(env_name, env_type, container_id, image_id)
    recorder = _EpisodeRecorder(
        db_session=db_session, db_session_factory=db_session_factory, episode_model=episode_model,
        episode_id=resolved_episode_id, env_name=env_name, task_name=resolved_task,
        env_version=env_ver, log=target_log,
    )
    model_outputs: list[dict] = []
    invoke = _attempt_invoker(fn, seed=seed, env_version=env_ver, model_outputs=model_outputs)
    attempts: list[AttemptRecord] = []
    last_crash_step = 0

    for attempt in range(settings.retry_cap + 1):
        plan = _plan_attempt(attempt, last_crash_step, settings, model_outputs)
        if attempt > 0 and plan.mode == "fresh":
            model_outputs.clear()
        out_of_retries = attempt >= settings.retry_cap
        try:
            result = invoke(plan)
        except DivergenceError as exc:
            last_crash_step = exc.step
            log_infrastructure_failure(
                target_log, episode_id=resolved_episode_id, attempt=attempt, step=exc.step,
                environment_version=env_ver, error=exc,
            )
            attempts.append(AttemptRecord(
                attempt=attempt, mode=plan.mode, resume_step=plan.resume_step,
                replayed_steps=exc.step, error=str(exc), diverged=True,
            ))
            recorder.flag_version(f"Nondeterministic replay divergence at step {exc.step}")
            if out_of_retries:
                recorder.quarantine(f"Divergence crash: {exc}")
                recorder.finish(attempts, "failed", FAILURE_TYPE_INFRASTRUCTURE, "diverged")
                raise
            continue
        except Exception as exc:
            failure_type, failure_reason = classify_failure(exc)
            last_crash_step = getattr(exc, "step", last_crash_step)
            if failure_type == FAILURE_TYPE_AGENT:
                attempts.append(AttemptRecord(
                    attempt=attempt, mode=plan.mode, resume_step=plan.resume_step,
                    replayed_steps=0, error=str(exc),
                ))
                recorder.finish(attempts, "failed", FAILURE_TYPE_AGENT, failure_reason)
                raise
            log_infrastructure_failure(
                target_log, episode_id=resolved_episode_id, attempt=attempt, step=last_crash_step,
                environment_version=env_ver, error=exc,
            )
            attempts.append(AttemptRecord(
                attempt=attempt, mode=plan.mode, resume_step=plan.resume_step,
                replayed_steps=plan.replayed, error=str(exc),
            ))
            if out_of_retries:
                recorder.quarantine(f"Crashed on all {attempt + 1} attempts: {failure_reason} ({exc})")
                recorder.flag_version(f"Task {resolved_task} crashed on all attempts: {failure_reason}")
                recorder.finish(attempts, "failed", FAILURE_TYPE_INFRASTRUCTURE, failure_reason)
                raise
            continue

        attempts.append(AttemptRecord(
            attempt=attempt, mode=plan.mode, resume_step=plan.resume_step,
            replayed_steps=plan.replayed, error=None,
        ))
        recorder.finish(attempts, "truncated" if _is_truncated(result) else "completed")
        if hasattr(result, "environment_version"):
            result.environment_version = env_ver
        if hasattr(result, "attempts"):
            result.attempts = [a.to_dict() for a in attempts]
        return ReliableEpisodeExecution(result, attempts)
