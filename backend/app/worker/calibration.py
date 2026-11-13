"""The Celery job that calibrates a saved batch against a target model.

Each task gets its attempts in a row, under the environment lock, and they
are stored before the next task starts. A failure keeps what was stored.

Streams progress to the Redis channel forge:calibration:{run_id}:
  {"log": ..., "done_tasks": n, "total_tasks": m}   a task finished
  {"done": true}                                     the run is complete
  {"error": "..."}                                   the run failed
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import timedelta

from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from backend.app.models import CalibrationRun, TaskBatch
from backend.app.services import calibration_registry, task_registry
from backend.app.services.task_factory_targets import open_target as default_open_target
from backend.app.worker.celery_app import celery
from forge.settings import redis_url
from forge.taskfactory.attempt import run_attempt
from forge.taskfactory.policy import Policy, agent_policy, target_problem
from forge.taskfactory.profile import EnvironmentProfile
from forge.taskfactory.schemas import SyntheticTask

logger = logging.getLogger(__name__)

# The job stops between tasks at this limit. Celery's soft limit fails a run
# whose current task overruns it by more than the grace, and the hard limit
# kills the process. The reaper then fails any run still "running" past it.
CALIBRATION_TIME_LIMIT_S = 24 * 60 * 60
TASK_OVERRUN_GRACE_S = 2 * 60 * 60
CALIBRATION_SOFT_LIMIT_S = CALIBRATION_TIME_LIMIT_S + TASK_OVERRUN_GRACE_S
CALIBRATION_HARD_LIMIT_S = CALIBRATION_SOFT_LIMIT_S + 10 * 60

MakePolicy = Callable[[str, SyntheticTask, EnvironmentProfile], Policy]


class _Stopped(RuntimeError):
    """The run reached its time limit between tasks."""


def channel_for(run_id: str) -> str:
    return f"forge:calibration:{run_id}"


def execute_calibration(
    db: Session,
    run_id: str,
    *,
    publish: Callable[[dict], None],
    exclusive: Callable[[str], AbstractContextManager[None]],
    open_target=default_open_target,
    make_policy: MakePolicy = agent_policy,
) -> None:
    """Attempt every task in the run's batch, storing each task's attempts as it finishes."""
    run = db.get(CalibrationRun, run_id)
    if run is None:
        logger.warning("[calibration] run %s vanished before it ran", run_id)
        return
    target_model, attempts = run.target_model, run.attempts_per_task
    try:
        problem = target_problem(target_model)
        if problem:
            raise ValueError(problem)
        batch = db.get(TaskBatch, run.batch_id)
        if batch is None or batch.deleted_at is not None or batch.version is None:
            raise ValueError(f"batch {run.batch_id} is not a saved batch")
        tasks = task_registry.batch_tasks(db, run.batch_id)
        calibration_registry.mark_running(db, run_id)
        deadline = time.monotonic() + CALIBRATION_TIME_LIMIT_S
        with open_target(db, batch.env_name, run_id) as target:
            with exclusive(batch.env_name):
                profile = target.profile()
            for done, task in enumerate(tasks):
                if time.monotonic() >= deadline:
                    raise _Stopped(
                        f"stopped at the {CALIBRATION_TIME_LIMIT_S // 3600}-hour time limit "
                        f"after {done} of {len(tasks)} tasks"
                    )
                policy = make_policy(target_model, task, profile)
                with exclusive(batch.env_name), target.runner.batch():
                    results = [run_attempt(target.runner, task, policy) for _ in range(attempts)]
                calibration_registry.record_attempts(db, run_id, task, results)
                passed = sum(result.passed for result in results)
                publish({
                    "log": f"{task.id}: {passed} of {attempts} attempts passed",
                    "done_tasks": done + 1, "total_tasks": len(tasks),
                })
        calibration_registry.complete_run(db, run_id)
    except SoftTimeLimitExceeded:
        _fail(db, run_id, publish, f"stopped by the {CALIBRATION_SOFT_LIMIT_S // 3600}-hour time limit while a task was running")
        return
    except Exception as exc:  # noqa: BLE001 — every failure is recorded on the run
        logger.exception("[calibration] run %s failed", run_id)
        _fail(db, run_id, publish, str(exc))
        return
    publish({"done": True, "log": f"Calibrated {len(tasks)} tasks with {target_model}"})


def _fail(db: Session, run_id: str, publish: Callable[[dict], None], error: str) -> None:
    db.rollback()
    calibration_registry.mark_failed(db, run_id, error)
    publish({"error": error})


def reap_abandoned_calibrations(db: Session) -> list[str]:
    """Fail runs whose worker died, judged by the hard limit no live worker passes."""
    reaped = calibration_registry.fail_abandoned_runs(db, older_than=timedelta(seconds=CALIBRATION_HARD_LIMIT_S))
    if reaped:
        logger.warning("[calibration] failed abandoned runs %s", reaped)
    return reaped


# The worker can boot before the API's init_db creates new tables, so the
# boot-time reap retries for about five minutes.
@celery.task(
    name="backend.app.worker.calibration.reap_abandoned_calibrations_task", ignore_result=True,
    autoretry_for=(OperationalError,), default_retry_delay=30, max_retries=10,
)
def reap_abandoned_calibrations_task() -> None:
    from backend.app.database import get_session_factory

    with get_session_factory()() as db:
        reap_abandoned_calibrations(db)


@celery.task(
    name="backend.app.worker.calibration.run_calibration_task", ignore_result=True,
    soft_time_limit=CALIBRATION_SOFT_LIMIT_S, time_limit=CALIBRATION_HARD_LIMIT_S,
)
def run_calibration_task(run_id: str) -> None:
    import redis

    from backend.app.database import get_session_factory
    from backend.app.worker.env_lock import exclusive_environment

    client = redis.from_url(redis_url())

    def publish(message: dict) -> None:
        try:
            client.publish(channel_for(run_id), json.dumps(message))
        except Exception:  # noqa: BLE001 — progress is best effort, the run row is the record
            logger.debug("[calibration] progress publish failed", exc_info=True)

    with get_session_factory()() as db:
        execute_calibration(
            db, run_id, publish=publish,
            exclusive=lambda env_name: exclusive_environment(client, env_name),
        )
