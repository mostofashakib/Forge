"""Run a queued policy training job and record its outcome on the TrainingRun row."""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy.orm import Session

from backend.app.models import TrainingRun
from forge.training.trainer import NoTrainingSignalError, TrainingConfig

logger = logging.getLogger("backend.app.api.training")


class Trainer(Protocol):
    def train(self, config: TrainingConfig): ...


def execute_training_run(
    run_id: str,
    config: TrainingConfig,
    *,
    session_factory: Callable[[], Session],
    trainer: Trainer,
) -> None:
    """Train, then mark the run completed or failed with the reason."""
    if not _update_run(session_factory, run_id, status="running"):
        return
    try:
        result = trainer.train(config)
    except NoTrainingSignalError as exc:
        logger.warning("[training] run %s failed with no training signal: %s", run_id, exc)
        _finish(session_factory, run_id, status="failed", error=f"No training signal: {exc}")
        return
    except Exception as exc:
        logger.exception("[training] run %s encountered an error: %s", run_id, exc)
        _finish(session_factory, run_id, status="failed", error=str(exc))
        return
    _finish(
        session_factory,
        run_id,
        status="completed",
        checkpoint_path=result.checkpoint_path,
        num_examples=result.num_examples,
        mean_reward=getattr(result, "mean_reward", 0.0),
    )
    logger.info("[training] run %s completed: checkpoint at %s", run_id, result.checkpoint_path)


def _finish(session_factory: Callable[[], Session], run_id: str, **fields) -> None:
    _update_run(session_factory, run_id, completed_at=datetime.now(timezone.utc), **fields)


def _update_run(session_factory: Callable[[], Session], run_id: str, **fields) -> bool:
    with session_factory() as db:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return False
        for name, value in fields.items():
            setattr(run, name, value)
        db.commit()
        return True
