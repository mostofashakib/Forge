"""The training-run service records each trainer outcome on its run."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.models import TrainingRun
from backend.app.services.training_runs import execute_training_run
from forge.training.trainer import NoTrainingSignalError, TrainingConfig, TrainingObjective


@pytest.fixture
def session_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/runs.db")
    from backend.app import database
    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_SessionLocal", None)
    database.init_db()
    factory = database.get_session_factory()
    with factory() as db:
        db.add(TrainingRun(
            id="tr_1", status="queued", objective="grpo", base_model="m",
            data_dir="d", output_dir="o", max_steps=1, created_at=datetime.now(timezone.utc),
        ))
        db.commit()
    return factory


CONFIG = TrainingConfig(
    data_dir=Path("d"), base_model="m", output_dir=Path("o"), objective=TrainingObjective("grpo"),
)


class _Trainer:
    def __init__(self, outcome):
        self._outcome = outcome

    def train(self, _config):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _run(factory):
    with factory() as db:
        run = db.get(TrainingRun, "tr_1")
        return run.status, run.error, run.checkpoint_path, run.completed_at is not None


def test_a_successful_run_records_its_checkpoint(session_factory):
    result = SimpleNamespace(checkpoint_path="ckpt", num_examples=4, mean_reward=0.5)
    execute_training_run("tr_1", CONFIG, session_factory=session_factory, trainer=_Trainer(result))

    assert _run(session_factory) == ("completed", None, "ckpt", True)


def test_a_run_without_signal_fails_with_that_reason(session_factory):
    trainer = _Trainer(NoTrainingSignalError("all rewards equal"))
    execute_training_run("tr_1", CONFIG, session_factory=session_factory, trainer=trainer)

    assert _run(session_factory) == ("failed", "No training signal: all rewards equal", None, True)


def test_a_crashing_trainer_fails_the_run(session_factory):
    trainer = _Trainer(RuntimeError("cuda oom"))
    execute_training_run("tr_1", CONFIG, session_factory=session_factory, trainer=trainer)

    assert _run(session_factory) == ("failed", "cuda oom", None, True)


def test_a_missing_run_never_trains(session_factory):
    trainer = _Trainer(AssertionError("must not train"))
    execute_training_run("tr_missing", CONFIG, session_factory=session_factory, trainer=trainer)
