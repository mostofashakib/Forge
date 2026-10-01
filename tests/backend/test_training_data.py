"""Training data staging writes only data that a real episode or batch produced."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from backend.app.services import training_data


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/staging.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "envs"))
    from backend.app import database
    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_SessionLocal", None)
    database.init_db()
    with database.get_session_factory()() as session:
        yield session


def _batch(tasks: list[dict], rejections: list[dict] | None = None) -> dict:
    return {"id": "tb_1", "env_name": "shell", "tasks": tasks, "rejections": rejections or []}


GOLDEN_TASK = {
    "id": "t1",
    "objective": "list the files",
    "golden": [{"command": "ls"}, {"command": "wc -l"}],
}


def test_nothing_is_staged_when_there_is_no_real_data(db, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    assert training_data.stage_available_data(db, data_dir) is False
    assert list(data_dir.iterdir()) == []


def test_generator_batch_stages_golden_trajectories_without_invented_rows(tmp_path):
    staged = training_data.stage_generator_batch(
        _batch([GOLDEN_TASK], rejections=[{"slot": 0, "reason": "not fair"}]), tmp_path
    )

    sft = [json.loads(line) for line in (staged / "sft_pairs.jsonl").read_text().splitlines()]
    assert [row["completion"] for row in sft] == ["$ ls\n$ wc -l"]
    rollouts = pd.read_parquet(staged / "grpo_rollouts.parquet")
    assert rollouts["completion"].tolist() == ["ls\nwc -l"]
    assert rollouts["passed"].tolist() == [True]
    # A rejected task is not an answer to an accepted one, so no pair is made up.
    assert not (staged / "preference_pairs.jsonl").exists()
    assert json.loads((staged / "batch_export.json").read_text())["id"] == "tb_1"


def test_generator_batch_without_golden_steps_writes_no_training_rows(tmp_path):
    staged = training_data.stage_generator_batch(
        _batch([{"id": "t2", "objective": "x", "golden": []}]), tmp_path
    )

    assert not (staged / "sft_pairs.jsonl").exists()
    assert not (staged / "grpo_rollouts.parquet").exists()


def test_latest_generator_batch_is_staged_when_no_episodes_exist(db, tmp_path, monkeypatch):
    from backend.app.models import TaskBatch
    from backend.app.services import task_registry

    db.add(TaskBatch(
        id="tb_1", env_name="shell", requested=1, status="completed", pass_k=1,
        writer_model="w", validator_model="v", created_at=datetime.now(timezone.utc),
    ))
    db.commit()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(task_registry, "get_batch", lambda _db, _id: _batch([GOLDEN_TASK]))

    assert training_data.stage_available_data(db, data_dir) is True

    assert (data_dir / "sft_pairs.jsonl").exists()
