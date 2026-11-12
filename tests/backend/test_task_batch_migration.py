"""init_db adds the edge-case column to a task_batches table made before it existed."""
from __future__ import annotations

import sqlite3

from sqlalchemy import inspect


def test_init_db_adds_edge_cases_to_an_older_task_batches_table(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE task_batches (id TEXT PRIMARY KEY, env_name TEXT, version INTEGER, status TEXT, "
            "requested INTEGER, delivered INTEGER, pass_k INTEGER, data_type TEXT, writer_model TEXT, "
            "validator_model TEXT, taxonomy_json TEXT, error TEXT, created_at DATETIME, "
            "completed_at DATETIME, deleted_at DATETIME)"
        )
        conn.execute(
            "INSERT INTO task_batches (id, env_name, status, requested, delivered, pass_k, writer_model, "
            "validator_model) VALUES ('tb_old', 'mail', 'complete', 3, 3, 1, 'a:b', 'c:d')"
        )
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{path}")
    from backend.app import database

    database._engine = None
    database._SessionLocal = None
    database.init_db()

    engine = database.get_engine()
    assert "edge_cases_json" in {c["name"] for c in inspect(engine).get_columns("task_batches")}
    from backend.app.models import TaskBatch
    from backend.app.services.task_registry import edge_cases_of

    with database.get_session_factory()() as db:
        assert edge_cases_of(db.get(TaskBatch, "tb_old")) == []


def test_a_second_init_db_does_not_overwrite_saved_edge_cases(tmp_path, monkeypatch):
    # A migration that re-ran its ALTER or reset the default would erase what a batch asked for.
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/fresh.db")
    from backend.app import database
    from backend.app.models import TaskBatch
    from backend.app.services.task_registry import edge_cases_of

    database._engine = None
    database._SessionLocal = None
    database.init_db()
    with database.get_session_factory()() as db:
        db.add(TaskBatch(
            id="tb_new", env_name="mail", status="queued", requested=3, pass_k=1,
            writer_model="a:b", validator_model="c:d", edge_cases_json='["recovery"]',
        ))
        db.commit()

    database.init_db()

    with database.get_session_factory()() as db:
        assert edge_cases_of(db.get(TaskBatch, "tb_new")) == ["recovery"]
