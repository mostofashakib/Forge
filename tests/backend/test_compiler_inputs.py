"""The compiled-input loader returns an environment's latest compile, or None."""
from __future__ import annotations

import pytest

from backend.app.services.compiler_inputs import db_compiler_input_loader
from forge.extraction.schemas import CompilerInput, SuccessCondition, TaskTemplate


def _template(name: str, *, success: int = 1, failure: int = 0) -> TaskTemplate:
    return TaskTemplate(
        name=name,
        description=f"Objective for {name}",
        success_conditions=[
            SuccessCondition(type="state_check", expression=f"x == {i}") for i in range(success)
        ],
        failure_conditions=[
            SuccessCondition(type="negative_check", expression=f"e_{i}") for i in range(failure)
        ],
    )


def _compiler_input(name: str, templates: list[TaskTemplate]) -> CompilerInput:
    return CompilerInput(
        project_name=name,
        domain="generated",
        entities=[],
        actions=[],
        tasks=templates,
    )


@pytest.fixture
def session_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/bench.db")
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    return database.get_session_factory()


def _insert_compile_job(session_factory, project_name, compiler_input, created_at):
    import uuid
    from backend.app.models import CompileJob
    db = session_factory()
    try:
        db.add(CompileJob(
            id=uuid.uuid4().hex,
            project_name=project_name,
            status="done",
            prompt="",
            compiler_input_json=compiler_input.model_dump_json(),
            created_at=created_at,
        ))
        db.commit()
    finally:
        db.close()


def test_db_loader_returns_latest_compiler_input(session_factory):
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    _insert_compile_job(
        session_factory, "crm_env",
        _compiler_input("crm_env", [_template("old_task")]),
        now - timedelta(hours=1),
    )
    _insert_compile_job(
        session_factory, "crm_env",
        _compiler_input("crm_env", [_template("new_task")]),
        now,
    )
    loader = db_compiler_input_loader(session_factory)
    ci = loader("crm_env")
    assert ci is not None
    assert [t.name for t in ci.tasks] == ["new_task"]


def test_db_loader_returns_none_for_unknown_env(session_factory):
    loader = db_compiler_input_loader(session_factory)
    assert loader("never_compiled") is None
