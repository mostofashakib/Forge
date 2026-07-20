from __future__ import annotations

from pathlib import Path

import pytest

from forge.benchmark.compiled_tasks import (
    CompiledTaskProvider,
    db_compiler_input_loader,
    task_from_template,
)
from forge.benchmark.data_collector import (
    CollectionCheckpoint,
    CollectionConfig,
    DataCollector,
)
from forge.extraction.schemas import CompilerInput, SuccessCondition, TaskTemplate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TaskTemplate -> Task mapping
# ---------------------------------------------------------------------------

def test_task_from_template_maps_core_fields():
    task = task_from_template(_template("close_ticket"), env_name="crm_env")
    assert task.name == "close_ticket"
    assert task.domain == "crm_env"
    assert task.objective == "Objective for close_ticket"
    assert callable(task.success_fn)


def test_difficulty_derived_from_condition_count():
    task = task_from_template(_template("t", success=3, failure=1), env_name="e")
    assert task.difficulty == 4


def test_difficulty_clamped_to_one_and_five():
    low = task_from_template(_template("t", success=0), env_name="e")
    high = task_from_template(_template("t", success=9), env_name="e")
    assert low.difficulty == 1
    assert high.difficulty == 5


# ---------------------------------------------------------------------------
# CompiledTaskProvider
# ---------------------------------------------------------------------------

def test_compiled_provider_returns_env_tasks():
    ci = _compiler_input("crm_env", [_template("a"), _template("b")])
    provider = CompiledTaskProvider(loader=lambda name: ci if name == "crm_env" else None)
    tasks = provider.tasks_for(domain="crm_env", depth=5)
    assert {t.name for t in tasks} == {"a", "b"}
    assert all(t.domain == "crm_env" for t in tasks)


def test_compiled_provider_filters_by_depth():
    ci = _compiler_input("crm_env", [
        _template("easy", success=1),   # difficulty 1
        _template("hard", success=5),   # difficulty 5
    ])
    provider = CompiledTaskProvider(loader=lambda name: ci)
    names = {t.name for t in provider.tasks_for(domain="crm_env", depth=2)}
    assert names == {"easy"}


def test_compiled_provider_empty_when_no_compile_job():
    # No compiled artifact for this env -> no tasks (must NOT invent any).
    provider = CompiledTaskProvider(loader=lambda name: None)
    assert provider.tasks_for(domain="unknown_env", depth=5) == []


# ---------------------------------------------------------------------------
# DataCollector honours the injected provider
# ---------------------------------------------------------------------------

def test_collector_produces_pending_runs_for_generated_env(tmp_path):
    ci = _compiler_input("crm_env", [_template("close_ticket")])
    provider = CompiledTaskProvider(loader=lambda name: ci)
    cfg = CollectionConfig(domains=["crm_env"], depth=5, seeds=2, output_dir=tmp_path)
    collector = DataCollector(cfg, task_provider=provider)
    runs = collector._pending_runs(CollectionCheckpoint(output_dir=tmp_path))
    assert [(r["domain"], r["task_name"], r["seed"]) for r in runs] == [
        ("crm_env", "close_ticket", 0),
        ("crm_env", "close_ticket", 1),
    ]


def test_collector_no_pending_runs_when_env_has_no_tasks(tmp_path):
    provider = CompiledTaskProvider(loader=lambda name: None)
    cfg = CollectionConfig(domains=["mystery_env"], depth=5, seeds=3, output_dir=tmp_path)
    collector = DataCollector(cfg, task_provider=provider)
    assert collector._pending_runs(CollectionCheckpoint(output_dir=tmp_path)) == []


# ---------------------------------------------------------------------------
# DB-backed loader
# ---------------------------------------------------------------------------

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
