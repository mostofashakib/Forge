import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    return TestClient(app)


def test_extract_endpoint_returns_job_id(client, monkeypatch):
    from forge.extraction.schemas import (
        CompilerInput, EntityDef, FieldDef, ActionDef, ActionParam,
        TaskTemplate, SuccessCondition,
    )
    mock_ci = CompilerInput(
        project_name="test_env",
        domain="test",
        entities=[EntityDef(name="item", fields=[FieldDef(name="id", type="string")])],
        actions=[ActionDef(name="use_item", params=[ActionParam(name="item_id", type="string")])],
        tasks=[TaskTemplate(
            name="complete_task",
            description="Complete the task",
            success_conditions=[SuccessCondition(type="state_check", expression="done")],
        )],
    )
    from backend.app.services import extraction_service
    monkeypatch.setattr(extraction_service, "run_extraction", lambda *a, **kw: mock_ci)

    response = client.post("/api/compile/extract", json={
        "prompt": "A simple test environment",
        "project_name": "test_env",
        "domain": "test",
    })
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "compiler_input" in data


def test_get_job_returns_404_for_unknown(client):
    response = client.get("/api/compile/nonexistent-job-id")
    assert response.status_code == 404


def test_generate_endpoint_triggers_compilation(client, monkeypatch, tmp_path):
    from forge.extraction.schemas import (
        CompilerInput, EntityDef, FieldDef, ActionDef, ActionParam,
        TaskTemplate, SuccessCondition,
    )
    mock_ci = CompilerInput(
        project_name="test_env2",
        domain="test",
        entities=[EntityDef(name="item", fields=[FieldDef(name="id", type="string")])],
        actions=[ActionDef(name="use_item", params=[ActionParam(name="item_id", type="string")])],
        tasks=[TaskTemplate(
            name="complete_task",
            description="desc",
            success_conditions=[SuccessCondition(type="state_check", expression="done")],
        )],
    )
    from backend.app.services import extraction_service, compiler_service
    monkeypatch.setattr(extraction_service, "run_extraction", lambda *a, **kw: mock_ci)
    monkeypatch.setattr(compiler_service, "run_compilation", lambda *a, **kw: None)

    extract_resp = client.post("/api/compile/extract", json={
        "prompt": "test", "project_name": "test_env2", "domain": "test",
    })
    job_id = extract_resp.json()["job_id"]

    gen_resp = client.post(f"/api/compile/generate/{job_id}", json=mock_ci.model_dump())
    assert gen_resp.status_code == 200
