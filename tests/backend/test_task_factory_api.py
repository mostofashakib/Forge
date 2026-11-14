"""The task factory API: pick an environment and a count, then browse versions."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models import SandboxEnvironment
from backend.app.services import calibration_registry, env_file, task_registry
from forge.taskfactory.calibration import FRONTIER_MAX, FRONTIER_MIN
from forge.taskfactory.model_settings import ModelSpec
from forge.taskfactory.pipeline import PipelineResult
from forge.taskfactory.schemas import Taxonomy, TaxonomyCategory
from tests.backend.safety_helpers import use_safety_env


@pytest.fixture(autouse=True)
def sdks_installed(monkeypatch):
    import forge.taskfactory.model_settings as settings

    monkeypatch.setattr(settings, "_installed", lambda module: True)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    monkeypatch.setenv("FORGE_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-5")
    use_safety_env(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text("FORGE_TASK_VALIDATOR_PROVIDER=openai\nFORGE_TASK_VALIDATOR_MODEL=gpt-5\n")
    monkeypatch.setattr(env_file, "BACKEND_ENV_FILE", env_path)
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    with database.get_session_factory()() as db:
        for name, status in (("mail", "running"), ("web", "stopped")):
            db.add(SandboxEnvironment(
                id=name, status=status, env_type="premade:gmail" if name == "mail" else "browser",
                container_id="c1", expires_at=datetime.now(timezone.utc) + timedelta(days=30),
            ))
        db.commit()
    client = TestClient(app)
    client.env_path = env_path
    client.session_factory = database.get_session_factory()
    return client


@pytest.fixture
def queued():
    with patch("backend.app.api.task_factory.create_task_batch_task") as task:
        task.delay = MagicMock()
        yield task


def test_environments_report_type_and_readiness(client):
    body = {e["name"]: e for e in client.get("/api/task-factory/environments").json()}

    assert body["mail"]["family"] == "state" and body["mail"]["ready"] is True
    assert body["web"]["ready"] is False


def test_starting_a_batch_queues_the_job(client, queued):
    resp = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": 12, "k": 3})

    assert resp.status_code == 202
    batch_id = resp.json()["batch_id"]
    queued.delay.assert_called_once_with(batch_id=batch_id)
    batch = client.get(f"/api/task-factory/batches/{batch_id}").json()
    assert batch["status"] == "queued"
    assert batch["requested"] == 12 and batch["pass_k"] == 3
    assert batch["validator_model"] == "openai:gpt-5"


def test_a_batch_of_twenty_thousand_is_accepted(client, queued):
    resp = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": 20_000, "k": 3})

    assert resp.status_code == 202


def test_a_stopped_environment_is_refused(client, queued):
    resp = client.post("/api/task-factory/batches", json={"env_name": "web", "count": 5, "k": 3})

    assert resp.status_code == 409
    assert "Start" in resp.json()["detail"]
    queued.delay.assert_not_called()


def test_an_unknown_environment_is_refused(client, queued):
    resp = client.post("/api/task-factory/batches", json={"env_name": "nope", "count": 5, "k": 3})

    assert resp.status_code == 404


def test_missing_validator_settings_are_refused_by_name(client, queued):
    client.env_path.write_text("")

    resp = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": 5, "k": 3})

    assert resp.status_code == 422
    assert "FORGE_TASK_VALIDATOR_PROVIDER" in resp.json()["detail"]


def test_a_validator_sdk_that_is_not_installed_is_refused_before_queueing(client, queued, monkeypatch):
    import forge.taskfactory.model_settings as settings

    monkeypatch.setattr(settings, "_installed", lambda module: module != "openai")

    resp = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": 5, "k": 3})

    assert resp.status_code == 422
    assert "openai" in resp.json()["detail"]
    queued.delay.assert_not_called()


@pytest.mark.parametrize("count, k", [(0, 3), (20_001, 3), (5, 0), (5, 11)])
def test_count_and_k_outside_their_limits_are_refused(client, queued, count, k):
    resp = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": count, "k": k})

    assert resp.status_code == 422


def _saved_batch(client) -> str:
    taxonomy = Taxonomy(
        categories=[TaxonomyCategory(name="triage", description="sort", exercises=[], difficulties=[1])],
    )
    with client.session_factory() as db:
        batch_id = task_registry.create_batch(
            db, env_name="mail", requested=2, pass_k=3,
            writer=ModelSpec("anthropic", "claude-sonnet-5"), validator=ModelSpec("openai", "gpt-5"),
        )
        task_registry.save_result(db, batch_id, PipelineResult(requested=2, taxonomy=taxonomy))
    return batch_id


def test_batches_are_listed_per_environment(client):
    batch_id = _saved_batch(client)

    listed = client.get("/api/task-factory/batches", params={"env_name": "mail"}).json()

    assert [b["id"] for b in listed] == [batch_id]
    assert client.get("/api/task-factory/batches", params={"env_name": "web"}).json() == []


def test_a_saved_batch_exports_as_a_dated_json_file(client):
    batch_id = _saved_batch(client)

    resp = client.get(f"/api/task-factory/batches/{batch_id}/export")

    assert resp.status_code == 200
    disposition = resp.headers["content-disposition"]
    assert "mail-" in disposition and disposition.endswith('.json"')
    body = resp.json()
    assert body["version"] == 1
    assert body["taxonomy"]["categories"][0]["name"] == "triage"
    assert body["created_at"]


def test_an_unsaved_batch_cannot_be_exported(client, queued):
    batch_id = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": 1, "k": 1}).json()["batch_id"]

    assert client.get(f"/api/task-factory/batches/{batch_id}/export").status_code == 409


def test_a_saved_batch_can_be_deleted(client):
    batch_id = _saved_batch(client)

    assert client.delete(f"/api/task-factory/batches/{batch_id}").status_code == 204
    assert client.get(f"/api/task-factory/batches/{batch_id}").status_code == 404


def test_a_running_batch_cannot_be_deleted(client, queued):
    batch_id = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": 1, "k": 1}).json()["batch_id"]

    assert client.delete(f"/api/task-factory/batches/{batch_id}").status_code == 409


def test_an_unknown_batch_is_not_found(client):
    assert client.get("/api/task-factory/batches/tb_missing").status_code == 404
    assert client.delete("/api/task-factory/batches/tb_missing").status_code == 404


def _preference_batch(client) -> str:
    from forge.taskfactory.schemas import (
        Check, GoldenStep, ReviewVerdict, SyntheticTask, TaskDraft, TaskRejection, TaskSeed,
    )

    def draft(slot: int, tool: str) -> TaskDraft:
        return TaskDraft(
            slot=slot, title=f"draft {slot}", objective="archive e1", seed=TaskSeed(),
            golden=[GoldenStep(tool=tool, args={})],
            checks=[Check(kind="value", path="archived", value=True)],
        )

    verdict = ReviewVerdict(
        slot=0, realistic=True, realistic_reason="ok", fair=True, fair_reason="ok",
        sensible=True, sensible_reason="ok",
    )
    task = SyntheticTask(
        id="t-00001", category="triage", difficulty=1, title="Archive", objective="archive e1",
        seed=TaskSeed(), golden=[GoldenStep(tool="/archive_email", args={})],
        checks=[Check(kind="value", path="archived", value=True)], reflection_points=[],
        step_budget=2, round=2, review=verdict, fingerprint="fp",
    )
    rejections = [
        TaskRejection(slot=0, category="triage", difficulty=1, round=1, stage="pass_k",
                      reason="flaky check", draft=draft(0, "/delete_email")),
        TaskRejection(slot=1, category="triage", difficulty=1, round=1, stage="review",
                      reason="unfair", draft=draft(1, "/star_email")),
    ]
    taxonomy = Taxonomy(categories=[TaxonomyCategory(name="triage", description="sort", exercises=[], difficulties=[1])])
    with client.session_factory() as db:
        batch_id = task_registry.create_batch(
            db, env_name="mail", requested=2, pass_k=3,
            writer=ModelSpec("anthropic", "claude-sonnet-5"), validator=ModelSpec("openai", "gpt-5"),
        )
        task_registry.save_result(db, batch_id, PipelineResult(
            requested=2, taxonomy=taxonomy, tasks=[task], rejections=rejections,
            data_type="preference_pairs",
        ))
    return batch_id


def test_exported_preference_pairs_compare_drafts_written_for_the_same_slot(client):
    batch_id = _preference_batch(client)

    pairs = client.get(f"/api/task-factory/batches/{batch_id}/export").json()["preference_pairs"]

    assert pairs == [{
        "prompt": "archive e1",
        "preferred_response": "/archive_email",
        "dispreferred_response": "/delete_email",
        "metadata": {"task_id": "t-00001", "difficulty": 1, "rejection_reason": "flaky check"},
    }]


def test_a_preference_batch_without_rejected_drafts_exports_no_pairs(client):
    with client.session_factory() as db:
        batch_id = task_registry.create_batch(
            db, env_name="mail", requested=1, pass_k=3,
            writer=ModelSpec("anthropic", "claude-sonnet-5"), validator=ModelSpec("openai", "gpt-5"),
        )
        task_registry.save_result(db, batch_id, PipelineResult(
            requested=1, taxonomy=Taxonomy(categories=[]), data_type="preference_pairs",
        ))

    body = client.get(f"/api/task-factory/batches/{batch_id}/export").json()

    assert body["preference_pairs"] == []


SECRET = "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv_wxyz"


def _task(task_id: str, objective: str, steps):
    from forge.taskfactory.schemas import Check, GoldenStep, ReviewVerdict, SyntheticTask, TaskSeed

    verdict = ReviewVerdict(slot=0, realistic=True, realistic_reason="ok", fair=True, fair_reason="ok",
                            sensible=True, sensible_reason="ok")
    return SyntheticTask(
        id=task_id, category="triage", difficulty=1, title=task_id, objective=objective, seed=TaskSeed(),
        golden=[GoldenStep(tool="shell", args={"command": command}) for command in steps],
        checks=[Check(kind="value", path="done", value=True)], reflection_points=[],
        step_budget=2, round=1, review=verdict, fingerprint=task_id,
    )


def _batch_of(client, tasks, rejections=(), data_type="sft") -> str:
    taxonomy = Taxonomy(categories=[TaxonomyCategory(name="triage", description="sort", exercises=[], difficulties=[1])])
    with client.session_factory() as db:
        batch_id = task_registry.create_batch(
            db, env_name="mail", requested=len(tasks), pass_k=3,
            writer=ModelSpec("anthropic", "claude-sonnet-5"), validator=ModelSpec("openai", "gpt-5"),
        )
        task_registry.save_result(db, batch_id, PipelineResult(
            requested=len(tasks), taxonomy=taxonomy, tasks=list(tasks), rejections=list(rejections), data_type=data_type,
        ))
    return batch_id


def _quarantined(tmp_path) -> str:
    files = list((tmp_path / "generated_envs" / "mail" / "exports" / "downloads").glob("*/safety_quarantine.jsonl"))
    return "".join(path.read_text() for path in files)


def test_a_batch_export_drops_harmful_tasks_and_redacts_secrets(client, tmp_path):
    batch_id = _batch_of(client, [
        _task("t-ok", f"Rotate the key {SECRET}", ["ls /var/log"]),
        _task("t-bad", "Free some disk", ["rm -rf /"]),
    ])

    body = client.get(f"/api/task-factory/batches/{batch_id}/export").json()

    text = json.dumps(body)
    assert [task["id"] for task in body["tasks"]] == ["t-ok"]
    assert [item["metadata"]["task_id"] for item in body["sft_items"]] == ["t-ok"]
    assert SECRET not in text and "rm -rf /" not in text
    assert body["safety"]["source"] == "model"
    assert (body["safety"]["rows_seen"], body["safety"]["rows_kept"]) == (2, 1)
    assert body["safety"]["quarantined_by_rule"] == {"destructive_command": 1}
    assert "rm -rf /" in _quarantined(tmp_path)


def test_rejected_drafts_are_screened_before_they_become_preference_pairs(client, tmp_path):
    from forge.taskfactory.schemas import Check, GoldenStep, TaskDraft, TaskRejection, TaskSeed

    harmful = TaskDraft(slot=0, title="d", objective="Free some disk", seed=TaskSeed(),
                        golden=[GoldenStep(tool="shell", args={"command": "rm -rf /"})],
                        checks=[Check(kind="value", path="done", value=True)])
    rejection = TaskRejection(slot=0, category="triage", difficulty=1, round=1, stage="review", reason="unsafe", draft=harmful)
    batch_id = _batch_of(client, [_task("t-ok", "Free some disk", ["du -sh /var/log"])], [rejection], "preference_pairs")

    body = client.get(f"/api/task-factory/batches/{batch_id}/export").json()

    assert body["preference_pairs"] == []
    assert body["rejections"] == []
    assert "rm -rf /" not in json.dumps(body)


def test_a_batch_export_without_a_working_gate_is_refused(client, monkeypatch):
    batch_id = _batch_of(client, [_task("t-ok", "List logs", ["ls /var/log"])])
    monkeypatch.setenv("FORGE_ADULT_THRESHOLD", "5")

    response = client.get(f"/api/task-factory/batches/{batch_id}/export")

    assert response.status_code == 503
    assert "FORGE_ADULT_THRESHOLD" in response.json()["detail"]


def test_a_batch_records_its_requested_edge_cases(client, queued):
    resp = client.post(
        "/api/task-factory/batches",
        json={"env_name": "mail", "count": 6, "k": 1, "edge_cases": ["recovery", "conflicting_state"]},
    )

    assert resp.status_code == 202
    batch = client.get(f"/api/task-factory/batches/{resp.json()['batch_id']}").json()
    assert batch["edge_cases"] == ["recovery", "conflicting_state"]


def test_a_batch_without_edge_cases_records_none(client, queued):
    resp = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": 6, "k": 1})

    batch = client.get(f"/api/task-factory/batches/{resp.json()['batch_id']}").json()
    assert batch["edge_cases"] == []


def test_an_unknown_edge_case_is_refused_by_name(client, queued):
    resp = client.post(
        "/api/task-factory/batches",
        json={"env_name": "mail", "count": 6, "k": 1, "edge_cases": ["permission_errors"]},
    )

    assert resp.status_code == 422
    assert "permission_errors" in resp.json()["detail"]
    queued.delay.assert_not_called()


def test_an_edge_case_the_environment_family_lacks_is_refused(client, queued):
    with client.session_factory() as db:
        db.add(SandboxEnvironment(
            id="site", status="running", env_type="browser", container_id="c2",
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        ))
        db.commit()

    resp = client.post(
        "/api/task-factory/batches",
        json={"env_name": "site", "count": 6, "k": 1, "edge_cases": ["missing_deps"]},
    )

    assert resp.status_code == 422
    assert "missing_deps" in resp.json()["detail"] and "browser" in resp.json()["detail"]
    queued.delay.assert_not_called()


def test_environments_list_the_edge_cases_their_family_offers(client):
    body = {e["name"]: e for e in client.get("/api/task-factory/environments").json()}

    assert body["mail"]["edge_cases"] == ["boundary_conditions", "conflicting_state", "missing_deps", "recovery"]
    assert body["web"]["edge_cases"] == ["boundary_conditions", "conflicting_state", "recovery"]


def _judged_batch(client) -> str:
    """Two SFT tasks: t-00001 calibrated suspect, t-00002 solved."""
    from backend.app.services import calibration_registry
    from tests.backend.test_calibration_registry import _attempt, _task

    tasks = [_task(1), _task(2)]
    taxonomy = Taxonomy(categories=[TaxonomyCategory(name="triage", description="sort", exercises=[], difficulties=[1])])
    with client.session_factory() as db:
        batch_id = task_registry.create_batch(
            db, env_name="mail", requested=2, pass_k=3,
            writer=ModelSpec("anthropic", "claude-sonnet-5"), validator=ModelSpec("openai", "gpt-5"), data_type="sft",
        )
        task_registry.save_result(db, batch_id, PipelineResult(requested=2, taxonomy=taxonomy, tasks=tasks, data_type="sft"))
        run_id = calibration_registry.create_run(db, batch_id=batch_id, target_model="ollama:m", attempts=3)
        calibration_registry.record_attempts(db, run_id, tasks[0], [_attempt(False)] * 3)
        calibration_registry.record_attempts(db, run_id, tasks[1], [_attempt(True)] * 3)
    return batch_id


def test_a_person_judges_a_task_hard_or_unfair(client):
    batch_id = _judged_batch(client)

    resp = client.put(f"/api/task-factory/batches/{batch_id}/tasks/t-00001/judgment", json={"verdict": "unfair"})

    assert resp.status_code == 200
    assert resp.json() == {"task_id": "t-00001", "verdict": "unfair"}


def test_a_judgment_with_another_verdict_or_an_unknown_task_is_refused(client):
    batch_id = _judged_batch(client)

    bad = client.put(f"/api/task-factory/batches/{batch_id}/tasks/t-00001/judgment", json={"verdict": "fine"})
    missing = client.put(f"/api/task-factory/batches/{batch_id}/tasks/t-00099/judgment", json={"verdict": "hard"})
    no_batch = client.put("/api/task-factory/batches/tb_missing/tasks/t-00001/judgment", json={"verdict": "hard"})

    assert bad.status_code == 422
    assert missing.status_code == 404 and no_batch.status_code == 404


def test_an_export_holds_out_suspect_tasks_and_says_why(client):
    batch_id = _judged_batch(client)

    body = client.get(f"/api/task-factory/batches/{batch_id}/export").json()

    assert [task["id"] for task in body["tasks"]] == ["t-00002"]
    assert [item["metadata"]["task_id"] for item in body["sft_items"]] == ["t-00002"]
    assert [held["task_id"] for held in body["held_out"]] == ["t-00001"]
    assert "suspect" in body["held_out"][0]["reason"]


def test_marking_a_suspect_task_hard_puts_it_back_in_the_export(client):
    batch_id = _judged_batch(client)

    client.put(f"/api/task-factory/batches/{batch_id}/tasks/t-00001/judgment", json={"verdict": "hard"})
    body = client.get(f"/api/task-factory/batches/{batch_id}/export").json()

    assert [task["id"] for task in body["tasks"]] == ["t-00001", "t-00002"]
    assert body["held_out"] == []


def test_a_task_marked_unfair_leaves_the_export(client):
    batch_id = _judged_batch(client)

    client.put(f"/api/task-factory/batches/{batch_id}/tasks/t-00002/judgment", json={"verdict": "unfair"})
    body = client.get(f"/api/task-factory/batches/{batch_id}/export").json()

    assert body["tasks"] == []
    assert sorted(held["task_id"] for held in body["held_out"]) == ["t-00001", "t-00002"]


# Calibration exports: a run's attempts as training data, through the batch export route.

FILES = {
    "grpo": "grpo_rollouts.parquet", "preference": "preference_pairs.jsonl",
    "sft": "sft_pairs.jsonl", "failures": "failure_dataset.jsonl",
}


def _calibrated_batch(client, *, complete: bool = True) -> tuple[str, str]:
    """t-00001 is suspect, t-00002 is on the frontier, t-00003 is too easy."""
    from tests.backend.test_calibration_registry import _attempt, _task

    tasks = [_task(1), _task(2), _task(3)]
    taxonomy = Taxonomy(categories=[TaxonomyCategory(name="triage", description="sort", exercises=[], difficulties=[1])])
    with client.session_factory() as db:
        batch_id = task_registry.create_batch(
            db, env_name="mail", requested=3, pass_k=3,
            writer=ModelSpec("anthropic", "claude-sonnet-5"), validator=ModelSpec("openai", "gpt-5"),
        )
        task_registry.save_result(db, batch_id, PipelineResult(requested=3, taxonomy=taxonomy, tasks=tasks))
        run_id = calibration_registry.create_run(db, batch_id=batch_id, target_model="ollama:m", attempts=3)
        calibration_registry.record_attempts(db, run_id, tasks[0], [_attempt(False)] * 3)
        calibration_registry.record_attempts(db, run_id, tasks[1], [_attempt(True), _attempt(False), _attempt(False)])
        calibration_registry.record_attempts(db, run_id, tasks[2], [_attempt(True)] * 3)
        if complete:
            calibration_registry.complete_run(db, run_id)
    return batch_id, run_id


def _export(client, batch_id: str, run_id: str, fmt: str):
    return client.get(f"/api/task-factory/batches/{batch_id}/export", params={"calibration": run_id, "format": fmt})


def _bundle(resp) -> tuple[dict, list[dict]]:
    """The manifest and the dataset rows of an exported zip."""
    archive = zipfile.ZipFile(io.BytesIO(resp.content))
    manifest = json.loads(archive.read("dataset_manifest.json"))
    (name,) = manifest["files"]
    data = archive.read(name)
    if name.endswith(".parquet"):
        return manifest, pd.read_parquet(io.BytesIO(data)).to_dict("records")
    return manifest, [json.loads(line) for line in data.decode().splitlines()]


@pytest.mark.parametrize("fmt", sorted(FILES))
def test_each_format_downloads_as_a_zip_with_its_file_and_a_manifest(client, fmt):
    batch_id, run_id = _calibrated_batch(client)

    resp = _export(client, batch_id, run_id, fmt)

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert f"{fmt}" in resp.headers["content-disposition"]
    names = zipfile.ZipFile(io.BytesIO(resp.content)).namelist()
    assert sorted(names) == sorted(["dataset_manifest.json", FILES[fmt]])


def test_grpo_holds_one_group_for_the_frontier_task(client):
    batch_id, run_id = _calibrated_batch(client)

    _, rows = _bundle(_export(client, batch_id, run_id, "grpo"))

    assert [(row["task_name"], row["total_reward"]) for row in rows] == [
        ("t-00002", 1.0), ("t-00002", 0.0), ("t-00002", 0.0),
    ]


def test_the_manifest_names_the_run_the_model_and_the_band(client):
    batch_id, run_id = _calibrated_batch(client)

    manifest, rows = _bundle(_export(client, batch_id, run_id, "preference"))

    assert manifest["source"] == "model_rollouts"
    assert manifest["calibration_run_id"] == run_id
    assert manifest["target_model"] == "ollama:m"
    assert manifest["attempts_per_task"] == 3
    assert manifest["frontier"] == {"min": FRONTIER_MIN, "max": FRONTIER_MAX}
    assert manifest["format"] == "preference" and manifest["batch_id"] == batch_id
    assert manifest["rows"] == len(rows) == 2
    assert manifest["safety"]["rows_seen"] == 2


def test_a_suspect_task_stays_out_until_a_person_marks_it_hard(client):
    batch_id, run_id = _calibrated_batch(client)

    before, before_rows = _bundle(_export(client, batch_id, run_id, "failures"))
    client.put(f"/api/task-factory/batches/{batch_id}/tasks/t-00001/judgment", json={"verdict": "hard"})
    after, after_rows = _bundle(_export(client, batch_id, run_id, "failures"))

    assert {row["task_id"] for row in before_rows} == {"t-00002"}
    assert [held["task_id"] for held in before["held_out"]] == ["t-00001"]
    assert {row["task_id"] for row in after_rows} == {"t-00001", "t-00002"}
    assert after["held_out"] == []


@pytest.mark.parametrize("fmt", sorted(FILES))
def test_a_task_marked_unfair_is_missing_from_every_format(client, fmt):
    batch_id, run_id = _calibrated_batch(client)
    client.put(f"/api/task-factory/batches/{batch_id}/tasks/t-00002/judgment", json={"verdict": "unfair"})

    manifest, rows = _bundle(_export(client, batch_id, run_id, fmt))

    assert all("t-00002" not in json.dumps(row) for row in rows)
    assert "t-00002" in {held["task_id"] for held in manifest["held_out"]}


def test_a_run_that_has_not_finished_cannot_be_exported(client):
    batch_id, run_id = _calibrated_batch(client, complete=False)

    resp = _export(client, batch_id, run_id, "grpo")

    assert resp.status_code == 409
    assert "complete" in resp.json()["detail"]


def test_a_run_of_another_batch_or_an_unknown_run_is_not_found(client):
    batch_id, _ = _calibrated_batch(client)
    _, other_run = _calibrated_batch(client)

    assert _export(client, batch_id, other_run, "grpo").status_code == 404
    assert _export(client, batch_id, "cr_missing", "grpo").status_code == 404


def test_an_unknown_or_missing_format_is_refused(client):
    batch_id, run_id = _calibrated_batch(client)

    unknown = _export(client, batch_id, run_id, "dpo")
    missing = client.get(f"/api/task-factory/batches/{batch_id}/export", params={"calibration": run_id})

    assert unknown.status_code == 422 and "grpo, preference, sft, failures" in unknown.json()["detail"]
    assert missing.status_code == 422


def test_the_batch_export_without_a_calibration_is_unchanged(client):
    batch_id, _ = _calibrated_batch(client)

    resp = client.get(f"/api/task-factory/batches/{batch_id}/export")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    assert [task["id"] for task in resp.json()["tasks"]] == ["t-00002", "t-00003"]
