"""GET /api/health: one answer to whether every dependency Forge relies on works."""
from __future__ import annotations

import os
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api import health
from backend.app.main import app
from backend.app.services.ollama_catalog import OllamaUnavailable

ALL_OK = {name: (lambda: "ok") for name in ("database", "redis", "docker", "worker", "llm")}


@pytest.fixture
def checks(monkeypatch):
    """Every check passes unless a test replaces it."""
    replaced = dict(ALL_OK, backup=lambda: "disabled")
    monkeypatch.setattr(health, "CHECKS", replaced)
    return replaced


@pytest.fixture
def client():
    return TestClient(app)


def _fail(message: str):
    def check():
        raise RuntimeError(message)
    return check


def test_all_checks_up_returns_200(checks, client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checks": {"database": "ok", "redis": "ok", "docker": "ok", "worker": "ok", "llm": "ok", "backup": "disabled"},
    }


@pytest.mark.parametrize("name", ["database", "redis", "docker", "worker", "llm", "backup"])
def test_one_failing_check_returns_503_and_names_it(checks, client, name):
    checks[name] = _fail(f"{name} is down")

    response = client.get("/api/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert "is down" in body["checks"][name]
    assert all(value in ("ok", "disabled") for key, value in body["checks"].items() if key != name)


def test_a_hanging_check_times_out_without_delaying_the_others(checks, monkeypatch):
    monkeypatch.setattr(health, "CHECK_TIMEOUT_S", 0.2)
    checks["docker"] = lambda: time.sleep(1.5) or "ok"

    # A live server keeps its event loop running, so open the client once and
    # time only the request, not the loop shutdown that waits for the hung thread.
    # The bare router skips Forge's startup, which would touch forge.db and Docker.
    bare = FastAPI()
    bare.include_router(health.router)
    with TestClient(bare) as client:
        started = time.monotonic()
        response = client.get("/api/health")
        elapsed = time.monotonic() - started

    assert response.status_code == 503
    assert response.json()["checks"]["docker"] == "timed out after 0.2 s"
    assert response.json()["checks"]["redis"] == "ok"
    assert elapsed < 1.0


def test_backup_reports_disabled_without_a_litestream_dir(monkeypatch):
    monkeypatch.delenv("FORGE_LITESTREAM_DIR", raising=False)

    assert health.check_backup() == "disabled"


def test_error_text_never_carries_a_connection_secret(checks, client):
    checks["redis"] = _fail("Error connecting to redis://:hunter2secret@127.0.0.1:6379/0")

    body = client.get("/api/health").text

    assert "hunter2secret" not in body
    assert "redis://" in body  # the scheme stays, so the error is still readable


def test_redis_check_reports_a_refused_connection(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://:topsecretvalue@127.0.0.1:1/0")

    with pytest.raises(Exception) as caught:
        health.check_redis()

    assert "topsecretvalue" not in health.describe_error(caught.value)


def test_database_check_runs_select_1(monkeypatch, tmp_path):
    from backend.app import database

    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path / 'health.db'}")
    monkeypatch.setattr(database, "_engine", None)

    assert health.check_database() == "ok"
    monkeypatch.setattr(database, "_engine", None)


class TestLlmCheck:
    def test_hosted_provider_without_a_key_fails_and_names_the_variable(self, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "mock_key")

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            health.check_llm()

    def test_hosted_provider_with_a_key_passes_without_calling_it(self, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-looking")

        assert health.check_llm() == "ok"

    def test_ollama_passes_when_every_generation_model_is_pulled(self, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("FORGE_LLM_MODEL", "llama3.1:8b")
        monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "llama3.1:8b")
        monkeypatch.setattr(health, "list_ollama_models", lambda base_url: [{"name": "llama3.1:8b"}])

        assert health.check_llm() == "ok"

    def test_ollama_fails_and_names_a_model_that_is_not_pulled(self, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("FORGE_LLM_MODEL", "llama3.1:8b")
        monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "qwen3:32b")
        monkeypatch.setattr(health, "list_ollama_models", lambda base_url: [{"name": "llama3.1:8b"}])

        with pytest.raises(RuntimeError, match="qwen3:32b"):
            health.check_llm()

    def test_ollama_unreachable_reports_the_connection_error(self, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_PROVIDER", "ollama")

        def unreachable(base_url):
            raise OllamaUnavailable(f"could not list models from {base_url}/api/tags: refused")

        monkeypatch.setattr(health, "list_ollama_models", unreachable)

        with pytest.raises(OllamaUnavailable, match="refused"):
            health.check_llm()


class TestBackupCheck:
    """Litestream lag: how long the database has had writes Litestream has not copied."""

    @pytest.fixture
    def layout(self, monkeypatch, tmp_path):
        db = tmp_path / "forge.db"
        db.write_text("")
        wal = tmp_path / "forge.db-wal"
        wal.write_text("")
        meta = tmp_path / ".forge.db-litestream"
        (meta / "segments").mkdir(parents=True)
        monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{db}")
        monkeypatch.setenv("FORGE_LITESTREAM_DIR", str(meta))
        return wal, meta / "segments"

    @staticmethod
    def _age(path, seconds_ago):
        stamp = time.time() - seconds_ago
        os.utime(path, (stamp, stamp))

    def test_passes_when_litestream_has_copied_the_latest_write(self, layout):
        wal, segments = layout
        segment = segments / "0001.wal"
        segment.write_text("x")
        self._age(wal, 30)
        self._age(segment, 5)

        assert health.check_backup() == "ok"

    def test_passes_on_an_idle_database_with_an_old_segment(self, layout):
        wal, segments = layout
        segment = segments / "0001.wal"
        segment.write_text("x")
        self._age(wal, 7200)
        self._age(segment, 7000)

        assert health.check_backup() == "ok"

    def test_fails_when_writes_have_gone_uncopied_for_over_10_minutes(self, layout):
        wal, segments = layout
        segment = segments / "0001.wal"
        segment.write_text("x")
        self._age(segment, 1800)
        self._age(wal, 60)

        with pytest.raises(RuntimeError, match="not copied"):
            health.check_backup()

    def test_fails_when_litestream_has_written_nothing(self, layout):
        with pytest.raises(RuntimeError, match="no Litestream files"):
            health.check_backup()
