"""The settings API: shows every platform setting and saves changes to backend/.env."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import env_file


VALIDATOR = {"FORGE_TASK_VALIDATOR_PROVIDER": "openai", "FORGE_TASK_VALIDATOR_MODEL": "gpt-5"}


@pytest.fixture(autouse=True)
def sdks_installed(monkeypatch):
    import forge.taskfactory.model_settings as settings

    monkeypatch.setattr(settings, "_installed", lambda module: True)


@pytest.fixture(autouse=True)
def ollama_models(monkeypatch):
    """No real Ollama server in tests. Tests replace the list they need."""
    models: list[dict] = []
    monkeypatch.setattr("backend.app.api.settings.list_ollama_models", lambda base_url: list(models))
    return models


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("ANTHROPIC_API_KEY=sk-ant-secret\n")
    monkeypatch.setattr(env_file, "BACKEND_ENV_FILE", path)
    for var in ("FORGE_TASK_VALIDATOR_PROVIDER", "FORGE_TASK_VALIDATOR_MODEL", "OPENAI_API_KEY",
                "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "FORGE_SANDBOX_LIMIT",
                "FORGE_JUDGE_PROVIDER", "FORGE_JUDGE_MODEL", "FORGE_QUORUM_MODELS", "FORGE_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FORGE_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-5")
    client = TestClient(app)
    client.env_path = path
    return client


def test_settings_show_the_writer_and_an_unconfigured_validator(client):
    body = client.get("/api/settings").json()

    assert body["writer"] == {"provider": "anthropic", "model": "claude-sonnet-5", "family": "claude"}
    assert body["task_validator"]["configured"] is False
    assert "FORGE_TASK_VALIDATOR_PROVIDER" in body["task_validator"]["error"]


def test_key_status_is_reported_without_the_key(client):
    body = client.get("/api/settings").json()

    providers = {p["name"]: p for p in body["providers"]}
    assert providers["anthropic"]["key_set"] is True
    assert providers["openai"]["key_set"] is False
    assert providers["ollama"]["needs_key"] is False
    assert "sk-ant-secret" not in str(body)


def test_saving_the_validator_writes_only_those_lines(client):
    resp = client.put("/api/settings", json={"values": VALIDATOR})

    assert resp.status_code == 200
    assert resp.json()["task_validator"] == {
        "provider": "openai", "model": "gpt-5", "family": "gpt", "configured": True, "error": None,
    }
    assert client.env_path.read_text() == (
        "ANTHROPIC_API_KEY=sk-ant-secret\n"
        "FORGE_TASK_VALIDATOR_PROVIDER=openai\n"
        "FORGE_TASK_VALIDATOR_MODEL=gpt-5\n"
    )


def test_the_saved_file_wins_over_the_process_environment(client, monkeypatch):
    monkeypatch.setenv("FORGE_TASK_VALIDATOR_PROVIDER", "gemini")
    monkeypatch.setenv("FORGE_TASK_VALIDATOR_MODEL", "gemini-3-pro")

    client.put("/api/settings", json={"values": VALIDATOR})

    assert client.get("/api/settings").json()["task_validator"]["model"] == "gpt-5"


def test_a_validator_whose_sdk_is_missing_shows_as_not_configured(client, monkeypatch):
    import forge.taskfactory.model_settings as settings

    client.put("/api/settings", json={"values": VALIDATOR})
    monkeypatch.setattr(settings, "_installed", lambda module: module != "openai")

    validator = client.get("/api/settings").json()["task_validator"]
    assert validator["configured"] is False
    assert "openai" in validator["error"]


def test_a_validator_from_the_writers_family_is_refused(client):
    resp = client.put("/api/settings", json={"values": {**VALIDATOR, "FORGE_TASK_VALIDATOR_PROVIDER": "anthropic", "FORGE_TASK_VALIDATOR_MODEL": "claude-haiku-4-5"}})

    assert resp.status_code == 422
    assert "family" in resp.json()["detail"]
    assert "FORGE_TASK_VALIDATOR" not in client.env_path.read_text()


def test_an_unknown_provider_is_refused(client):
    resp = client.put("/api/settings", json={"values": {**VALIDATOR, "FORGE_TASK_VALIDATOR_PROVIDER": "mystery"}})

    assert resp.status_code == 422


def test_a_model_name_that_could_inject_a_line_is_refused(client):
    resp = client.put(
        "/api/settings", json={"values": {**VALIDATOR, "FORGE_TASK_VALIDATOR_MODEL": "gpt-5\nANTHROPIC_API_KEY=x"}}
    )

    assert resp.status_code == 422
    assert client.env_path.read_text() == "ANTHROPIC_API_KEY=sk-ant-secret\n"


def test_settings_list_the_pulled_ollama_models(client, ollama_models):
    ollama_models.append({"name": "qwen3:32b", "family": "qwen", "parameters": "32.8B", "cloud": False})

    ollama = client.get("/api/settings").json()["ollama"]

    assert ollama == {"models": [{"name": "qwen3:32b", "family": "qwen", "parameters": "32.8B", "cloud": False}], "error": None}


def test_an_unreachable_ollama_server_is_reported_not_raised(client, monkeypatch):
    from backend.app.services.ollama_catalog import OllamaUnavailable

    def down(base_url):
        raise OllamaUnavailable(f"could not list models from {base_url}/api/tags")

    monkeypatch.setattr("backend.app.api.settings.list_ollama_models", down)

    body = client.get("/api/settings")

    assert body.status_code == 200
    assert body.json()["ollama"]["models"] == []
    assert "api/tags" in body.json()["ollama"]["error"]


def _row(body, key):
    return next(row for row in body["settings"] if row["key"] == key)


def test_every_setting_is_listed_with_its_value_default_and_restart_rule(client, monkeypatch):
    monkeypatch.setenv("FORGE_SANDBOX_LIMIT", "4")

    body = client.get("/api/settings").json()

    sandbox = _row(body, "FORGE_SANDBOX_LIMIT")
    assert sandbox["value"] == "4"
    assert sandbox["default"] == "10"
    assert sandbox["group"] == "runtime"
    assert sandbox["live"] is False
    assert sandbox["pending_restart"] is False
    assert _row(body, "FORGE_TASK_VALIDATOR_MODEL")["live"] is True


def test_an_unset_setting_shows_its_default(client):
    assert _row(client.get("/api/settings").json(), "FORGE_SANDBOX_LIMIT")["value"] == "10"


def test_saving_a_restart_setting_marks_it_pending_until_restart(client, monkeypatch):
    monkeypatch.setenv("FORGE_SANDBOX_LIMIT", "4")

    body = client.put("/api/settings", json={"values": {"FORGE_SANDBOX_LIMIT": "16"}}).json()

    sandbox = _row(body, "FORGE_SANDBOX_LIMIT")
    assert sandbox["value"] == "16"
    assert sandbox["pending_restart"] is True
    assert "FORGE_SANDBOX_LIMIT=16\n" in client.env_path.read_text()


def test_a_saved_live_setting_is_never_pending(client):
    body = client.put("/api/settings", json={"values": VALIDATOR}).json()

    assert _row(body, "FORGE_TASK_VALIDATOR_MODEL")["pending_restart"] is False


def test_a_quorum_list_with_commas_is_saved(client):
    quorum = "openai:gpt-5,ollama:qwen3:32b"

    resp = client.put("/api/settings", json={"values": {"FORGE_QUORUM_MODELS": quorum}})

    assert resp.status_code == 200
    assert f"FORGE_QUORUM_MODELS={quorum}\n" in client.env_path.read_text()


def test_an_api_key_cannot_be_written_through_the_page(client):
    resp = client.put("/api/settings", json={"values": {"ANTHROPIC_API_KEY": "sk-ant-other"}})

    assert resp.status_code == 422
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]
    assert client.env_path.read_text() == "ANTHROPIC_API_KEY=sk-ant-secret\n"


def test_one_bad_value_saves_nothing(client):
    resp = client.put("/api/settings", json={"values": {"FORGE_SANDBOX_LIMIT": "16", "FORGE_CONTAINER_PIDS": "0"}})

    assert resp.status_code == 422
    assert "FORGE_CONTAINER_PIDS" in resp.json()["detail"]
    assert "FORGE_SANDBOX_LIMIT" not in client.env_path.read_text()


def test_a_judge_from_the_generator_family_is_refused(client):
    resp = client.put("/api/settings", json={"values": {"FORGE_JUDGE_MODEL": "claude-opus-5"}})

    assert resp.status_code == 422
    assert "FORGE_JUDGE_MODEL" not in client.env_path.read_text()


def test_a_saved_validator_blocks_moving_the_generator_into_its_family(client):
    client.put("/api/settings", json={"values": VALIDATOR})

    resp = client.put("/api/settings", json={"values": {
        "FORGE_LLM_PROVIDER": "openai", "FORGE_LLM_MODEL": "gpt-5-mini", "FORGE_LLM_MODEL_CAPABLE": "gpt-4o",
    }})

    assert resp.status_code == 422
    assert "FORGE_LLM_PROVIDER" not in client.env_path.read_text()


def test_the_ollama_list_uses_the_saved_server_url(client, monkeypatch):
    seen = []
    monkeypatch.setattr("backend.app.api.settings.list_ollama_models", lambda base_url: seen.append(base_url) or [])

    client.put("/api/settings", json={"values": {"OLLAMA_BASE_URL": "http://ollama:11434"}})

    assert seen[-1] == "http://ollama:11434"
