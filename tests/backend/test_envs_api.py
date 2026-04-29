import tempfile
from pathlib import Path
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


@pytest.fixture
def env_with_config(tmp_path, monkeypatch):
    envs_dir = tmp_path / "generated_envs"
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(envs_dir))
    pkg_dir = envs_dir / "my_env" / "custom"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "config.yaml").write_text("reward:\n  base_success: 2.0\n")
    return envs_dir


def test_list_envs_returns_empty_when_no_envs(client, tmp_path, monkeypatch):
    envs_dir = tmp_path / "generated_envs"
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(envs_dir))
    response = client.get("/api/envs/")
    assert response.status_code == 200
    assert response.json() == []


def test_list_envs_returns_env_names(client, tmp_path, monkeypatch):
    envs_dir = tmp_path / "generated_envs"
    (envs_dir / "my_env").mkdir(parents=True)
    (envs_dir / "other_env").mkdir(parents=True)
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(envs_dir))
    response = client.get("/api/envs/")
    assert response.status_code == 200
    names = response.json()
    assert "my_env" in names
    assert "other_env" in names


def test_get_config_returns_yaml(client, tmp_path, monkeypatch, env_with_config):
    response = client.get("/api/envs/my_env/config")
    assert response.status_code == 200
    data = response.json()
    assert "yaml" in data
    assert "base_success" in data["yaml"]


def test_get_config_returns_404_for_unknown_env(client, tmp_path, monkeypatch):
    envs_dir = tmp_path / "generated_envs"
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(envs_dir))
    response = client.get("/api/envs/nonexistent/config")
    assert response.status_code == 404


def test_put_config_writes_yaml(client, tmp_path, monkeypatch, env_with_config):
    new_yaml = "reward:\n  base_success: 3.0\n"
    response = client.put("/api/envs/my_env/config", json={"yaml": new_yaml})
    assert response.status_code == 200
    config_path = env_with_config / "my_env" / "custom" / "config.yaml"
    assert config_path.read_text() == new_yaml
