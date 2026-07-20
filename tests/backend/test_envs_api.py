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


def _insert_sandbox(env_name, status, expires_at):
    from backend.app import database
    from backend.app.models import SandboxEnvironment
    db = database.get_session_factory()()
    try:
        db.add(SandboxEnvironment(id=env_name, status=status, ttl_days=30, expires_at=expires_at))
        db.commit()
    finally:
        db.close()


def test_list_envs_excludes_deleted_and_expired_sandbox_dirs(client, tmp_path, monkeypatch):
    """A generated env directory whose sandbox is expired/time-expired must not
    appear in inventory; an orphan dir with no sandbox row still shows."""
    from datetime import datetime, timezone, timedelta
    envs_dir = tmp_path / "generated_envs"
    for name in ("active_env", "expired_env", "lapsed_env", "orphan_env"):
        (envs_dir / name).mkdir(parents=True)
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(envs_dir))

    future = datetime.now(timezone.utc) + timedelta(days=1)
    past = datetime.now(timezone.utc) - timedelta(days=1)
    _insert_sandbox("active_env", "running", future)
    _insert_sandbox("expired_env", "expired", past)
    _insert_sandbox("lapsed_env", "running", past)  # time-expired, not yet swept

    names = client.get("/api/envs/").json()
    assert "active_env" in names
    assert "orphan_env" in names  # compiled but never sandboxed — still valid inventory
    assert "expired_env" not in names
    assert "lapsed_env" not in names


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


def test_get_config_rejects_path_traversal(client):
    response = client.get("/api/envs/../etc/passwd/config")
    # FastAPI will either reject the path or return 400/404
    assert response.status_code in (400, 404, 422)


# ---------------------------------------------------------------------------
# Source-bundle download
# ---------------------------------------------------------------------------
import io
import zipfile


def _make_downloadable_env(envs_dir: Path, name: str) -> None:
    d = envs_dir / name
    (d / "app").mkdir(parents=True)
    (d / "app" / "main.py").write_text("app = object()\n")
    (d / "app" / "requirements.txt").write_text("fastapi\n")
    (d / "app" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (d / "reward_fn.py").write_text("def compute_reward(*a): return 0.0\n")


def test_download_env_source_returns_zip(client, tmp_path):
    _make_downloadable_env(tmp_path / "generated_envs", "dl_env")
    resp = client.get("/api/envs/dl_env/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert 'filename="dl_env.zip"' in resp.headers["content-disposition"]
    names = set(zipfile.ZipFile(io.BytesIO(resp.content)).namelist())
    assert "dl_env/app/main.py" in names
    assert "dl_env/README.md" in names
    assert "dl_env/docker-compose.yml" in names


def test_download_missing_env_returns_404(client):
    resp = client.get("/api/envs/does_not_exist/download")
    assert resp.status_code == 404


def test_download_incomplete_env_returns_404(client, tmp_path):
    # An env dir with no app/main.py cannot be bundled into a runnable package.
    (tmp_path / "generated_envs" / "empty_env" / "app").mkdir(parents=True)
    resp = client.get("/api/envs/empty_env/download")
    assert resp.status_code == 404


def test_download_rejects_traversal_name(client):
    resp = client.get("/api/envs/foo..bar/download")
    assert resp.status_code == 400
