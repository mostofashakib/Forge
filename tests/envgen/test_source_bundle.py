from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest

from forge.envgen.source_bundle import SourceBundleError, build_source_bundle


def _make_env(root: Path, name: str = "my_env") -> Path:
    d = root / name
    (d / "app").mkdir(parents=True)
    (d / "app" / "main.py").write_text("app = object()\n")
    (d / "app" / "ui.html").write_text("<html></html>")
    (d / "app" / "requirements.txt").write_text("fastapi\n")
    (d / "app" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (d / "container_env.py").write_text("class ContainerForgeEnv: pass\n")
    (d / "reward_fn.py").write_text("def compute_reward(*a): return 0.0\n")
    (d / "custom").mkdir()
    (d / "custom" / "policies.yaml").write_text("policies: []\n")
    (d / "state_schema.json").write_text('{"fields": {}}')
    return d


def _names(data: bytes) -> set[str]:
    return set(zipfile.ZipFile(io.BytesIO(data)).namelist())


def _read(data: bytes, name: str) -> str:
    return zipfile.ZipFile(io.BytesIO(data)).read(name).decode()


def test_bundle_contains_full_source_readme_and_compose(tmp_path):
    env = _make_env(tmp_path)
    names = _names(build_source_bundle(env, "my_env"))
    assert {
        "my_env/app/main.py",
        "my_env/app/ui.html",
        "my_env/app/requirements.txt",
        "my_env/app/Dockerfile",
        "my_env/container_env.py",
        "my_env/reward_fn.py",
        "my_env/custom/policies.yaml",
        "my_env/state_schema.json",
        "my_env/README.md",
        "my_env/docker-compose.yml",
    } <= names


def test_readme_documents_local_and_docker_run(tmp_path):
    readme = _read(build_source_bundle(_make_env(tmp_path), "my_env"), "my_env/README.md")
    assert "REDIS_URL" in readme
    assert "uvicorn" in readme
    assert "8000" in readme
    assert "docker compose" in readme.lower()


def test_compose_defines_app_and_redis_services(tmp_path):
    compose = _read(
        build_source_bundle(_make_env(tmp_path), "my_env"), "my_env/docker-compose.yml"
    )
    assert "redis" in compose
    assert "8000" in compose
    assert "REDIS_URL" in compose


def test_bundle_includes_a_centralized_run_script(tmp_path):
    data = build_source_bundle(_make_env(tmp_path), "my_env")
    script = _read(data, "my_env/run.sh")
    # One script that covers every path to run it locally.
    assert "docker compose" in script.lower()          # docker path
    assert "requirements.txt" in script                 # install dependencies
    assert "uvicorn" in script                          # serve the app
    assert "REDIS_URL" in script                        # telemetry backend


def test_run_script_is_marked_executable(tmp_path):
    data = build_source_bundle(_make_env(tmp_path), "my_env")
    info = zipfile.ZipFile(io.BytesIO(data)).getinfo("my_env/run.sh")
    mode = info.external_attr >> 16
    assert mode & 0o111, "run.sh must carry an executable permission bit"


def test_readme_points_at_the_run_script(tmp_path):
    readme = _read(build_source_bundle(_make_env(tmp_path), "my_env"), "my_env/README.md")
    assert "run.sh" in readme


def test_bundle_excludes_runtime_artifacts(tmp_path):
    env = _make_env(tmp_path)
    (env / "episodes").mkdir()
    (env / "episodes" / "ep1.jsonl").write_text("{}")
    (env / "episodes" / "ep1.trace.jsonl").write_text("{}")
    (env / "app" / "__pycache__").mkdir()
    (env / "app" / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"x")

    names = _names(build_source_bundle(env, "my_env"))
    assert not any("episodes" in n for n in names)
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)


def test_missing_environment_raises(tmp_path):
    with pytest.raises(SourceBundleError, match="not found"):
        build_source_bundle(tmp_path / "nope", "nope")


def test_incomplete_environment_raises(tmp_path):
    d = tmp_path / "broken"
    (d / "app").mkdir(parents=True)
    (d / "app" / "ui.html").write_text("<html></html>")  # no main.py
    with pytest.raises(SourceBundleError, match="incomplete|main.py"):
        build_source_bundle(d, "broken")


def test_bundle_does_not_follow_symlinks_pointing_outside_the_env(tmp_path):
    env = _make_env(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("SUPER_SECRET")
    link = env / "app" / "leak.txt"
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported on this platform")

    data = build_source_bundle(env, "my_env")
    assert "my_env/app/leak.txt" not in _names(data)
    assert "SUPER_SECRET" not in data.decode("latin-1")


def test_bundle_arcnames_never_escape_the_env_root(tmp_path):
    env = _make_env(tmp_path)
    for name in _names(build_source_bundle(env, "my_env")):
        assert not name.startswith("/")
        assert ".." not in name.split("/")
        assert name.startswith("my_env/")
