import pytest
from pathlib import Path
from forge.runtime.network_isolation import check_file, check_generated_env, NetworkIsolationViolation


def write_py(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(content)
    return p


def test_clean_file_has_no_violations(tmp_path):
    p = write_py(tmp_path, "clean.py", "x = 1\nprint(x)\n")
    assert check_file(p) == []


def test_import_requests_raises(tmp_path):
    p = write_py(tmp_path, "bad.py", "import requests\n")
    violations = check_file(p)
    assert len(violations) == 1
    assert violations[0].module == "requests"


def test_from_requests_import_get_raises(tmp_path):
    p = write_py(tmp_path, "bad.py", "from requests import get\n")
    violations = check_file(p)
    assert len(violations) == 1
    assert violations[0].module == "requests"


def test_import_httpx_raises(tmp_path):
    p = write_py(tmp_path, "bad.py", "import httpx\n")
    assert len(check_file(p)) == 1


def test_import_socket_raises(tmp_path):
    p = write_py(tmp_path, "bad.py", "import socket\n")
    assert len(check_file(p)) == 1


def test_import_urllib_raises(tmp_path):
    p = write_py(tmp_path, "bad.py", "from urllib.request import urlopen\n")
    assert len(check_file(p)) == 1


def test_forge_dev_network_skips_check(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DEV_NETWORK", "true")
    p = write_py(tmp_path, "bad.py", "import requests\n")
    assert check_file(p) == []


def test_check_generated_env_finds_violations(tmp_path, monkeypatch):
    monkeypatch.delenv("FORGE_DEV_NETWORK", raising=False)
    env_dir = tmp_path / "my_env"
    env_dir.mkdir()
    (env_dir / "transitions.py").write_text("import requests\nfrom httpx import get\n")
    (env_dir / "clean.py").write_text("x = 1\n")
    violations = check_generated_env(env_dir)
    assert len(violations) == 2
    modules = [v.module for v in violations]
    assert "requests" in modules
    assert "httpx" in modules


def test_check_generated_env_clean(tmp_path, monkeypatch):
    monkeypatch.delenv("FORGE_DEV_NETWORK", raising=False)
    env_dir = tmp_path / "clean_env"
    env_dir.mkdir()
    (env_dir / "transitions.py").write_text("def apply(state, action): return state\n")
    assert check_generated_env(env_dir) == []
