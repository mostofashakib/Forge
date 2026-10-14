"""In production, the API and worker refuse to start with a missing or placeholder setting."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.app.production_checks import problems

ROOT = Path(__file__).resolve().parents[2]

COMPLETE = {
    "FORGE_ENV": "production",
    "FORGE_SCRUB_KEY": "a" * 64,
    "REDIS_URL": "redis://:b3f1c2@127.0.0.1:6379/0",
    "CORS_ORIGINS": "https://forge.example.ts.net",
    "FORGE_SANDBOX_HOST": "127.0.0.1",
    "FORGE_DB_URL": "sqlite:////data/forge.db",
    "FORGE_LLM_PROVIDER": "anthropic",
    "ANTHROPIC_API_KEY": "sk-ant-real",
}


def test_a_complete_production_environment_has_no_problems():
    assert problems(COMPLETE) == []


@pytest.mark.parametrize("name", ["FORGE_SCRUB_KEY", "REDIS_URL", "CORS_ORIGINS", "FORGE_SANDBOX_HOST", "FORGE_DB_URL"])
def test_each_missing_setting_is_named(name):
    env = {key: value for key, value in COMPLETE.items() if key != name}

    found = problems(env)

    assert len(found) == 1
    assert name in found[0]


@pytest.mark.parametrize("name", ["FORGE_SCRUB_KEY", "FORGE_DB_URL"])
def test_a_placeholder_counts_as_missing(name):
    assert name in " ".join(problems({**COMPLETE, name: "mock_key"}))


def test_a_blank_value_counts_as_missing():
    assert "CORS_ORIGINS" in " ".join(problems({**COMPLETE, "CORS_ORIGINS": "  "}))


def test_redis_without_a_password_is_rejected():
    found = problems({**COMPLETE, "REDIS_URL": "redis://127.0.0.1:6379/0"})

    assert found == ["REDIS_URL has no password. Put FORGE_REDIS_PASSWORD in it: redis://:<password>@host:6379/0"]


def test_a_hosted_provider_without_its_key_is_named():
    found = problems({**COMPLETE, "ANTHROPIC_API_KEY": "mock_key"})

    assert len(found) == 1
    assert "ANTHROPIC_API_KEY" in found[0]


def test_ollama_needs_no_key():
    env = {**COMPLETE, "FORGE_LLM_PROVIDER": "ollama"}
    del env["ANTHROPIC_API_KEY"]

    assert problems(env) == []


def test_an_unknown_provider_is_rejected():
    assert "FORGE_LLM_PROVIDER" in " ".join(problems({**COMPLETE, "FORGE_LLM_PROVIDER": "mystery"}))


@pytest.mark.parametrize("origins", ["http://localhost:3000", "https://forge.example.ts.net,http://127.0.0.1:3000"])
def test_a_loopback_cors_origin_is_rejected(origins):
    assert "CORS_ORIGINS" in " ".join(problems({**COMPLETE, "CORS_ORIGINS": origins}))


def test_problems_never_print_a_secret_value():
    found = " ".join(problems({**COMPLETE, "ANTHROPIC_API_KEY": "mock_key", "REDIS_URL": "redis://127.0.0.1:6379/0"}))

    assert "a" * 64 not in found
    assert "sk-ant-real" not in found


def _import(module: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp")}
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=ROOT, env={**base, **env}, capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("module", ["backend.app.main", "backend.app.worker.celery_app"])
def test_the_api_and_worker_exit_1_and_name_a_missing_setting(module):
    # Blank, not absent: main.py loads backend/.env, which never overrides a set variable.
    env = {**COMPLETE, "FORGE_SCRUB_KEY": ""}

    result = _import(module, env)

    assert result.returncode == 1
    assert "FORGE_SCRUB_KEY" in result.stderr


@pytest.mark.parametrize("module", ["backend.app.main", "backend.app.worker.celery_app"])
def test_the_api_and_worker_start_when_production_is_complete(module):
    assert _import(module, COMPLETE).returncode == 0


def test_development_starts_without_any_production_setting():
    assert _import("backend.app.main", {}).returncode == 0
