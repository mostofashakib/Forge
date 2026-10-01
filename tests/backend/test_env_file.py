"""Rewriting backend/.env from the settings page.

Only the named keys change. Every other line, comment, and secret stays
byte for byte, and the file is swapped in whole, never half written.
"""
from __future__ import annotations

import os
import stat

import pytest

from backend.app.services.env_file import read_env_values, update_env_file

ORIGINAL = (
    "# keys\n"
    "ANTHROPIC_API_KEY=sk-ant-secret\n"
    "FORGE_TASK_VALIDATOR_PROVIDER=gemini\n"
    "\n"
    "export FORGE_LLM_PROVIDER=anthropic   # inline comment\n"
)


@pytest.fixture
def env_path(tmp_path):
    path = tmp_path / ".env"
    path.write_text(ORIGINAL)
    os.chmod(path, 0o600)
    return path


def test_an_existing_key_is_replaced_in_place(env_path):
    update_env_file(env_path, {"FORGE_TASK_VALIDATOR_PROVIDER": "openai"})

    assert env_path.read_text() == ORIGINAL.replace("=gemini", "=openai")


def test_a_missing_key_is_appended(env_path):
    update_env_file(env_path, {"FORGE_TASK_VALIDATOR_MODEL": "gpt-5"})

    assert env_path.read_text() == ORIGINAL + "FORGE_TASK_VALIDATOR_MODEL=gpt-5\n"


def test_secrets_and_comments_are_untouched(env_path):
    update_env_file(env_path, {"FORGE_TASK_VALIDATOR_PROVIDER": "openai", "FORGE_TASK_VALIDATOR_MODEL": "gpt-5"})

    text = env_path.read_text()
    assert "ANTHROPIC_API_KEY=sk-ant-secret\n" in text
    assert "# keys\n" in text
    assert "export FORGE_LLM_PROVIDER=anthropic   # inline comment\n" in text


def test_the_file_mode_is_kept(env_path):
    update_env_file(env_path, {"FORGE_TASK_VALIDATOR_MODEL": "gpt-5"})

    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


def test_a_file_that_does_not_exist_is_created(tmp_path):
    path = tmp_path / ".env"

    update_env_file(path, {"FORGE_TASK_VALIDATOR_MODEL": "gpt-5"})

    assert path.read_text() == "FORGE_TASK_VALIDATOR_MODEL=gpt-5\n"


def test_a_file_without_a_trailing_newline_gets_one_before_the_append(tmp_path):
    path = tmp_path / ".env"
    path.write_text("A=1")

    update_env_file(path, {"B": "2"})

    assert path.read_text() == "A=1\nB=2\n"


@pytest.mark.parametrize("value", ["gpt-5\nANTHROPIC_API_KEY=stolen", "a\rb", "has space", "quote\"d", "#hash"])
def test_a_value_that_could_inject_lines_or_break_parsing_is_rejected(env_path, value):
    with pytest.raises(ValueError):
        update_env_file(env_path, {"FORGE_TASK_VALIDATOR_MODEL": value})

    assert env_path.read_text() == ORIGINAL


def test_an_invalid_key_is_rejected(env_path):
    with pytest.raises(ValueError):
        update_env_file(env_path, {"BAD KEY": "x"})


def test_reading_returns_only_the_asked_keys_that_are_present(env_path):
    values = read_env_values(env_path, ["FORGE_TASK_VALIDATOR_PROVIDER", "FORGE_TASK_VALIDATOR_MODEL", "FORGE_LLM_PROVIDER"])

    assert values == {"FORGE_TASK_VALIDATOR_PROVIDER": "gemini", "FORGE_LLM_PROVIDER": "anthropic"}


def test_reading_a_missing_file_returns_nothing(tmp_path):
    assert read_env_values(tmp_path / ".env", ["A"]) == {}
