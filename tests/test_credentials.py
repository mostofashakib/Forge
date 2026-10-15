"""Every credential Forge needs is declared and checked in one place."""
from __future__ import annotations

import pytest

from forge.credentials import (
    ANTHROPIC_API_KEY,
    CREDENTIALS,
    FORGE_SCRUB_KEY,
    GEMINI_API_KEY,
    GITHUB_TOKEN,
    PLACEHOLDER,
    PROVIDER_CREDENTIALS,
    CredentialNotConfigured,
    credential_value,
    is_configured,
    require_credential,
)


def test_a_set_value_is_returned_without_surrounding_whitespace():
    assert credential_value(GITHUB_TOKEN, {"GITHUB_TOKEN": "  ghp_real  "}) == "ghp_real"


@pytest.mark.parametrize("value", ["", "   ", PLACEHOLDER, f" {PLACEHOLDER} "])
def test_blank_and_placeholder_values_are_not_configured(value):
    env = {"ANTHROPIC_API_KEY": value}

    assert credential_value(ANTHROPIC_API_KEY, env) is None
    assert not is_configured(ANTHROPIC_API_KEY, env)


def test_an_unset_credential_is_not_configured():
    assert not is_configured(FORGE_SCRUB_KEY, {})


def test_a_credential_falls_back_to_its_alias():
    env = {"GEMINI_API_KEY": PLACEHOLDER, "GOOGLE_API_KEY": "real-google-key"}

    assert credential_value(GEMINI_API_KEY, env) == "real-google-key"


def test_the_first_configured_name_wins():
    env = {"GEMINI_API_KEY": "gemini-key", "GOOGLE_API_KEY": "google-key"}

    assert credential_value(GEMINI_API_KEY, env) == "gemini-key"


def test_values_are_read_from_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("FORGE_SCRUB_KEY", "process-key")

    assert require_credential(FORGE_SCRUB_KEY) == "process-key"


def test_requiring_a_missing_credential_names_the_variable_and_its_purpose():
    with pytest.raises(CredentialNotConfigured) as caught:
        require_credential(GITHUB_TOKEN, {"GITHUB_TOKEN": PLACEHOLDER})

    message = str(caught.value)
    assert "GITHUB_TOKEN" in message
    assert GITHUB_TOKEN.purpose in message
    assert PLACEHOLDER not in message


def test_requiring_a_missing_aliased_credential_names_every_variable():
    with pytest.raises(CredentialNotConfigured, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
        require_credential(GEMINI_API_KEY, {})


def test_every_keyed_provider_maps_to_a_declared_credential():
    assert set(PROVIDER_CREDENTIALS) == {"anthropic", "openai", "gemini"}
    assert set(PROVIDER_CREDENTIALS.values()) <= set(CREDENTIALS)


def test_credential_names_are_unique():
    names = [var for credential in CREDENTIALS for var in credential.env_vars]

    assert len(names) == len(set(names))


class TestRedactUrlCredentials:
    def test_removes_the_password_and_user_from_a_url(self):
        from forge.credentials import redact_url_credentials

        assert redact_url_credentials("redis://:s3cret@127.0.0.1:6379/0") == "redis://***@127.0.0.1:6379/0"
        assert redact_url_credentials("postgres://forge:pw@db/forge") == "postgres://***@db/forge"

    def test_redacts_every_url_inside_free_text(self):
        from forge.credentials import redact_url_credentials

        text = "Error 111 connecting to redis://:a1@h:6379/0, fell back to amqp://u:b2@q:5672//"

        redacted = redact_url_credentials(text)

        assert "a1" not in redacted and "b2" not in redacted
        assert "redis://***@h:6379/0" in redacted

    def test_leaves_a_url_without_credentials_unchanged(self):
        from forge.credentials import redact_url_credentials

        assert redact_url_credentials("redis://127.0.0.1:6379/0") == "redis://127.0.0.1:6379/0"
        assert redact_url_credentials("not a url") == "not a url"
