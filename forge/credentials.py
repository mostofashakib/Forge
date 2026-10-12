"""Every credential Forge reads, and the one place that reads them.

`.env.example` sets each credential to `PLACEHOLDER`. A placeholder or blank
value counts as not configured, so a feature that needs it refuses to run and
names the variable. Callers pass values to SDKs explicitly, so no library
falls back to reading a placeholder from the environment on its own.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

PLACEHOLDER = "mock_key"


@dataclass(frozen=True)
class Credential:
    """One secret. The first name is canonical, the rest are accepted aliases."""

    env_vars: tuple[str, ...]
    purpose: str


ANTHROPIC_API_KEY = Credential(("ANTHROPIC_API_KEY",), "the Anthropic model provider")
OPENAI_API_KEY = Credential(("OPENAI_API_KEY",), "the OpenAI model provider")
GEMINI_API_KEY = Credential(("GEMINI_API_KEY", "GOOGLE_API_KEY"), "the Gemini model provider")
GITHUB_TOKEN = Credential(
    ("GITHUB_TOKEN",), "GitHub repo discovery and downloads (public read access is enough)"
)
FORGE_SCRUB_KEY = Credential(
    ("FORGE_SCRUB_KEY",), "keying PII pseudonyms in internet and human data"
)

CREDENTIALS = (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GITHUB_TOKEN, FORGE_SCRUB_KEY)

PROVIDER_CREDENTIALS: dict[str, Credential] = {
    "anthropic": ANTHROPIC_API_KEY,
    "openai": OPENAI_API_KEY,
    "gemini": GEMINI_API_KEY,
}


class CredentialNotConfigured(RuntimeError):
    """A feature needs a credential that is unset, blank, or still the placeholder."""


def credential_value(credential: Credential, env: Mapping[str, str] | None = None) -> str | None:
    """The first configured value among the credential's names, or None."""
    source = os.environ if env is None else env
    for name in credential.env_vars:
        value = (source.get(name) or "").strip()
        if value and value != PLACEHOLDER:
            return value
    return None


def is_configured(credential: Credential, env: Mapping[str, str] | None = None) -> bool:
    return credential_value(credential, env) is not None


def require_credential(credential: Credential, env: Mapping[str, str] | None = None) -> str:
    value = credential_value(credential, env)
    if value is None:
        names = " or ".join(credential.env_vars)
        raise CredentialNotConfigured(
            f"{names} is not configured. Set it in backend/.env. Forge needs it for {credential.purpose}."
        )
    return value
