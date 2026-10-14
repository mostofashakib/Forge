"""In production, refuse to start with a missing setting instead of half-working.

The API and the worker call `enforce_production_settings()` as they load. With
FORGE_ENV=production, each missing, blank, or placeholder setting prints one line
naming it, and the process exits with status 1. Outside production this does nothing.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from urllib.parse import urlparse

from forge.credentials import FORGE_SCRUB_KEY, PLACEHOLDER, PROVIDER_CREDENTIALS, is_configured

REQUIRED = {
    "REDIS_URL": "the Redis connection for the queue and live progress",
    "CORS_ORIGINS": "the browser origin allowed to call the API",
    "FORGE_SANDBOX_HOST": "the host sandbox ports are reached on",
    "FORGE_DB_URL": "the database location",
}
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _value(env: Mapping[str, str], name: str) -> str:
    value = (env.get(name) or "").strip()
    return "" if value == PLACEHOLDER else value


def problems(env: Mapping[str, str]) -> list[str]:
    """One line per production setting that is missing or wrong. Never includes a value."""
    found = [f"{name} is not set. It is {purpose}." for name, purpose in REQUIRED.items() if not _value(env, name)]

    if not is_configured(FORGE_SCRUB_KEY, env):
        found.append(f"FORGE_SCRUB_KEY is not set. It keys {FORGE_SCRUB_KEY.purpose.removeprefix('keying ')}.")

    redis = _value(env, "REDIS_URL")
    if redis and not urlparse(redis).password:
        found.append("REDIS_URL has no password. Put FORGE_REDIS_PASSWORD in it: redis://:<password>@host:6379/0")

    origins = [origin.strip() for origin in _value(env, "CORS_ORIGINS").split(",") if origin.strip()]
    loopback = [origin for origin in origins if (urlparse(origin).hostname or "") in _LOOPBACK_HOSTS]
    if loopback:
        found.append(f"CORS_ORIGINS includes a loopback origin ({', '.join(loopback)}). Use the deployed host.")

    provider = (env.get("FORGE_LLM_PROVIDER") or "anthropic").strip().lower()
    if provider != "ollama":
        credential = PROVIDER_CREDENTIALS.get(provider)
        if credential is None:
            found.append(f"FORGE_LLM_PROVIDER {provider!r} is not one of: ollama, {', '.join(PROVIDER_CREDENTIALS)}.")
        elif not is_configured(credential, env):
            found.append(f"{credential.env_vars[0]} is not set. FORGE_LLM_PROVIDER={provider} needs it.")
    return found


def enforce_production_settings(env: Mapping[str, str] | None = None) -> None:
    source = os.environ if env is None else env
    if (source.get("FORGE_ENV") or "").strip().lower() != "production":
        return
    found = problems(source)
    if found:
        for line in found:
            print(f"[forge] {line}", file=sys.stderr)
        raise SystemExit(1)
