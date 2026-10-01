"""Settings page API: every editable platform setting, saved to backend/.env.

API keys stay in that file only. This API never reads a key's value and
refuses to write one. It reports whether each provider has a key, so the
page can warn before you pick a provider that cannot run.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services import env_file
from backend.app.services.ollama_catalog import OllamaUnavailable, list_ollama_models
from backend.app.services.settings_registry import (
    DEFAULT_OLLAMA_URL,
    KEYS,
    SETTINGS,
    SettingsError,
    check_models,
    check_value,
    setting,
)
from forge.taskfactory.model_settings import (
    SUPPORTED_PROVIDERS,
    ModelSpec,
    TaskFactoryConfigError,
    require_sdks,
    resolve_models,
    validator_spec,
    writer_spec,
)

router = APIRouter(prefix="/api/settings")

# Any one of these variables holding a value means the provider has a key.
PROVIDER_KEY_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "ollama": (),
}


class SettingsUpdate(BaseModel):
    values: dict[str, str]


def _spec_view(spec: ModelSpec) -> dict:
    return {"provider": spec.provider, "model": spec.model, "family": spec.family}


def _validator_view(env: dict[str, str]) -> dict:
    try:
        writer, validator = resolve_models(env)
        require_sdks(writer, validator)
    except TaskFactoryConfigError as exc:
        try:
            partial = validator_spec(env)
        except TaskFactoryConfigError:
            return {"provider": None, "model": None, "family": None, "configured": False, "error": str(exc)}
        return {**_spec_view(partial), "configured": False, "error": str(exc)}
    return {**_spec_view(validator), "configured": True, "error": None}


def _providers(env: dict[str, str]) -> list[dict]:
    return [
        {
            "name": name,
            "needs_key": bool(PROVIDER_KEY_VARS.get(name)),
            "key_set": any(env.get(var) for var in PROVIDER_KEY_VARS.get(name, ())),
        }
        for name in SUPPORTED_PROVIDERS
    ]


def _ollama(env: dict[str, str]) -> dict:
    try:
        return {"models": list_ollama_models(env.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL), "error": None}
    except OllamaUnavailable as exc:
        return {"models": [], "error": str(exc)}


def _effective(saved: dict[str, str]) -> dict[str, str]:
    """What each setting is now: the saved file, else the process, else the default."""
    return {spec.key: saved.get(spec.key, os.environ.get(spec.key, spec.default)) for spec in SETTINGS}


def _setting_rows(saved: dict[str, str], effective: dict[str, str]) -> list[dict]:
    return [
        {
            "key": spec.key,
            "group": spec.group,
            "label": spec.label,
            "kind": spec.kind,
            "help": spec.help,
            "choices": list(spec.choices),
            "minimum": spec.minimum,
            "optional": spec.optional,
            "default": spec.default,
            "value": effective[spec.key],
            "live": spec.live,
            # Saved, but the running API and workers still hold the old value.
            "pending_restart": not spec.live and spec.key in saved and saved[spec.key] != os.environ.get(spec.key),
        }
        for spec in SETTINGS
    ]


def _settings_body() -> dict:
    key_vars = [var for names in PROVIDER_KEY_VARS.values() for var in names]
    saved = env_file.read_env_values(env_file.BACKEND_ENV_FILE, KEYS)
    env = {**os.environ, **env_file.read_env_values(env_file.BACKEND_ENV_FILE, key_vars), **saved}
    return {
        "writer": _spec_view(writer_spec(env)),
        "task_validator": _validator_view(env),
        "providers": _providers(env),
        "ollama": _ollama(env),
        "settings": _setting_rows(saved, _effective(saved)),
    }


@router.get("")
def get_settings() -> dict:
    return _settings_body()


@router.put("")
def update_settings(body: SettingsUpdate) -> dict:
    """Check every value and the model families together, then save them in one write."""
    try:
        updates = {key: check_value(setting(key), raw) for key, raw in body.values.items()}
        saved = env_file.read_env_values(env_file.BACKEND_ENV_FILE, KEYS)
        check_models({**_effective(saved), **updates})
        env_file.update_env_file(env_file.BACKEND_ENV_FILE, updates)
    except (SettingsError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _settings_body()
