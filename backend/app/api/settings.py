"""Settings page API: the task writer and validator models.

The validator is saved to backend/.env. API keys stay in that file only:
this API reports whether each provider has one, never the key itself.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services import env_file
from forge.taskfactory.model_settings import (
    SUPPORTED_PROVIDERS,
    VALIDATOR_MODEL_VAR,
    VALIDATOR_PROVIDER_VAR,
    VALIDATOR_VARS,
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


class TaskValidatorUpdate(BaseModel):
    provider: str
    model: str


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


def _settings_body() -> dict:
    key_vars = [var for names in PROVIDER_KEY_VARS.values() for var in names]
    env = env_file.environ_with_saved([*VALIDATOR_VARS, *key_vars])
    return {
        "writer": _spec_view(writer_spec(env)),
        "task_validator": _validator_view(env),
        "providers": _providers(env),
    }


@router.get("")
def get_settings() -> dict:
    return _settings_body()


@router.put("/task-validator")
def update_task_validator(body: TaskValidatorUpdate) -> dict:
    candidate = {
        **os.environ,
        VALIDATOR_PROVIDER_VAR: body.provider.strip().lower(),
        VALIDATOR_MODEL_VAR: body.model.strip(),
    }
    try:
        resolve_models(candidate)
        env_file.update_env_file(
            env_file.BACKEND_ENV_FILE,
            {VALIDATOR_PROVIDER_VAR: candidate[VALIDATOR_PROVIDER_VAR], VALIDATOR_MODEL_VAR: candidate[VALIDATOR_MODEL_VAR]},
        )
    except (TaskFactoryConfigError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _settings_body()
