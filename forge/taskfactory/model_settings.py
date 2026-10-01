"""Which models write tasks and which validate them.

The writer is the capable generation model. The validator has its own two
settings and must come from a different model family, or the review is the
writer grading itself.
"""
from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass

from forge.extraction.llm_client import _ANTHROPIC_CAPABLE, _PROVIDER_DEFAULTS
from forge.grading_provenance import model_family

WRITER_PROVIDER_VAR = "FORGE_LLM_PROVIDER"
WRITER_MODEL_VAR = "FORGE_LLM_MODEL_CAPABLE"
VALIDATOR_PROVIDER_VAR = "FORGE_TASK_VALIDATOR_PROVIDER"
VALIDATOR_MODEL_VAR = "FORGE_TASK_VALIDATOR_MODEL"
VALIDATOR_VARS = (VALIDATOR_PROVIDER_VAR, VALIDATOR_MODEL_VAR)
SUPPORTED_PROVIDERS: tuple[str, ...] = tuple(sorted(_PROVIDER_DEFAULTS))
# The package each provider's client imports.
PROVIDER_SDKS: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google.genai",
    "ollama": "ollama",
}


class TaskFactoryConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    @property
    def family(self) -> str:
        return model_family(self.model)


def writer_spec(env: Mapping[str, str]) -> ModelSpec:
    """The model `get_client(capable=True)` would use under `env`."""
    provider = (env.get(WRITER_PROVIDER_VAR) or "anthropic").strip().lower()
    if provider not in _PROVIDER_DEFAULTS:
        raise TaskFactoryConfigError(f"{WRITER_PROVIDER_VAR}={provider!r} is not a supported provider")
    fallback = _ANTHROPIC_CAPABLE if provider == "anthropic" else _PROVIDER_DEFAULTS[provider]
    model = (env.get(WRITER_MODEL_VAR) or "").strip() or fallback
    return ModelSpec(provider=provider, model=model)


def validator_spec(env: Mapping[str, str]) -> ModelSpec:
    provider = (env.get(VALIDATOR_PROVIDER_VAR) or "").strip().lower()
    model = (env.get(VALIDATOR_MODEL_VAR) or "").strip()
    if not provider:
        raise TaskFactoryConfigError(f"{VALIDATOR_PROVIDER_VAR} is not set")
    if provider not in _PROVIDER_DEFAULTS:
        raise TaskFactoryConfigError(
            f"{VALIDATOR_PROVIDER_VAR}={provider!r} is not one of {list(SUPPORTED_PROVIDERS)}"
        )
    if not model:
        raise TaskFactoryConfigError(f"{VALIDATOR_MODEL_VAR} is not set")
    return ModelSpec(provider=provider, model=model)


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def require_sdks(*specs: ModelSpec) -> None:
    """Refuse to start when a provider's SDK is missing, before any tokens are spent."""
    for spec in specs:
        module = PROVIDER_SDKS[spec.provider]
        if not _installed(module):
            raise TaskFactoryConfigError(
                f"the {spec.provider} provider needs the {module!r} Python package, which is not installed"
            )


def resolve_models(env: Mapping[str, str]) -> tuple[ModelSpec, ModelSpec]:
    """Return (writer, validator), refusing a validator from the writer's family."""
    writer = writer_spec(env)
    validator = validator_spec(env)
    if validator.family == writer.family:
        raise TaskFactoryConfigError(
            f"{VALIDATOR_MODEL_VAR}={validator.model!r} is from the {writer.family!r} family, "
            f"the same as the task writer {writer.model!r}. Pick a model from another family."
        )
    return writer, validator
