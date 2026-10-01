"""The writer and validator must come from different model families.

A missing validator setting, or a validator from the writer's family, stops
the job before any LLM call, with the setting named in the error.
"""
from __future__ import annotations

import pytest

from forge.taskfactory.model_settings import (
    VALIDATOR_MODEL_VAR,
    VALIDATOR_PROVIDER_VAR,
    ModelSpec,
    TaskFactoryConfigError,
    resolve_models,
    validator_spec,
    writer_spec,
)

WRITER_ENV = {"FORGE_LLM_PROVIDER": "anthropic", "FORGE_LLM_MODEL_CAPABLE": "claude-sonnet-5"}


def test_the_writer_is_the_capable_generation_model():
    assert writer_spec(WRITER_ENV) == ModelSpec(provider="anthropic", model="claude-sonnet-5")


def test_the_writer_falls_back_to_the_provider_default():
    spec = writer_spec({"FORGE_LLM_PROVIDER": "openai"})

    assert spec.provider == "openai"
    assert spec.model


def test_a_cross_family_validator_resolves():
    env = {**WRITER_ENV, VALIDATOR_PROVIDER_VAR: "openai", VALIDATOR_MODEL_VAR: "gpt-5"}

    writer, validator = resolve_models(env)
    assert writer.family != validator.family
    assert validator == ModelSpec(provider="openai", model="gpt-5")


@pytest.mark.parametrize("missing", [VALIDATOR_PROVIDER_VAR, VALIDATOR_MODEL_VAR])
def test_a_missing_validator_setting_is_named(missing):
    env = {**WRITER_ENV, VALIDATOR_PROVIDER_VAR: "openai", VALIDATOR_MODEL_VAR: "gpt-5"}
    env[missing] = "  "

    with pytest.raises(TaskFactoryConfigError, match=missing):
        validator_spec(env)


def test_an_unknown_validator_provider_is_rejected():
    env = {VALIDATOR_PROVIDER_VAR: "mystery", VALIDATOR_MODEL_VAR: "m-1"}

    with pytest.raises(TaskFactoryConfigError, match=VALIDATOR_PROVIDER_VAR):
        validator_spec(env)


def test_a_provider_whose_sdk_is_missing_is_named_before_any_call(monkeypatch):
    import forge.taskfactory.model_settings as settings

    monkeypatch.setattr(settings, "_installed", lambda module: module != "openai")

    with pytest.raises(TaskFactoryConfigError, match="openai"):
        settings.require_sdks(ModelSpec(provider="openai", model="gpt-5"))


def test_installed_sdks_pass_the_check(monkeypatch):
    import forge.taskfactory.model_settings as settings

    monkeypatch.setattr(settings, "_installed", lambda module: True)

    settings.require_sdks(ModelSpec(provider="anthropic", model="claude-sonnet-5"), ModelSpec(provider="gemini", model="gemini-3"))


def test_a_validator_from_the_writers_family_is_rejected():
    # A cheaper tier of the same vendor is the same family: one opinion twice.
    env = {**WRITER_ENV, VALIDATOR_PROVIDER_VAR: "anthropic", VALIDATOR_MODEL_VAR: "claude-haiku-4-5"}

    with pytest.raises(TaskFactoryConfigError, match=VALIDATOR_MODEL_VAR):
        resolve_models(env)
