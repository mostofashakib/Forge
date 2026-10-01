"""The editable platform settings: what each accepts, and the cross-model family rules."""
from __future__ import annotations

import pytest

from backend.app.services.settings_registry import (
    SETTINGS,
    SettingsError,
    check_models,
    check_value,
    setting,
)
from forge.envgen.config import EnvGenConfig

GENERATOR = {
    "FORGE_LLM_PROVIDER": "anthropic",
    "FORGE_LLM_MODEL": "claude-haiku-4-5",
    "FORGE_LLM_MODEL_CAPABLE": "claude-sonnet-5",
}


def test_keys_are_unique_and_no_api_key_is_editable():
    keys = [s.key for s in SETTINGS]

    assert len(keys) == len(set(keys))
    assert not [k for k in keys if k.endswith("_API_KEY") or "SECRET" in k]


def test_infrastructure_settings_are_left_off_the_page():
    keys = {s.key for s in SETTINGS}

    assert keys.isdisjoint({"FORGE_DB_URL", "FORGE_GENERATED_ENVS_DIR", "FORGE_HOST", "FORGE_DEBUG", "FORGE_ENV"})


def test_budget_defaults_match_the_generation_config():
    assert setting("FORGE_ENVGEN_CAPABLE_TOKENS").default == str(EnvGenConfig().capable_llm_tokens)
    assert setting("FORGE_RESEARCH_HTTP_TIMEOUT").default == str(EnvGenConfig().research_http_timeout)


def test_only_the_task_validator_applies_without_a_restart():
    live = {s.key for s in SETTINGS if s.live}

    assert live == {"FORGE_TASK_VALIDATOR_PROVIDER", "FORGE_TASK_VALIDATOR_MODEL"}


@pytest.mark.parametrize(
    ("key", "raw", "saved"),
    [
        ("FORGE_SANDBOX_LIMIT", " 12 ", "12"),
        ("FORGE_DETERMINISM", "off", "off"),
        ("FORGE_LLM_PROVIDER", "Ollama", "ollama"),
        ("FORGE_CONTAINER_MEMORY", "512m", "512m"),
        ("FORGE_RESEARCH_HTTP_TIMEOUT", "2.5", "2.5"),
        ("FORGE_ENVGEN_MAX_REPAIR_ROUNDS", "0", "0"),
        ("FORGE_JUDGE_MODEL", "", ""),
        ("OLLAMA_BASE_URL", "http://ollama:11434", "http://ollama:11434"),
    ],
)
def test_good_values_are_normalized(key, raw, saved):
    assert check_value(setting(key), raw) == saved


@pytest.mark.parametrize(
    ("key", "raw"),
    [
        ("FORGE_SANDBOX_LIMIT", "0"),
        ("FORGE_SANDBOX_LIMIT", "ten"),
        ("FORGE_ENVGEN_MAX_REPAIR_ROUNDS", "-1"),
        ("FORGE_RESEARCH_HTTP_TIMEOUT", "0"),
        ("FORGE_DETERMINISM", "maybe"),
        ("FORGE_LLM_PROVIDER", "mystery"),
        ("FORGE_CONTAINER_MEMORY", "lots"),
        ("FORGE_LLM_MODEL", ""),
        ("FORGE_CLI_IMAGE", ""),
        ("OLLAMA_BASE_URL", "localhost:11434"),
    ],
)
def test_bad_values_are_refused_with_the_setting_name(key, raw):
    with pytest.raises(SettingsError, match=key):
        check_value(setting(key), raw)


def test_an_unknown_setting_is_refused():
    with pytest.raises(SettingsError, match="ANTHROPIC_API_KEY"):
        setting("ANTHROPIC_API_KEY")


def test_independent_judge_quorum_and_validator_pass():
    check_models({
        **GENERATOR,
        "FORGE_JUDGE_PROVIDER": "openai",
        "FORGE_JUDGE_MODEL": "gpt-5",
        "FORGE_QUORUM_MODELS": "openai:gpt-5,ollama:qwen3:32b",
        "FORGE_TASK_VALIDATOR_PROVIDER": "ollama",
        "FORGE_TASK_VALIDATOR_MODEL": "glm-4.7-flash:latest",
    })


def test_unset_judge_quorum_and_validator_pass():
    check_models(GENERATOR)


def test_a_judge_from_the_generator_family_is_refused():
    with pytest.raises(SettingsError, match="FORGE_JUDGE_MODEL"):
        check_models({**GENERATOR, "FORGE_JUDGE_MODEL": "claude-opus-5"})


def test_a_judge_provider_without_a_model_is_refused():
    with pytest.raises(SettingsError, match="FORGE_JUDGE_MODEL"):
        check_models({**GENERATOR, "FORGE_JUDGE_PROVIDER": "openai"})


def test_a_quorum_member_from_the_generator_family_is_refused():
    with pytest.raises(SettingsError, match="FORGE_QUORUM_MODELS"):
        check_models({**GENERATOR, "FORGE_QUORUM_MODELS": "openai:gpt-5,anthropic:claude-opus-5"})


def test_a_quorum_with_two_members_of_one_family_is_refused():
    with pytest.raises(SettingsError, match="FORGE_QUORUM_MODELS"):
        check_models({**GENERATOR, "FORGE_QUORUM_MODELS": "openai:gpt-5,openai:gpt-4o"})


def test_changing_the_generator_into_the_validators_family_is_refused():
    with pytest.raises(SettingsError, match="family"):
        check_models({
            **GENERATOR,
            "FORGE_LLM_PROVIDER": "openai",
            "FORGE_LLM_MODEL": "gpt-5-mini",
            "FORGE_LLM_MODEL_CAPABLE": "gpt-5",
            "FORGE_TASK_VALIDATOR_PROVIDER": "openai",
            "FORGE_TASK_VALIDATOR_MODEL": "gpt-4o",
        })


def test_a_half_set_validator_is_refused():
    with pytest.raises(SettingsError, match="FORGE_TASK_VALIDATOR_MODEL"):
        check_models({**GENERATOR, "FORGE_TASK_VALIDATOR_PROVIDER": "openai"})


def test_an_unset_generator_model_counts_as_the_providers_default():
    with pytest.raises(SettingsError, match="FORGE_JUDGE_MODEL"):
        check_models({"FORGE_LLM_PROVIDER": "openai", "FORGE_JUDGE_MODEL": "gpt-4o"})
