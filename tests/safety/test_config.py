"""The safety service is built from env and fails clearly when misconfigured."""
import pytest

from forge.safety import config
from forge.safety.config import SafetyConfigError, safety_service_from_env
from forge.safety.ner import DEFAULT_NER_MODEL
from forge.safety.classifier import SafetyVerdict


class NoNames:
    def batch_extract_entities_long(self, texts, entity_types, threshold, include_spans):
        return [{"entities": {"person": []}} for _ in texts]


@pytest.fixture
def env(monkeypatch):
    for name in ("FORGE_SCRUB_KEY", "FORGE_NER_MODEL", "FORGE_NER_THRESHOLD", "FORGE_SAFETY_LLM", "FORGE_JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FORGE_SCRUB_KEY", "k" * 32)
    loaded = []
    monkeypatch.setattr(config, "load_gliner", lambda model_id: loaded.append(model_id) or NoNames())
    return loaded


def test_a_missing_scrub_key_fails_naming_the_variable(env, monkeypatch):
    monkeypatch.delenv("FORGE_SCRUB_KEY")
    with pytest.raises(SafetyConfigError, match="FORGE_SCRUB_KEY"):
        safety_service_from_env()


def test_a_blank_scrub_key_fails_too(env, monkeypatch):
    monkeypatch.setenv("FORGE_SCRUB_KEY", "   ")
    with pytest.raises(SafetyConfigError, match="FORGE_SCRUB_KEY"):
        safety_service_from_env()


def test_defaults_build_a_rules_only_service_on_the_default_model(env):
    safety = safety_service_from_env()
    assert safety.mode == "rules"
    assert env == [DEFAULT_NER_MODEL]
    result = safety.service.screen({"c": "mail bob@acme.com"})
    assert result.kept and "bob@acme.com" not in result.record["c"]


def test_the_ner_model_is_configurable(env, monkeypatch):
    monkeypatch.setenv("FORGE_NER_MODEL", "org/other-model")
    safety_service_from_env()
    assert env == ["org/other-model"]


def test_a_model_that_fails_to_load_names_the_variable(env, monkeypatch):
    def broken(model_id):
        raise OSError("no such repo")
    monkeypatch.setattr(config, "load_gliner", broken)
    with pytest.raises(SafetyConfigError, match="FORGE_NER_MODEL") as info:
        safety_service_from_env()
    assert "no such repo" in str(info.value)


def test_a_bad_threshold_names_the_variable(env, monkeypatch):
    monkeypatch.setenv("FORGE_NER_THRESHOLD", "high")
    with pytest.raises(SafetyConfigError, match="FORGE_NER_THRESHOLD"):
        safety_service_from_env()


def test_the_llm_flag_adds_the_judge(env, monkeypatch):
    class Judge:
        def extract(self, system, user, schema):
            return SafetyVerdict(harmful=True, categories=["hate"], reason="slur")
    monkeypatch.setenv("FORGE_SAFETY_LLM", "1")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "judge-model")
    monkeypatch.setattr(config, "get_judge_client", lambda: Judge())

    safety = safety_service_from_env()

    assert safety.mode == "rules+llm:judge-model"
    assert not safety.service.screen({"c": "text"}).kept


def test_a_judge_that_cannot_be_built_fails_clearly(env, monkeypatch):
    def broken():
        raise ValueError("Unknown LLM provider: 'nope'")
    monkeypatch.setenv("FORGE_SAFETY_LLM", "1")
    monkeypatch.setattr(config, "get_judge_client", broken)
    with pytest.raises(SafetyConfigError, match="FORGE_SAFETY_LLM"):
        safety_service_from_env()
