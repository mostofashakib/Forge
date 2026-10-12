"""The safety service is built from env per source and fails clearly when misconfigured."""
from types import SimpleNamespace

import pytest

from forge.safety import config
from forge.safety.adult import DEFAULT_ADULT_MODEL
from forge.safety.classifier import SafetyVerdict
from forge.safety.config import SafetyConfigError, safety_service_from_env
from forge.safety.ner import DEFAULT_NER_MODEL
from forge.safety.types import DataSource


class NoNames:
    def batch_extract_entities_long(self, texts, entity_types, threshold, include_spans):
        return [{"entities": {"person": []}} for _ in texts]


class SafePipeline:
    def __init__(self, labels=("safe", "nsfw")):
        self.model = SimpleNamespace(config=SimpleNamespace(id2label=dict(enumerate(labels))))

    def __call__(self, windows, truncation, max_length, batch_size):
        return [[{"label": "nsfw", "score": 0.0}, {"label": "safe", "score": 1.0}] for _ in windows]


@pytest.fixture
def env(monkeypatch):
    for name in ("FORGE_SCRUB_KEY", "FORGE_NER_MODEL", "FORGE_NER_THRESHOLD", "FORGE_SAFETY_LLM",
                 "FORGE_JUDGE_MODEL", "FORGE_ADULT_MODEL", "FORGE_ADULT_THRESHOLD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FORGE_SCRUB_KEY", "k" * 32)
    loaded = {"ner": [], "adult": []}
    monkeypatch.setattr(config, "load_gliner", lambda model_id: loaded["ner"].append(model_id) or NoNames())
    monkeypatch.setattr(config, "load_adult_pipeline", lambda model_id: loaded["adult"].append(model_id) or SafePipeline())
    return loaded


def test_model_data_needs_no_scrub_key_and_loads_no_ner(env, monkeypatch):
    monkeypatch.delenv("FORGE_SCRUB_KEY")
    safety = safety_service_from_env(DataSource.MODEL)
    assert env == {"ner": [], "adult": [DEFAULT_ADULT_MODEL]}
    result = safety.service.screen({"c": "mail bob@acme.com"})
    assert result.kept and result.record["c"] == "mail bob@acme.com"


@pytest.mark.parametrize("source", [DataSource.INTERNET, DataSource.HUMAN])
def test_external_data_is_scrubbed_with_the_default_ner_model(env, source):
    safety = safety_service_from_env(source)
    assert safety.mode == "rules"
    assert env["ner"] == [DEFAULT_NER_MODEL]
    result = safety.service.screen({"c": "mail bob@acme.com"})
    assert result.kept and "bob@acme.com" not in result.record["c"]


def test_a_missing_scrub_key_fails_external_data_naming_the_variable(env, monkeypatch):
    monkeypatch.delenv("FORGE_SCRUB_KEY")
    with pytest.raises(SafetyConfigError, match="FORGE_SCRUB_KEY"):
        safety_service_from_env(DataSource.INTERNET)


def test_a_blank_scrub_key_fails_too(env, monkeypatch):
    monkeypatch.setenv("FORGE_SCRUB_KEY", "   ")
    with pytest.raises(SafetyConfigError, match="FORGE_SCRUB_KEY"):
        safety_service_from_env(DataSource.HUMAN)


def test_the_ner_model_is_configurable(env, monkeypatch):
    monkeypatch.setenv("FORGE_NER_MODEL", "org/other-model")
    safety_service_from_env(DataSource.INTERNET)
    assert env["ner"] == ["org/other-model"]


def test_a_ner_model_that_fails_to_load_names_the_variable(env, monkeypatch):
    def broken(model_id):
        raise OSError("no such repo")
    monkeypatch.setattr(config, "load_gliner", broken)
    with pytest.raises(SafetyConfigError, match="FORGE_NER_MODEL") as info:
        safety_service_from_env(DataSource.INTERNET)
    assert "no such repo" in str(info.value)


def test_a_bad_ner_threshold_names_the_variable(env, monkeypatch):
    monkeypatch.setenv("FORGE_NER_THRESHOLD", "high")
    with pytest.raises(SafetyConfigError, match="FORGE_NER_THRESHOLD"):
        safety_service_from_env(DataSource.INTERNET)


def test_the_adult_model_is_configurable(env, monkeypatch):
    monkeypatch.setenv("FORGE_ADULT_MODEL", "org/nsfw")
    safety_service_from_env(DataSource.MODEL)
    assert env["adult"] == ["org/nsfw"]


def test_an_adult_model_that_fails_to_load_names_the_variable(env, monkeypatch):
    def broken(model_id):
        raise OSError("offline")
    monkeypatch.setattr(config, "load_adult_pipeline", broken)
    with pytest.raises(SafetyConfigError, match="FORGE_ADULT_MODEL"):
        safety_service_from_env(DataSource.MODEL)


def test_an_adult_model_without_an_nsfw_label_names_the_variable(env, monkeypatch):
    monkeypatch.setattr(config, "load_adult_pipeline", lambda _id: SafePipeline(labels=("LABEL_0", "LABEL_1")))
    with pytest.raises(SafetyConfigError, match="FORGE_ADULT_MODEL"):
        safety_service_from_env(DataSource.MODEL)


def test_a_bad_adult_threshold_names_the_variable(env, monkeypatch):
    monkeypatch.setenv("FORGE_ADULT_THRESHOLD", "1.5")
    with pytest.raises(SafetyConfigError, match="FORGE_ADULT_THRESHOLD"):
        safety_service_from_env(DataSource.MODEL)


def test_the_llm_flag_adds_the_judge(env, monkeypatch):
    class Judge:
        def extract(self, system, user, schema):
            return SafetyVerdict(harmful=True, categories=["hate"], reason="slur")
    monkeypatch.setenv("FORGE_SAFETY_LLM", "1")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "judge-model")
    monkeypatch.setattr(config, "get_judge_client", lambda: Judge())

    safety = safety_service_from_env(DataSource.MODEL)

    assert safety.mode == "rules+llm:judge-model"
    assert not safety.service.screen({"c": "text"}).kept


def test_a_judge_that_cannot_be_built_fails_clearly(env, monkeypatch):
    def broken():
        raise ValueError("Unknown LLM provider: 'nope'")
    monkeypatch.setenv("FORGE_SAFETY_LLM", "1")
    monkeypatch.setattr(config, "get_judge_client", broken)
    with pytest.raises(SafetyConfigError, match="FORGE_SAFETY_LLM"):
        safety_service_from_env(DataSource.MODEL)


def test_code_scrubbing_skips_the_name_model_but_still_needs_the_key(env, monkeypatch):
    safety = safety_service_from_env(DataSource.INTERNET, detect_names=False)
    assert env["ner"] == []
    result = safety.service.screen({"c": "import Widgets  # maintainer bob@acme.com"})
    assert result.record["c"].startswith("import Widgets  # maintainer user_")
    monkeypatch.delenv("FORGE_SCRUB_KEY")
    with pytest.raises(SafetyConfigError, match="FORGE_SCRUB_KEY"):
        safety_service_from_env(DataSource.INTERNET, detect_names=False)
