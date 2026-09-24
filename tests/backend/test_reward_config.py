from __future__ import annotations

import json
import logging

from backend.app.services import reward_config
from tests.backend.test_sandbox_e2e import _add_sandbox, client  # noqa: F401


def _write(tmp_path, monkeypatch, payload: str) -> None:
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path))
    (tmp_path / "env").mkdir(exist_ok=True)
    (tmp_path / "env" / "reward_config.json").write_text(payload)


def test_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path))
    config = reward_config.load_reward_config("env")
    assert config.scoring_methods == ["llm"]
    assert config.reward_preset == "full_layered_partial"


def test_reads_the_legacy_single_method_key(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, json.dumps({"scoring_method": "rouge"}))
    assert reward_config.load_reward_config("env").scoring_methods == ["rouge"]


def test_empty_method_list_falls_back_to_llm(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, json.dumps({"scoring_methods": []}))
    assert reward_config.load_reward_config("env").scoring_methods == ["llm"]


def test_corrupt_file_falls_back_to_defaults_and_says_so(tmp_path, monkeypatch, caplog):
    _write(tmp_path, monkeypatch, "{not json")
    with caplog.at_level(logging.WARNING):
        config = reward_config.load_reward_config("env")
    assert config == reward_config.RewardConfig()
    assert "reward_config.json" in caplog.text


def test_unknown_preset_falls_back_to_default(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, json.dumps({"reward_preset": "made_up", "scoring_methods": ["bleu"]}))
    config = reward_config.load_reward_config("env")
    assert config.reward_preset == "full_layered_partial"
    assert config.scoring_methods == ["bleu"]


def test_saving_one_field_keeps_the_other(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path))
    reward_config.save_reward_config("env", scoring_methods=["rouge", "bleu"])
    reward_config.save_reward_config("env", reward_preset="judge_only")
    config = reward_config.load_reward_config("env")
    assert config.scoring_methods == ["rouge", "bleu"]
    assert config.reward_preset == "judge_only"


def test_evaluate_api_writes_under_the_configured_envs_dir(client, tmp_path):
    _add_sandbox(client, "cfg_env")
    response = client.put("/api/sandbox/cfg_env/evaluate", json={"scoring_methods": ["rouge"]})
    assert response.status_code == 200
    saved = tmp_path / "generated_envs" / "cfg_env" / "reward_config.json"
    assert json.loads(saved.read_text())["scoring_methods"] == ["rouge"]
    assert client.get("/api/sandbox/cfg_env/evaluate").json()["scoring_methods"] == ["rouge"]
