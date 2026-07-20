"""The checkpoint manifest round-trips and loads into a runtime agent."""
from __future__ import annotations

import pytest

from forge.training.checkpoint import PolicyCheckpoint, load_policy_agent


def test_checkpoint_save_load_round_trip(tmp_path):
    ckpt = PolicyCheckpoint(
        objective="grpo", base_model="Qwen/Qwen2.5-3B",
        model_path=str(tmp_path / "forge_policy"), num_examples=12, mean_reward=0.62,
    )
    ckpt.save(tmp_path)
    loaded = PolicyCheckpoint.load(tmp_path)
    assert loaded.objective == "grpo"
    assert loaded.num_examples == 12
    assert loaded.model_path == str(tmp_path / "forge_policy")
    assert loaded.created_at  # stamped on save


def test_load_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PolicyCheckpoint.load(tmp_path)


def test_load_policy_agent_serves_the_trained_model(tmp_path):
    PolicyCheckpoint(
        objective="dpo", base_model="base", model_path="forge-policy-v1",
        num_examples=4, mean_reward=0.5,
    ).save(tmp_path)
    # A dummy client avoids requiring the openai package / a live endpoint.
    agent = load_policy_agent(tmp_path, client=object())
    assert agent._model == "forge-policy-v1"
