from __future__ import annotations

import pytest

from forge.experiments import ExperimentConfig


def test_experiment_config_loads_split(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "train_envs: [a, b]\nheldout_envs: [c]\nreward_preset: full_layered_partial\n"
        "base_model: model\nseeds: [1, 2]\n"
    )
    config = ExperimentConfig.load(path)
    assert config.train_envs == ["a", "b"]
    assert config.heldout_envs == ["c"]


def test_experiment_config_rejects_split_overlap():
    with pytest.raises(ValueError, match="overlap"):
        ExperimentConfig(
            train_envs=["same"], heldout_envs=["same"], reward_preset="full_layered_partial",
            base_model="model", seeds=[0],
        )


def test_experiment_config_rejects_unknown_reward_preset():
    with pytest.raises(ValueError, match="reward_preset"):
        ExperimentConfig(
            train_envs=["a"], heldout_envs=["b"], reward_preset="unknown",
            base_model="model", seeds=[0],
        )
