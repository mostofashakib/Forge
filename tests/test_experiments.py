from __future__ import annotations

import pytest

from forge.experiments import ExperimentConfig


def test_experiment_config_loads_split(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "train_envs: [a, b]\nheldout_envs: [c]\nreward_preset: balanced\n"
        "base_model: model\nseeds: [1, 2]\n"
    )
    config = ExperimentConfig.load(path)
    assert config.train_envs == ["a", "b"]
    assert config.heldout_envs == ["c"]


def test_experiment_config_rejects_split_overlap():
    with pytest.raises(ValueError, match="overlap"):
        ExperimentConfig(
            train_envs=["same"], heldout_envs=["same"], reward_preset="balanced",
            base_model="model", seeds=[0],
        )
