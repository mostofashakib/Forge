from __future__ import annotations

import pytest

import json

from forge.experiments import ExperimentConfig, RunResult


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


# ---------------------------------------------------------------------------
# Grader independence
# ---------------------------------------------------------------------------

def test_experiment_requires_grader_independence_by_default(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "train_envs: [a]\nheldout_envs: [b]\nreward_preset: judge_only\n"
        "base_model: m\nseeds: [0]\n",
        encoding="utf-8",
    )
    assert ExperimentConfig.load(path).require_grader_independence is True


def test_experiment_can_waive_grader_independence(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "train_envs: [a]\nheldout_envs: [b]\nreward_preset: judge_only\n"
        "base_model: m\nseeds: [0]\nrequire_grader_independence: false\n",
        encoding="utf-8",
    )
    assert ExperimentConfig.load(path).require_grader_independence is False


def test_experiment_rejects_a_non_boolean_independence_flag(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "train_envs: [a]\nheldout_envs: [b]\nreward_preset: judge_only\n"
        "base_model: m\nseeds: [0]\nrequire_grader_independence: sometimes\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        ExperimentConfig.load(path)


def test_run_result_persists_grading_provenance(tmp_path):
    result = RunResult(
        config={"k": "v"}, seed=0, determinism="on",
        heldout_pass_rate=1.0, reward_hacking_rate=0.0, reward_variance=0.0,
        grading={"llm_graded": False, "independent": True},
    )
    path = result.save(tmp_path, "run-1")
    assert json.loads(path.read_text())["grading"]["independent"] is True


def test_run_result_without_grading_provenance_records_none(tmp_path):
    result = RunResult(
        config={}, seed=0, determinism="on",
        heldout_pass_rate=1.0, reward_hacking_rate=0.0, reward_variance=0.0,
    )
    path = result.save(tmp_path, "run-2")
    assert json.loads(path.read_text())["grading"] is None


def test_experiment_defaults_a_maximum_abstention_rate(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "train_envs: [a]\nheldout_envs: [b]\nreward_preset: full_layered_partial\n"
        "base_model: m\nseeds: [0]\n",
        encoding="utf-8",
    )
    assert ExperimentConfig.load(path).max_abstention_rate == 0.2


def test_experiment_rejects_an_abstention_ceiling_above_one(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "train_envs: [a]\nheldout_envs: [b]\nreward_preset: full_layered_partial\n"
        "base_model: m\nseeds: [0]\nmax_abstention_rate: 1.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        ExperimentConfig.load(path)


def test_experiment_rejects_a_negative_abstention_ceiling(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "train_envs: [a]\nheldout_envs: [b]\nreward_preset: full_layered_partial\n"
        "base_model: m\nseeds: [0]\nmax_abstention_rate: -0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        ExperimentConfig.load(path)


def test_run_result_records_the_abstention_rate(tmp_path):
    result = RunResult(
        config={}, seed=0, determinism="on",
        heldout_pass_rate=1.0, reward_hacking_rate=0.0, reward_variance=0.0,
        abstention_rate=0.125,
    )
    path = result.save(tmp_path, "run-a")
    assert json.loads(path.read_text())["abstention_rate"] == 0.125


def test_run_result_defaults_the_abstention_rate_to_zero(tmp_path):
    result = RunResult(
        config={}, seed=0, determinism="on",
        heldout_pass_rate=1.0, reward_hacking_rate=0.0, reward_variance=0.0,
    )
    assert result.abstention_rate == 0.0
