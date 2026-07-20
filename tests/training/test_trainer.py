"""PolicyTrainer wiring, checkpoint contract, and negative/false-positive paths.

Heavy training is mocked via an injected backend, so these exercise the
data → reward-mapping → train → checkpoint orchestration, not a GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from forge.training.checkpoint import PolicyCheckpoint
from forge.training.dataset import MalformedExportError
from forge.training.trainer import (
    NoTrainingSignalError,
    PolicyTrainer,
    TrainingConfig,
    TrainingObjective,
)


class _FakeBackend:
    """Records the examples it was handed and writes a stand-in checkpoint."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def train(self, base_model, examples, output_dir, max_steps) -> str:
        self.calls.append({"base_model": base_model, "n": len(examples), "max_steps": max_steps})
        model_dir = Path(output_dir) / "forge_policy"
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir)


def _write_grpo(data_dir: Path, rows: list[dict]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    cols = ["episode_id", "task_name", "prompt", "completion", "total_reward", "passed", "per_step_rewards"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
    df.to_parquet(data_dir / "grpo_rollouts.parquet", index=False)


def _grpo_row(task, reward, passed=True, completion="c"):
    return {
        "episode_id": f"ep-{task}-{reward}", "task_name": task, "prompt": f"Task: {task}",
        "completion": completion, "total_reward": reward, "passed": passed,
        "per_step_rewards": json.dumps([reward]),
    }


def _write_prefs(data_dir: Path, pairs: list[tuple[float, float]]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "preference_pairs.jsonl").open("w") as f:
        for chosen_r, rejected_r in pairs:
            f.write(json.dumps({
                "chosen": [{"role": "user", "content": "Task: t"}, {"role": "assistant", "content": "good"}],
                "rejected": [{"role": "user", "content": "Task: t"}, {"role": "assistant", "content": "bad"}],
                "chosen_reward": chosen_r, "rejected_reward": rejected_r, "task": "t",
                "chosen_passed": True, "rejected_passed": False,
            }) + "\n")


# ---------------------------------------------------------------------------
# Happy path: GRPO and DPO wiring + checkpoint contract
# ---------------------------------------------------------------------------

def test_grpo_training_produces_a_loadable_checkpoint(tmp_path):
    data_dir = tmp_path / "data"
    _write_grpo(data_dir, [_grpo_row("t", 0.0), _grpo_row("t", 1.0), _grpo_row("t", 0.5)])
    backend = _FakeBackend()
    out = tmp_path / "out"

    result = PolicyTrainer(backend=backend).train(TrainingConfig(
        data_dir=data_dir, base_model="Qwen/Qwen2.5-3B", output_dir=out,
        objective=TrainingObjective.GRPO,
    ))

    assert backend.calls[0]["n"] == 3            # three signal-bearing rollouts
    assert result.objective == "grpo"
    # Checkpoint the runtime agents can load.
    ckpt = PolicyCheckpoint.load(out)
    assert ckpt.objective == "grpo" and ckpt.num_examples == 3
    assert Path(ckpt.model_path).exists()


def test_dpo_training_uses_preference_examples(tmp_path):
    data_dir = tmp_path / "data"
    _write_prefs(data_dir, [(1.0, 0.0), (0.9, 0.2)])
    backend = _FakeBackend()

    result = PolicyTrainer(backend=backend).train(TrainingConfig(
        data_dir=data_dir, base_model="base", output_dir=tmp_path / "out",
        objective=TrainingObjective.DPO,
    ))
    assert backend.calls[0]["n"] == 2
    assert result.objective == "dpo"


# ---------------------------------------------------------------------------
# Negative / false-positive: must NOT produce an update
# ---------------------------------------------------------------------------

def test_empty_rollouts_raise_no_signal_and_write_no_checkpoint(tmp_path):
    data_dir = tmp_path / "data"
    _write_grpo(data_dir, [])
    out = tmp_path / "out"
    with pytest.raises(NoTrainingSignalError):
        PolicyTrainer(backend=_FakeBackend()).train(TrainingConfig(
            data_dir=data_dir, base_model="base", output_dir=out,
            objective=TrainingObjective.GRPO,
        ))
    assert not (out / "policy_checkpoint.json").exists()


def test_all_equal_reward_rollouts_produce_no_update(tmp_path):
    # A graded set with no relative signal must NOT train (false-positive guard).
    data_dir = tmp_path / "data"
    _write_grpo(data_dir, [_grpo_row("t", 0.5), _grpo_row("t", 0.5), _grpo_row("t", 0.5)])
    backend = _FakeBackend()
    with pytest.raises(NoTrainingSignalError):
        PolicyTrainer(backend=backend).train(TrainingConfig(
            data_dir=data_dir, base_model="base", output_dir=tmp_path / "out",
            objective=TrainingObjective.GRPO,
        ))
    assert backend.calls == []  # backend never invoked


def test_all_tied_preferences_produce_no_update(tmp_path):
    data_dir = tmp_path / "data"
    _write_prefs(data_dir, [(0.5, 0.5), (1.0, 1.0)])
    with pytest.raises(NoTrainingSignalError):
        PolicyTrainer(backend=_FakeBackend()).train(TrainingConfig(
            data_dir=data_dir, base_model="base", output_dir=tmp_path / "out",
            objective=TrainingObjective.DPO,
        ))


def test_missing_export_raises_no_signal(tmp_path):
    with pytest.raises(NoTrainingSignalError):
        PolicyTrainer(backend=_FakeBackend()).train(TrainingConfig(
            data_dir=tmp_path / "nonexistent", base_model="base",
            output_dir=tmp_path / "out", objective=TrainingObjective.GRPO,
        ))


def test_malformed_export_raises(tmp_path):
    data_dir = tmp_path / "data"
    _write_grpo(data_dir, [{"episode_id": "x", "total_reward": 0.5}])  # missing columns
    with pytest.raises(MalformedExportError):
        PolicyTrainer(backend=_FakeBackend()).train(TrainingConfig(
            data_dir=data_dir, base_model="base", output_dir=tmp_path / "out",
            objective=TrainingObjective.GRPO,
        ))


# ---------------------------------------------------------------------------
# Dependency gating (mirrors task #1)
# ---------------------------------------------------------------------------

def test_default_backend_gates_on_missing_training_deps(tmp_path):
    import importlib.util

    if importlib.util.find_spec("trl") is not None:
        pytest.skip("trl is installed; gating message not exercised")

    data_dir = tmp_path / "data"
    _write_grpo(data_dir, [_grpo_row("t", 0.0), _grpo_row("t", 1.0)])
    # No backend injected → the real GPU-gated backend runs and should refuse.
    with pytest.raises(RuntimeError, match="pip install trl"):
        PolicyTrainer().train(TrainingConfig(
            data_dir=data_dir, base_model="base", output_dir=tmp_path / "out",
            objective=TrainingObjective.GRPO,
        ))
