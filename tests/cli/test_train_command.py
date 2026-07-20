"""`forge train` CLI: end-to-end wiring and error surfacing."""
from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from forge.cli.main import app

runner = CliRunner()


def _write_grpo(data_dir, rows):
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(data_dir / "grpo_rollouts.parquet", index=False)


def _row(reward, completion="c"):
    return {
        "episode_id": f"ep-{reward}-{completion}", "task_name": "t", "prompt": "Task: t",
        "completion": completion, "total_reward": reward, "passed": reward > 0,
        "per_step_rewards": json.dumps([reward]),
    }


def test_train_reports_no_signal_on_all_equal_rewards(tmp_path):
    data_dir = tmp_path / "data"
    _write_grpo(data_dir, [_row(0.5, "a"), _row(0.5, "b")])
    result = runner.invoke(app, [
        "train", "--data", str(data_dir), "--base-model", "base",
        "--output", str(tmp_path / "out"), "--objective", "grpo",
    ])
    assert result.exit_code == 1
    assert "no training signal" in result.output.lower()


def test_train_rejects_unknown_objective(tmp_path):
    result = runner.invoke(app, [
        "train", "--data", str(tmp_path), "--base-model", "base", "--objective", "sft",
    ])
    assert result.exit_code == 2
    assert "unknown objective" in result.output.lower()
