"""Loaders read the exact export-writer formats and reject malformed files."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from forge.training.dataset import (
    MalformedExportError,
    load_preferences,
    load_rollouts,
)


def _write_rollouts(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_load_rollouts_round_trips_grpo_parquet(tmp_path):
    path = tmp_path / "grpo_rollouts.parquet"
    _write_rollouts(path, [{
        "episode_id": "ep1", "env_name": "e", "task_name": "t", "seed": 0,
        "agent_id": "random", "prompt": "Task: t", "completion": "$ ls",
        "total_reward": 0.8, "passed": True, "total_steps": 3,
        "per_step_rewards": json.dumps([0.1, 0.3, 0.4]),
    }])
    records = load_rollouts(path)
    assert len(records) == 1
    r = records[0]
    assert r.episode_id == "ep1" and r.task_name == "t"
    assert r.total_reward == 0.8 and r.passed is True
    assert r.per_step_rewards == [0.1, 0.3, 0.4]


def test_load_rollouts_rejects_missing_columns(tmp_path):
    path = tmp_path / "grpo_rollouts.parquet"
    _write_rollouts(path, [{"episode_id": "ep1", "total_reward": 0.5}])
    with pytest.raises(MalformedExportError):
        load_rollouts(path)


def test_load_preferences_parses_chosen_rejected(tmp_path):
    path = tmp_path / "preference_pairs.jsonl"
    record = {
        "chosen": [
            {"role": "user", "content": "Task: t"},
            {"role": "assistant", "content": "$ good"},
        ],
        "rejected": [
            {"role": "user", "content": "Task: t"},
            {"role": "assistant", "content": "$ bad"},
        ],
        "chosen_reward": 1.0, "rejected_reward": 0.0,
        "task": "t", "chosen_passed": True, "rejected_passed": False,
    }
    path.write_text(json.dumps(record) + "\n")
    prefs = load_preferences(path)
    assert len(prefs) == 1
    p = prefs[0]
    assert p.prompt == "Task: t"
    assert p.chosen == "$ good" and p.rejected == "$ bad"
    assert p.chosen_reward == 1.0 and p.rejected_reward == 0.0


def test_load_preferences_rejects_malformed_line(tmp_path):
    path = tmp_path / "preference_pairs.jsonl"
    path.write_text('{"chosen": [], "rejected": []}\n')  # missing rewards
    with pytest.raises(MalformedExportError):
        load_preferences(path)
