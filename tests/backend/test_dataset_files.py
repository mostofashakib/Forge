"""How exported rows are screened and stored: JSONL, GRPO parquet, and the manifest."""
from __future__ import annotations

import io
import json

import pandas as pd

from backend.app.services.export_writers.files import jsonl_lines, stored_rollout, write_jsonl, write_rollouts
from backend.app.services.export_writers.manifest import MANIFEST_FILE, dataset_manifest
from forge.contracts.dataset_rows import RolloutRow
from tests.backend.safety_helpers import make_gate


def _rollout(episode_id: str) -> dict:
    return RolloutRow(
        episode_id=episode_id, agent_id="a", total_steps=1, steps=1,
        verification_results=[{"passed": True}], reward_breakdown={"total_reward": 1.0},
    ).model_dump()


def test_the_gate_screens_a_stream_and_yields_only_kept_rows(tmp_path):
    gate = make_gate(tmp_path, "Sarah Chen")

    kept = list(gate.admitted([{"c": "Hi Sarah Chen"}, {"c": "rm -rf /"}, {"c": "fine"}], "unit"))

    assert len(kept) == 2 and "Sarah Chen" not in kept[0]["c"]
    assert gate.stats()["rows_seen"] == 3


def test_jsonl_has_one_json_object_per_line(tmp_path):
    rows = [{"a": 1}, {"b": [2]}]
    path = tmp_path / "rows.jsonl"

    write_jsonl(path, iter(rows))

    assert "".join(jsonl_lines(rows)) == '{"a": 1}\n{"b": [2]}\n'
    assert path.read_text() == '{"a": 1}\n{"b": [2]}\n'


def test_a_stored_rollout_keeps_nested_columns_as_json_text():
    stored = stored_rollout(_rollout("e1"))

    assert stored["verification_results"] == '[{"passed": true}]'
    assert stored["per_step_rewards"] == "[]"
    assert stored["reward_breakdown"] == '{"total_reward": 1.0}'


def test_rollouts_are_written_with_every_rollout_row_column_even_when_empty():
    for rows in ([stored_rollout(_rollout("e1"))], []):
        buffer = io.BytesIO()

        write_rollouts(buffer, rows)

        frame = pd.read_parquet(io.BytesIO(buffer.getvalue()))
        assert list(frame.columns) == list(RolloutRow.model_fields)
        assert len(frame) == len(rows)


def test_a_manifest_carries_the_version_files_time_and_safety_plus_its_own_fields():
    manifest = dataset_manifest(files=["a.jsonl"], safety={"rows_seen": 1}, source="model_rollouts")

    assert MANIFEST_FILE == "dataset_manifest.json"
    assert manifest["manifest_version"] == "1.0.0"
    assert (manifest["files"], manifest["safety"], manifest["source"]) == (["a.jsonl"], {"rows_seen": 1}, "model_rollouts")
    assert manifest["created_at"]
    json.dumps(manifest)
