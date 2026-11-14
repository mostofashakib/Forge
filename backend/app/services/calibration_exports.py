"""A calibration run's attempts as a training dataset: one data file and its manifest, zipped.

File names match the episode export writers in `export_writers/`. Tasks a person
marked unfair, and suspect tasks nobody has marked hard, stay out, as they do
in the batch export. Every row passes the safety gate before it is written.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.models import CalibrationRun, TaskBatch
from backend.app.services import calibration_registry, task_judgments, task_registry
from backend.app.services.export_writers.gate import DatasetGate
from forge.contracts import RolloutRecord
from forge.taskfactory.calibration import FRONTIER_MAX, FRONTIER_MIN
from forge.taskfactory.rollout_datasets import Calibrated, dataset_rows

MANIFEST = "dataset_manifest.json"
FILES = {
    "grpo": "grpo_rollouts.parquet", "preference": "preference_pairs.jsonl",
    "sft": "sft_pairs.jsonl", "failures": "failure_dataset.jsonl",
}
# Nested GRPO columns are stored as JSON text, as the episode GRPO writer stores them.
_JSON_COLUMNS = ("per_step_rewards", "verification_results", "reward_breakdown")


def bundle(db: Session, batch: TaskBatch, run: CalibrationRun, fmt: str, gate: DatasetGate) -> bytes:
    """The zipped dataset. Raises ValueError for an unknown format."""
    held = task_judgments.held_out(db, batch.id)
    attempts = calibration_registry.attempts_by_task(db, run.id)
    calibrated = [
        Calibrated(task=task, attempts=attempts[task.id])
        for task in task_registry.batch_tasks(db, batch.id)
        if task.id in attempts and task.id not in held
    ]
    rows = dataset_rows(fmt, calibrated, env_name=batch.env_name, run_id=run.id, target_model=run.target_model)
    kept = [admitted for row in rows if (admitted := gate.admit(row, f"calibration {run.id} {fmt}")) is not None]
    manifest = {
        "manifest_version": "1.0.0",
        "source": "model_rollouts",
        "format": fmt,
        "files": [FILES[fmt]],
        "rows": len(kept),
        "batch_id": batch.id,
        "batch_version": batch.version,
        "env_name": batch.env_name,
        "calibration_run_id": run.id,
        "target_model": run.target_model,
        "attempts_per_task": run.attempts_per_task,
        "frontier": {"min": FRONTIER_MIN, "max": FRONTIER_MAX},
        "held_out": [{"task_id": task_id, "reason": reason} for task_id, reason in sorted(held.items())],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "safety": gate.stats(),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(FILES[fmt], _parquet(kept) if fmt == "grpo" else _jsonl(kept))
        archive.writestr(MANIFEST, json.dumps(manifest, indent=2))
    return buffer.getvalue()


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def _parquet(rows: list[dict]) -> bytes:
    stored = [{**row, **{column: json.dumps(row[column]) for column in _JSON_COLUMNS}} for row in rows]
    frame = pd.DataFrame(stored, columns=[*RolloutRecord.model_fields, "agent_id", "total_steps"])
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()
