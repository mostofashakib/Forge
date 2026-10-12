"""The safety gate for datasets downloaded directly, without an export job.

Batch and trajectory downloads are model-generated, so they get harm checks and
secret redaction. Flagged rows go to a quarantine file under the environment's
`exports/downloads/` directory so false positives can be audited.
"""
from __future__ import annotations

from datetime import datetime, timezone

from backend.app.services.export_writers.gate import DatasetGate, gate_from_env
from forge.safety.types import DataSource
from forge.settings import generated_envs_root


def download_gate(env_name: str, label: str) -> DatasetGate:
    """A gate for one download. Raises SafetyConfigError when it cannot be built."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = generated_envs_root() / env_name / "exports" / "downloads" / f"{label}-{stamp}"
    return gate_from_env(out_dir, DataSource.MODEL)
