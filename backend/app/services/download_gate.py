"""The safety gate for datasets downloaded directly, without an export job.

Batch and trajectory downloads are model-generated, so they get harm checks and
secret redaction. Flagged rows go to a quarantine file under the environment's
`exports/downloads/` directory so false positives can be audited.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone

from backend.app.services.export_writers.gate import DatasetGate, gate_from_env
from forge.safety.types import DataSource
from forge.settings import generated_envs_root

logger = logging.getLogger(__name__)


def download_gate(env_name: str, label: str) -> DatasetGate:
    """A gate for one download. Raises SafetyConfigError when it cannot be built."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = generated_envs_root() / env_name / "exports" / "downloads" / f"{label}-{stamp}"
    return gate_from_env(out_dir, DataSource.MODEL)


def screened_lines(lines: Iterable[str], gate: DatasetGate, source: str) -> Iterator[str]:
    """JSONL lines that passed the gate, scrubbed. A line that is not a JSON object cannot be screened and is dropped."""
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        if not isinstance(record, dict):
            logger.warning("Dropping an unscreenable trajectory line from %s", source)
            continue
        kept = gate.admit(record, source)
        if kept is not None:
            yield json.dumps(kept) + "\n"
    logger.info("Screened download %s: %s", source, gate.stats())
