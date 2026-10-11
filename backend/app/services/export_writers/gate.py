"""The safety gate every dataset row passes before it is written.

Kept rows come back scrubbed. Flagged rows go, scrubbed, to a quarantine file
next to the dataset so false positives can be audited, and never into it.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from forge.safety.config import safety_service_from_env
from forge.safety.service import DatasetSafetyService

QUARANTINE_FILE = "safety_quarantine.jsonl"


class DatasetGate:
    def __init__(self, service: DatasetSafetyService, quarantine_path: Path, mode: str) -> None:
        self._service = service
        self._quarantine_path = quarantine_path
        self._mode = mode
        self._seen = 0
        self._kept = 0
        self._replacements: Counter[str] = Counter()
        self._quarantined_by_rule: Counter[str] = Counter()

    def admit(self, record: dict, source: str) -> dict | None:
        """The scrubbed row when it may enter the dataset, else None."""
        result = self._service.screen(record)
        self._seen += 1
        self._replacements.update(result.replacements)
        if result.kept:
            self._kept += 1
            return result.record
        self._quarantined_by_rule.update({finding.rule for finding in result.findings})
        entry = {
            "source": source,
            "findings": [{"rule": f.rule, "detail": f.detail} for f in result.findings],
            "record": result.record,
        }
        with self._quarantine_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        return None

    def stats(self) -> dict:
        return {
            "mode": self._mode,
            "rows_seen": self._seen,
            "rows_kept": self._kept,
            "rows_quarantined": self._seen - self._kept,
            "replacements": dict(self._replacements),
            "quarantined_by_rule": dict(self._quarantined_by_rule),
            "quarantine_file": self._quarantine_path.name if self._quarantine_path.exists() else None,
        }


def gate_from_env(out_dir: Path) -> DatasetGate:
    """A gate configured from env. Raises SafetyConfigError when it cannot be built."""
    safety = safety_service_from_env()
    return DatasetGate(safety.service, out_dir / QUARANTINE_FILE, safety.mode)
