"""Stream JSONL objects with file and line context on invalid records."""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def iter_jsonl_objects(path: Path, *, skip_invalid: bool = False) -> Iterator[dict]:
    """Read objects strictly, or log and skip bad records in partial traces."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Expected a JSON object")
            except ValueError as exc:
                reason = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                error = f"Invalid JSON object in {path} at line {line_number}: {reason}"
                if not skip_invalid:
                    raise ValueError(error) from exc
                logger.warning("%s", error)
                continue
            yield record
