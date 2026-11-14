"""How screened rows are stored: JSON Lines, and GRPO rollouts as parquet."""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from forge.contracts.dataset_rows import RolloutRow

# Nested rollout columns are stored as JSON text, so every parquet reader sees flat columns.
_JSON_COLUMNS = ("per_step_rewards", "verification_results", "reward_breakdown")


def jsonl_lines(rows: Iterable[dict]) -> Iterator[str]:
    for row in rows:
        yield json.dumps(row) + "\n"


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Stream rows to a file, one JSON object per line."""
    with path.open("w") as handle:
        handle.writelines(jsonl_lines(rows))


def stored_rollout(row: dict) -> dict:
    """A `RolloutRow` dump as it is stored, with nested columns as JSON text."""
    return {**row, **{column: json.dumps(row[column]) for column in _JSON_COLUMNS}}


def write_rollouts(target: Path | BinaryIO, rows: list[dict]) -> None:
    """Stored rollouts as parquet, with every `RolloutRow` column even when there are no rows."""
    pd.DataFrame(rows, columns=list(RolloutRow.model_fields)).to_parquet(target, index=False)
