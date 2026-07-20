"""Typed loaders for Forge's exported graded rollouts.

These read the exact files the export writers produce
(`backend/app/services/export_writers/`): `grpo_rollouts.parquet` and
`preference_pairs.jsonl`. Malformed exports raise :class:`MalformedExportError`
so the trainer fails with an actionable message instead of a cryptic one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class MalformedExportError(ValueError):
    """An export file exists but does not match the expected schema."""


@dataclass
class RolloutRecord:
    """One graded episode from `grpo_rollouts.parquet`."""

    episode_id: str
    task_name: str
    prompt: str
    completion: str
    total_reward: float
    passed: bool
    per_step_rewards: list[float]


@dataclass
class PreferenceRecord:
    """One chosen/rejected pair from `preference_pairs.jsonl`."""

    task: str
    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    chosen_passed: bool
    rejected_passed: bool


_ROLLOUT_COLUMNS = {"episode_id", "task_name", "prompt", "completion", "total_reward", "passed"}


def load_rollouts(path: Path) -> list[RolloutRecord]:
    """Load graded episodes from a `grpo_rollouts.parquet` file."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas ships with the app
        raise RuntimeError("loading rollouts requires pandas") from exc

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise MalformedExportError(f"could not read rollouts parquet {path}: {exc}") from exc

    missing = _ROLLOUT_COLUMNS - set(df.columns)
    if missing:
        raise MalformedExportError(f"rollouts {path} missing columns: {sorted(missing)}")

    records: list[RolloutRecord] = []
    for row in df.to_dict("records"):
        records.append(RolloutRecord(
            episode_id=str(row["episode_id"]),
            task_name=str(row["task_name"]),
            prompt=str(row["prompt"]),
            completion=str(row["completion"]),
            total_reward=float(row["total_reward"]),
            passed=bool(row["passed"]),
            per_step_rewards=_parse_per_step(row.get("per_step_rewards")),
        ))
    return records


def _parse_per_step(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return []


def load_preferences(path: Path) -> list[PreferenceRecord]:
    """Load chosen/rejected pairs from a `preference_pairs.jsonl` file."""
    records: list[PreferenceRecord] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(PreferenceRecord(
                    task=str(obj.get("task", "")),
                    prompt=_prompt_of(obj["chosen"]),
                    chosen=_assistant_of(obj["chosen"]),
                    rejected=_assistant_of(obj["rejected"]),
                    chosen_reward=float(obj["chosen_reward"]),
                    rejected_reward=float(obj["rejected_reward"]),
                    chosen_passed=bool(obj.get("chosen_passed", False)),
                    rejected_passed=bool(obj.get("rejected_passed", False)),
                ))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise MalformedExportError(
                    f"preference_pairs {path} line {lineno} is malformed: {exc}"
                ) from exc
    return records


def _prompt_of(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(messages[0].get("content", "")) if messages else ""


def _assistant_of(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""
