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

from forge.contracts import RolloutRecord


class MalformedExportError(ValueError):
    """An export file exists but does not match the expected schema."""


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
    env_name: str = ""


@dataclass
class SFTRecord:
    """One supervised demonstration for SFT training."""

    prompt: str
    completion: str
    env_name: str = ""
    reward: float = 1.0
    passed: bool = True
    behavior_model: str = ""


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
            env_name=str(row.get("env_name") or _environment_of(str(row["prompt"]))),
            task_name=str(row["task_name"]),
            prompt=str(row["prompt"]),
            completion=str(row["completion"]),
            total_reward=float(row["total_reward"]),
            passed=bool(row["passed"]),
            per_step_rewards=_parse_per_step(row.get("per_step_rewards")),
            behavior_model=str(row.get("behavior_model") or ""),
            termination_reason=str(row.get("termination_reason") or "unknown"),
            verification_results=_parse_json_list(row.get("verification_results")),
            reward_breakdown=_parse_json_dict(row.get("reward_breakdown")),
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


def _parse_json_list(value) -> list[dict]:
    parsed = _parse_json(value)
    return parsed if isinstance(parsed, list) else []


def _parse_json_dict(value) -> dict:
    parsed = _parse_json(value)
    return parsed if isinstance(parsed, dict) else {}


def _parse_json(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


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
                    env_name=str(obj.get("env_name") or _environment_of(_prompt_of(obj["chosen"]))),
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


def _environment_of(prompt: str) -> str:
    """Read the environment marker used by all Forge export writers."""
    for line in prompt.splitlines():
        if line.startswith("Environment:"):
            return line.partition(":")[2].strip()
    return ""


def load_sft(path_or_dir: Path) -> list[SFTRecord]:
    """Load SFT demonstrations from jsonl, batch export json, or rollouts parquet."""
    path = Path(path_or_dir)
    if path.is_dir():
        # Check files in priority
        candidates = [
            path / "sft_pairs.jsonl",
            path / "sft_dataset.jsonl",
            path / "sft_data.jsonl",
            path / "batch_export.json",
            path / "tasks.json",
            path / "preference_pairs.jsonl",
            path / "grpo_rollouts.parquet",
        ]
        target_file = next((c for c in candidates if c.exists()), None)
        if target_file is None:
            # Check for any .jsonl or .json file
            target_file = next(iter(path.glob("*.jsonl")), next(iter(path.glob("*.json")), None))
        if target_file is None:
            return []
        path = target_file

    if path.suffix == ".parquet":
        rollouts = load_rollouts(path)
        # Use rollouts that passed or scored reward > 0
        valid = [r for r in rollouts if r.passed or r.total_reward > 0]
        return [
            SFTRecord(
                prompt=r.prompt,
                completion=r.completion,
                env_name=r.env_name,
                reward=r.total_reward,
                passed=r.passed,
                behavior_model=r.behavior_model,
            )
            for r in valid
        ]

    if path.suffix == ".json":
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            raise MalformedExportError(f"could not load json {path}: {exc}") from exc

        records: list[SFTRecord] = []
        if isinstance(data, dict):
            # Check for sft_items
            if "sft_items" in data and isinstance(data["sft_items"], list):
                for item in data["sft_items"]:
                    records.append(SFTRecord(
                        prompt=str(item.get("prompt", "")),
                        completion=str(item.get("completion", "")),
                        env_name=str(data.get("env_name", "")),
                    ))
            # Check for tasks
            elif "tasks" in data and isinstance(data["tasks"], list):
                for task in data["tasks"]:
                    prompt = str(task.get("objective") or task.get("prompt") or "")
                    golden = task.get("golden", [])
                    completion = "\n".join(
                        f"$ {s.get('command') or s.get('tool', '')}" if isinstance(s, dict) else str(s)
                        for s in golden
                    ) if golden else str(task.get("completion", ""))
                    records.append(SFTRecord(
                        prompt=prompt,
                        completion=completion,
                        env_name=str(data.get("env_name", "")),
                    ))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    records.append(SFTRecord(
                        prompt=str(item.get("prompt", "")),
                        completion=str(item.get("completion", "")),
                        env_name=str(item.get("env_name", "")),
                    ))
        return [r for r in records if r.prompt and r.completion]

    # jsonl parsing
    records = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "messages" in obj and isinstance(obj["messages"], list):
                    prompt = _prompt_of(obj["messages"])
                    completion = _assistant_of(obj["messages"])
                    records.append(SFTRecord(
                        prompt=prompt,
                        completion=completion,
                        env_name=_environment_of(prompt),
                        reward=float(obj.get("total_reward", 1.0)),
                    ))
                elif "prompt" in obj and "completion" in obj:
                    records.append(SFTRecord(
                        prompt=str(obj["prompt"]),
                        completion=str(obj["completion"]),
                        env_name=str(obj.get("env_name", "") or _environment_of(str(obj["prompt"]))),
                        reward=float(obj.get("reward", 1.0)),
                    ))
                elif "chosen" in obj and isinstance(obj["chosen"], list):
                    prompt = _prompt_of(obj["chosen"])
                    completion = _assistant_of(obj["chosen"])
                    records.append(SFTRecord(
                        prompt=prompt,
                        completion=completion,
                        env_name=str(obj.get("env_name", "") or _environment_of(prompt)),
                        reward=float(obj.get("chosen_reward", 1.0)),
                    ))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise MalformedExportError(f"sft jsonl {path} line {lineno} malformed: {exc}") from exc
    return [r for r in records if r.prompt and r.completion]
