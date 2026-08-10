"""Declarative experiment and result-record contracts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentConfig(BaseModel):
    """The complete train/held-out protocol for one experiment."""

    model_config = ConfigDict(extra="forbid")

    train_envs: list[str] = Field(min_length=1)
    heldout_envs: list[str] = Field(min_length=1)
    reward_preset: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_split(self) -> "ExperimentConfig":
        self.train_envs = _unique_names(self.train_envs, "train_envs")
        self.heldout_envs = _unique_names(self.heldout_envs, "heldout_envs")
        overlap = sorted(set(self.train_envs) & set(self.heldout_envs))
        if overlap:
            raise ValueError(f"train_envs and heldout_envs overlap: {overlap}")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"experiment config not found: {config_path}")
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid experiment YAML {config_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"experiment config must be a YAML mapping: {config_path}")
        return cls.model_validate(raw)


class RunResult(BaseModel):
    """Machine-readable headline outcomes for one seeded train/eval run."""

    config: dict[str, Any]
    seed: int
    heldout_pass_rate: float = Field(ge=0.0, le=1.0)
    reward_hacking_rate: float = Field(ge=0.0, le=1.0)
    reward_variance: float = Field(ge=0.0)

    def save(self, runs_dir: str | Path, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id must be a non-empty path-safe name")
        path = Path(runs_dir) / run_id / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            self.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        temporary_path.replace(path)
        return path


def _unique_names(names: list[str], field_name: str) -> list[str]:
    cleaned = [name.strip() for name in names]
    if any(not name for name in cleaned):
        raise ValueError(f"{field_name} cannot contain blank environment names")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{field_name} must not contain duplicates")
    return cleaned
