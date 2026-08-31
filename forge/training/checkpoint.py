"""The serializable checkpoint contract produced by policy training.

A trained policy is a directory containing a `policy_checkpoint.json` manifest
that records what was trained, from how many graded examples, and the model
path/name the runtime agent should serve. Runtime loading intentionally lives
outside this training package so training never depends on an agent.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

MANIFEST_NAME = "policy_checkpoint.json"


class PolicyCheckpoint(BaseModel):
    """Manifest describing a policy trained from graded rollouts."""

    objective: str            # "grpo" | "dpo"
    base_model: str
    model_path: str           # what the runtime agent serves (local path or served name)
    num_examples: int
    mean_reward: float
    created_at: str = ""
    experiment_config: dict = Field(default_factory=dict)
    train_envs: list[str] = Field(default_factory=list)
    seed: int | None = None
    run_id: str = ""

    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        path = out_dir / MANIFEST_NAME
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, out_dir: Path) -> "PolicyCheckpoint":
        path = Path(out_dir) / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(f"no policy checkpoint manifest at {path}")
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
