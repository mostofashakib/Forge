"""Heavy training backends — the actual gradient updates.

Kept separate from the orchestration (mirroring `forge/benchmark/_fine_tune.py`)
so the trainer's data loading, reward mapping, gating, and checkpoint contract
are testable without a GPU. Each backend gates on `trl` + `transformers` and,
until implemented on a GPU node, raises with actionable install guidance.

Distinct from task #1 (`forge/benchmark/`): that path is zero-shot transfer
*evaluation* on external suites; these backends *train* Forge's own policy from
its own graded experience via GRPO / DPO.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Protocol

_INSTALL_HINT = (
    "Install trl + transformers + datasets on a GPU node: "
    "pip install trl transformers datasets"
)


class TrainingBackend(Protocol):
    """Runs the gradient update and returns the checkpoint's model path."""

    def train(self, base_model: str, examples: list, output_dir: Path, max_steps: int) -> str: ...


def _require_training_deps() -> None:
    for pkg in ("trl", "transformers"):
        if importlib.util.find_spec(pkg) is None:
            raise RuntimeError(
                f"policy training requires '{pkg}', which is not installed. {_INSTALL_HINT}"
            )


class GRPOBackend:
    """GRPO trainer over group-relative-advantage examples (trl GRPOTrainer)."""

    def train(self, base_model: str, examples: list, output_dir: Path, max_steps: int) -> str:
        _require_training_deps()
        raise NotImplementedError(
            "GRPOBackend.train is not yet implemented. Implement it in "
            "forge/training/_backends.py on a GPU node using trl's GRPOTrainer, "
            "consuming the group-relative advantages from reward_mapping.grpo_advantages."
        )


class DPOBackend:
    """DPO trainer over chosen/rejected preference examples (trl DPOTrainer)."""

    def train(self, base_model: str, examples: list, output_dir: Path, max_steps: int) -> str:
        _require_training_deps()
        raise NotImplementedError(
            "DPOBackend.train is not yet implemented. Implement it in "
            "forge/training/_backends.py on a GPU node using trl's DPOTrainer, "
            "consuming the preference examples from reward_mapping.dpo_examples."
        )
