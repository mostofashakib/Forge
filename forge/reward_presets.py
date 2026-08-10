"""Canonical reward-ablation presets shared by every Forge reward path."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RewardPreset(str, Enum):
    FULL_LAYERED_PARTIAL = "full_layered_partial"
    BINARY_FINAL_STATE = "binary_final_state"
    JUDGE_ONLY = "judge_only"
    FULL_NO_AUDITOR = "full_no_auditor"


@dataclass(frozen=True)
class RewardPresetSpec:
    enabled_layers: frozenset[str]
    scoring_mode: str
    auditor_enabled: bool
    binary_final_state: bool = False
    judge_only: bool = False


_ALL_LAYERS = frozenset({"state", "invariant", "trajectory", "judge", "negative"})
_PRESETS: dict[RewardPreset, RewardPresetSpec] = {
    RewardPreset.FULL_LAYERED_PARTIAL: RewardPresetSpec(
        enabled_layers=_ALL_LAYERS,
        scoring_mode="partial",
        auditor_enabled=True,
    ),
    RewardPreset.BINARY_FINAL_STATE: RewardPresetSpec(
        enabled_layers=frozenset({"state"}),
        scoring_mode="binary",
        auditor_enabled=False,
        binary_final_state=True,
    ),
    RewardPreset.JUDGE_ONLY: RewardPresetSpec(
        enabled_layers=frozenset({"judge"}),
        scoring_mode="partial",
        auditor_enabled=False,
        judge_only=True,
    ),
    RewardPreset.FULL_NO_AUDITOR: RewardPresetSpec(
        enabled_layers=_ALL_LAYERS,
        scoring_mode="partial",
        auditor_enabled=False,
    ),
}


def reward_preset_spec(preset: RewardPreset | str) -> RewardPresetSpec:
    return _PRESETS[RewardPreset(preset)]
