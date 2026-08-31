"""Authoritative result of grading one completed episode."""
from __future__ import annotations

from pydantic import BaseModel, Field

from forge.contracts.types import RewardBreakdown, VerificationResult


class EpisodeEvaluation(BaseModel):
    """One final verdict shared by runtimes, telemetry, exports, and training."""

    passed: bool
    reward: RewardBreakdown
    verification_results: list[VerificationResult] = Field(default_factory=list)
    reason: str

    @property
    def total_reward(self) -> float:
        return self.reward.total_reward
