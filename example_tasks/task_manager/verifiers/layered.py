"""One verdict from four layers.

The composition rule is what gives the stack its meaning: partial credit is
allowed to describe *how much* of the work was done, and is never allowed to
overrule *whether it was done honestly*. A forbidden side effect or an
impossible ordering fails the episode outright, however high the other layers
score. That is what stops a correct end state reached by deleting the tasks that
would have contradicted it from grading as correct work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verifiers.checks import Contract, Verifier
from verifiers.episode import Episode
from verifiers.results import LAYERS, LayerScore, RewardBreakdown, VerifierResult

#: Outcome and critical-path completion dominate. Trajectory is only an
#: efficiency adjustment, so taking a tidy route cannot compensate for leaving
#: important work unfinished. Negative checks veto instead of earning points.
DEFAULT_WEIGHTS: dict[str, float] = {
    "final_state": 0.40,
    "milestones": 0.50,
    "trajectory": 0.10,
    "negative": 0.00,
}


@dataclass(frozen=True, slots=True)
class LayeredVerifier:
    """Composes the four grading paths into a single verdict."""

    name: str
    verifiers: tuple[Verifier, ...] = ()
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    #: The milestone list, kept so `RewardHackingAuditor.for_verifier` can
    #: inherit it instead of being handed a second copy that can drift.
    milestone_events: tuple[str, ...] = ()
    #: Layers that must pass for the episode to pass, whatever the score.
    required_layers: tuple[str, ...] = ("negative",)
    #: Below this, an episode has not done enough to be called a pass.
    pass_threshold: float = 0.999

    @classmethod
    def from_contract(
        cls, contract: Contract, weights: dict[str, float] | None = None, **kwargs: Any
    ) -> "LayeredVerifier":
        verifiers = (
            *contract.final_state, *contract.milestones, *contract.trajectory,
            *contract.negative,
        )
        return cls(
            name=contract.name,
            verifiers=tuple(verifiers),
            weights=dict(weights or DEFAULT_WEIGHTS),
            milestone_events=contract.milestone_events,
            **kwargs,
        )

    # -- running ---------------------------------------------------------

    def run(self, episode: Episode) -> tuple[VerifierResult, ...]:
        return tuple(verifier.verify(episode) for verifier in self.verifiers)

    def layer_scores(self, results: tuple[VerifierResult, ...]) -> tuple[LayerScore, ...]:
        """Average each layer's verifiers, then apply the layer's weight.

        A layer nobody configured, or one that could not run, is reported as
        unavailable and its weight is redistributed across the layers that did
        run.
        """
        by_layer: dict[str, list[VerifierResult]] = {layer: [] for layer in LAYERS}
        for result in results:
            by_layer.setdefault(result.layer, []).append(result)

        raw: list[LayerScore] = []
        for layer in LAYERS:
            weight = float(self.weights.get(layer, 0.0))
            found = by_layer.get(layer) or []
            usable = [result for result in found if result.available]
            if not found or (found and not usable):
                raw.append(LayerScore(layer, weight, 0.0, available=False))
                continue
            score = sum(result.score for result in usable) / len(usable)
            raw.append(LayerScore(layer, weight, score, available=True))

        live = [layer for layer in raw if layer.available and layer.weight > 0]
        live_weight = sum(layer.weight for layer in live)
        lost = sum(
            layer.weight for layer in raw if not layer.available and layer.weight > 0
        )
        if lost and live_weight:
            scale = (live_weight + lost) / live_weight
            raw = [
                LayerScore(layer.layer, layer.weight * scale, layer.score, True)
                if (layer.available and layer.weight > 0) else layer
                for layer in raw
            ]
        return tuple(raw)

    def evaluate(self, episode: Episode) -> tuple[RewardBreakdown, tuple[VerifierResult, ...], bool, tuple[str, ...]]:
        """Score the layers and say whether anything blocks a pass.

        Returns `blocked`, not `passed`: whether an episode passed depends on
        the reward after penalties, and the penalties are applied by the engine
        that owns the whole number. Deciding it here from the pre-penalty total
        reported a pass on a docked score.
        """
        results = self.run(episode)
        layers = self.layer_scores(results)
        base = sum(layer.contribution for layer in layers)
        breakdown = RewardBreakdown(layers=layers, penalties=(), base=base)

        notes: list[str] = []
        blocked = False
        for result in results:
            if result.fatal_on_failure and result.available and not result.passed:
                blocked = True
                notes.append(
                    f"{result.layer}/{result.verifier} is disqualifying: "
                    f"{', '.join(result.failed_checks)}"
                )
        for layer in self.required_layers:
            for result in results:
                if result.layer == layer and result.available and not result.passed:
                    blocked = True
        return breakdown, results, blocked, tuple(notes)
