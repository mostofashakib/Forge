"""Named reward contracts, so an ablation is a config change rather than a fork.

The point of a preset is that `VerifierComposer` and `TieredRewardEngine`
resolve the *same* name to the same behaviour. If the composer decided which
layers ran and the engine independently decided how they scored, "which reward
was this run graded under" would stop being answerable from the run's own
output -- which is the thing an ablation study most needs.

    full_layered_deterministic  all four layers, weighted partial credit
    binary_final_state          final-state checks plus the veto; reward is 0 or 1
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verifiers.auditor import RewardHackingAuditor
from verifiers.checks import Contract
from verifiers.episode import Episode
from verifiers.layered import DEFAULT_WEIGHTS, LayeredVerifier
from verifiers.results import EpisodeEvaluation, Penalty, RewardBreakdown


@dataclass(frozen=True, slots=True)
class RewardPreset:
    name: str
    weights: dict[str, float]
    #: Layers whose verifiers are built at all. A layer left out is not scored
    #: zero -- it does not run, and cannot be mistaken for a failure.
    layers: tuple[str, ...]
    #: Collapse the reward to 0.0 or 1.0 on the pass verdict.
    binary: bool = False
    #: Whether the auditor runs at all.
    audit: bool = True
    description: str = ""


REWARD_PRESETS: dict[str, RewardPreset] = {
    "full_layered_deterministic": RewardPreset(
        name="full_layered_deterministic",
        weights=dict(DEFAULT_WEIGHTS),
        layers=("final_state", "milestones", "trajectory", "negative"),
        description="The four structural layers, weighted partial credit. No model in the loop.",
    ),
    "binary_final_state": RewardPreset(
        name="binary_final_state",
        weights={"final_state": 1.0, "milestones": 0.0, "trajectory": 0.0,
                 "negative": 0.0},
        layers=("final_state", "negative"),
        binary=True,
        audit=False,
        description="Final-state checks only; reward is exactly 0 or 1.",
    ),
}

DEFAULT_PRESET = "full_layered_deterministic"


def resolve(preset: str | RewardPreset | None) -> RewardPreset:
    if isinstance(preset, RewardPreset):
        return preset
    name = preset or DEFAULT_PRESET
    if name not in REWARD_PRESETS:
        raise KeyError(
            f"unknown reward_preset {name!r}; known presets: {sorted(REWARD_PRESETS)}"
        )
    return REWARD_PRESETS[name]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerifierComposer:
    """Builds the `LayeredVerifier` a preset calls for, from one contract."""

    contract: Contract

    def compose(self, preset: str | RewardPreset | None = None) -> LayeredVerifier:
        chosen = resolve(preset)
        selected = tuple(
            verifier
            for group, layer in (
                (self.contract.final_state, "final_state"),
                (self.contract.milestones, "milestones"),
                (self.contract.trajectory, "trajectory"),
                (self.contract.negative, "negative"),
            )
            if layer in chosen.layers
            for verifier in group
        )
        return LayeredVerifier(
            name=f"{self.contract.name}:{chosen.name}",
            verifiers=selected,
            weights=dict(chosen.weights),
            milestone_events=self.contract.milestone_events,
            required_layers=("negative",),
        )


@dataclass(slots=True)
class TieredRewardEngine:
    """Runs a composed verifier under a preset and produces the evaluation."""

    contract: Contract
    preset: RewardPreset = field(default_factory=lambda: REWARD_PRESETS[DEFAULT_PRESET])
    auditor: RewardHackingAuditor | None = None

    @classmethod
    def for_preset(
        cls, contract: Contract, preset: str | RewardPreset | None = None,
        auditor: RewardHackingAuditor | None = None,
    ) -> "TieredRewardEngine":
        return cls(contract=contract, preset=resolve(preset), auditor=auditor)

    def evaluate(self, episode: Episode) -> EpisodeEvaluation:
        verifier = VerifierComposer(self.contract).compose(self.preset)
        breakdown, results, blocked, notes = verifier.evaluate(episode)

        notes = list(notes)
        valid = True
        # A layer that was asked for and could not run is a grading failure, not
        # a bad episode. Reporting it as a zero would put a broken grader into
        # an experiment as evidence about a model.
        for layer in breakdown.layers:
            if layer.layer in self.preset.layers and not layer.available:
                notes.append(f"{layer.layer} layer produced no verifiers")

        # Negative findings, in one pass. The verifier that found them decided
        # which kind they are; here they are only applied, in the order a reader
        # of the breakdown needs: what the layers earned, then what was charged,
        # then -- if anything forbidden happened -- whatever credit was left.
        #
        # Forbidden acts are not degrees of doing the task badly, so a veto takes
        # everything rather than charging a price: a price is something an agent
        # can decide to pay.
        charges = tuple(
            penalty for result in results if result.available
            for penalty in result.penalties
        )
        disqualified = tuple(
            check for result in results if result.available
            for check in result.disqualifying_checks
        )
        penalties = breakdown.penalties + charges

        if disqualified:
            blocked = True
            # One entry naming every violation rather than one each: splitting
            # the charge would print the second act at -0.0000, which reads as
            # though it were free.
            penalties += (
                Penalty(
                    reason="forbidden: " + ", ".join(disqualified),
                    amount=max(0.0, breakdown.base - sum(p.amount for p in penalties)),
                    source="forbidden",
                ),
            )
            notes.append(
                "disqualified: " + ", ".join(disqualified)
                + " -- a veto makes the whole episode ineligible for reward"
            )

        if penalties != breakdown.penalties:
            breakdown = RewardBreakdown(
                layers=breakdown.layers, penalties=penalties, base=breakdown.base,
            )

        # One place decides both, from the same number: a pass is a full reward
        # that nothing blocked. Deriving the verdict from the pre-penalty total
        # let a charged episode report a pass while scoring below one.
        passed = (not blocked) and breakdown.total >= verifier.pass_threshold

        # Diagnostic only. The auditor asks whether a pass looks earned and
        # says so in the result; it does not move the number. What it flags --
        # a short episode, one record hammered, writes without reads -- is
        # evidence for a reader, not a charge the grader is confident enough to
        # levy on its own.
        audit_report = None
        if self.preset.audit:
            auditor = self.auditor or RewardHackingAuditor.for_verifier(verifier)
            audit_report = auditor.audit(episode, passed)

        if self.preset.binary:
            reward = 1.0 if passed else 0.0
            breakdown = RewardBreakdown(
                layers=breakdown.layers, penalties=breakdown.penalties, base=reward,
            )
            reward = breakdown.total if not passed else 1.0
        else:
            reward = breakdown.total

        if not valid:
            reward = 0.0

        return EpisodeEvaluation(
            preset=self.preset.name,
            passed=passed and valid,
            reward=reward,
            breakdown=breakdown,
            results=results,
            audit=audit_report.as_dict() if audit_report else None,
            valid=valid,
            notes=tuple(notes),
        )


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Read an experiment file for its `reward_preset` and friends.

    Deliberately a tiny reader for flat `key: value` documents rather than a
    YAML dependency: the graded container is built without pip, and a verifier
    that cannot import its own config is a verifier that reports zero for
    reasons that have nothing to do with the agent. JSON is accepted too.
    """
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):
        return json.loads(text)

    config: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or line.startswith(" "):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        config[key.strip()] = _coerce(value.strip())
    return config


def _coerce(value: str) -> Any:
    if value in ("", "~", "null"):
        return None
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) > 1:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
