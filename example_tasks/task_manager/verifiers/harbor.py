"""Turning a verdict into the flat reward stream a harness stores.

Harbor keeps one number per key per episode, so this is where the structured
`EpisodeEvaluation` is flattened. Two properties are worth the care:

The key set never varies. A blank reward declares exactly the keys a graded one
does, so a missing key can only ever mean "this run did not finish", never
"this run scored zero on something the analysis forgot about".

Every layer is reported next to the weight it carried. Weight is redistributed
when a configured structural layer cannot run.
"""

from __future__ import annotations


from verifiers.results import LAYERS, EpisodeEvaluation

#: The verdict itself, then the terms that produced it.
HEADLINE_KEYS = (
    "reward", "success", "valid", "base_reward", "penalty_total",
    "audit_pass", "audit_findings",
)


def layer_keys() -> tuple[str, ...]:
    return tuple(
        key
        for layer in LAYERS
        for key in (f"layer_{layer}", f"layer_weight_{layer}", f"layer_available_{layer}")
    )


def reward_keys() -> tuple[str, ...]:
    """Every key the reward stream carries. There is no extension point: a
    second grader's keys travelling beside these is a second answer to how the
    run did, and only one of them was ever the reward."""
    return (*HEADLINE_KEYS, *layer_keys())


def blank_reward_dict(valid: float) -> dict[str, float]:
    """Every key at zero except `valid`, which says why there is nothing here."""
    blank = {key: 0.0 for key in reward_keys()}
    blank["valid"] = float(valid)
    # Nothing was graded, so nothing was found suspicious. Reporting an audit
    # failure for a run that never happened would be an accusation.
    blank["audit_pass"] = 1.0
    return blank


def reward_dict(evaluation: EpisodeEvaluation) -> dict[str, float]:
    flat = blank_reward_dict(valid=float(evaluation.valid))
    flat.update({
        "reward": round(float(evaluation.reward), 6),
        "success": float(evaluation.passed),
        "base_reward": round(float(evaluation.breakdown.base), 6),
        "penalty_total": round(float(evaluation.breakdown.penalty_total), 6),
    })
    for layer in evaluation.breakdown.layers:
        flat[f"layer_{layer.layer}"] = round(float(layer.score), 6)
        flat[f"layer_weight_{layer.layer}"] = round(float(layer.weight), 6)
        flat[f"layer_available_{layer.layer}"] = float(layer.available)
    if evaluation.audit is not None:
        flat["audit_pass"] = float(evaluation.audit.get("audited_pass", True))
        flat["audit_findings"] = float(len(evaluation.audit.get("findings", ())))
    return flat


__all__ = ["HEADLINE_KEYS", "blank_reward_dict", "layer_keys", "reward_dict", "reward_keys"]
