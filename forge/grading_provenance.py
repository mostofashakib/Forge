"""Generator/grader independence for Forge's verification stack.

Forge grades agents two different ways, and only one of them can be
contaminated by the model that authored the environment:

* **Structural verification** — final-state assertions, milestone order,
  necessary/unnecessary tool calls, forbidden side effects. These execute
  against recorded state and trajectories. No model issues a verdict, so a
  structural run needs no independence guarantee.
* **LLM judging** — the ``judge`` layer, semantic checks, and objective
  scoring. Here a model issues the verdict, and if that model shares a family
  with the one that generated the environment and its success conditions, the
  grade is not independent evidence.

This module makes that distinction explicit and auditable: it resolves which
models generated and which judged, decides whether the pair is independent, and
records the answer alongside every run result.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

# Models are named ``<family><variant>`` in every provider we support:
# ``claude-sonnet-4-6``, ``gpt-4o``, ``gemma4:26b``, ``llama-3.1-8b``. The
# leading alphabetic run identifies the family; the rest is a tier or version.
_LEADING_ALPHA = re.compile(r"^[a-z]+")

_JUDGE_MODEL_VAR = "FORGE_JUDGE_MODEL"
_JUDGE_PROVIDER_VAR = "FORGE_JUDGE_PROVIDER"


class GraderContaminationError(ValueError):
    """Raised when an LLM-graded run judges with the family that generated it."""


def model_family(model_id: str) -> str:
    """Return the vendor/family of ``model_id``.

    Different tiers of one family (Haiku vs. Sonnet, 4o vs. 4o-mini) collapse to
    the same family on purpose: swapping tiers changes cost, not provenance.
    """
    cleaned = model_id.strip().lower()
    if not cleaned:
        raise ValueError("model id cannot be blank")
    # ``Qwen/Qwen2.5-3B-Instruct`` — the namespace is the vendor.
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[0]
    match = _LEADING_ALPHA.match(cleaned)
    if not match:
        raise ValueError(
            f"cannot derive a model family from {model_id!r}: no alphabetic prefix"
        )
    return match.group(0)


@dataclass(frozen=True)
class GradingProvenance:
    """Which models authored the environment, and which model graded the agent."""

    generator_models: tuple[str, ...]
    judge_model: str | None
    llm_graded: bool

    @property
    def generator_families(self) -> tuple[str, ...]:
        seen: list[str] = []
        for model in self.generator_models:
            family = model_family(model)
            if family not in seen:
                seen.append(family)
        return tuple(seen)

    @property
    def judge_family(self) -> str | None:
        return model_family(self.judge_model) if self.judge_model else None

    @property
    def structural_only(self) -> bool:
        """True when no model issued a verdict in this run."""
        return not self.llm_graded

    @property
    def independent(self) -> bool:
        # Structural verification never asks a model for a grade, so there is
        # no generator/grader relationship to contaminate.
        if self.structural_only:
            return True
        if self.judge_family is None:
            return False
        return self.judge_family not in self.generator_families

    def as_record(self) -> dict:
        """JSON-safe provenance for the run result record."""
        return {
            "generator_models": list(self.generator_models),
            "generator_families": list(self.generator_families),
            "judge_model": self.judge_model,
            "judge_family": self.judge_family,
            "llm_graded": self.llm_graded,
            "independent": self.independent,
        }


def resolve_grading_provenance(
    *,
    llm_graded: bool,
    generator_models: tuple[str, ...] | None = None,
    judge_model: str | None = None,
) -> GradingProvenance:
    """Resolve provenance from the active environment.

    ``llm_graded`` must reflect whether a judge actually ran, not whether one is
    configured — a judge that never issues a verdict does not contaminate a
    structural result, and recording it as if it had would misstate the run.
    """
    if generator_models is None:
        from forge.extraction.llm_client import generation_models

        generator_models = generation_models()
    if not llm_graded:
        return GradingProvenance(
            generator_models=generator_models, judge_model=None, llm_graded=False
        )
    if judge_model is None:
        judge_model = os.environ.get(_JUDGE_MODEL_VAR) or None
        if judge_model is None:
            # Unset means every grading call falls back to the generation model,
            # so the generator grades its own work.
            judge_model = generator_models[0] if generator_models else None
    return GradingProvenance(
        generator_models=generator_models, judge_model=judge_model, llm_graded=True
    )


def require_independent_grader(provenance: GradingProvenance) -> None:
    """Raise when an LLM-graded run is judged by the generating family."""
    if provenance.independent:
        return
    if provenance.judge_family is None:
        raise GraderContaminationError(
            "this run uses LLM grading but no judge model is configured, so the "
            f"generating model grades its own environments; set {_JUDGE_MODEL_VAR} "
            f"(and {_JUDGE_PROVIDER_VAR} if the judge uses another provider) to a "
            "model outside "
            f"{sorted(provenance.generator_families)}"
        )
    raise GraderContaminationError(
        f"grader is not independent: judge {provenance.judge_model!r} shares family "
        f"{provenance.judge_family!r} with the generating models "
        f"{list(provenance.generator_models)}; set {_JUDGE_MODEL_VAR} to a model "
        "from another family, or run a structural-only reward preset"
    )
