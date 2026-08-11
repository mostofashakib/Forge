"""Build a quorum of independent LLM members from configuration.

``FORGE_QUORUM_MODELS`` is a comma-separated ``provider:model`` list. Empty means
no quorum: validation falls back to the single judge configured by
``FORGE_JUDGE_MODEL``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from forge.grading_provenance import GraderContaminationError, model_family
from forge.validation.member import MemberVerdict

QUORUM_MODELS_VAR = "FORGE_QUORUM_MODELS"

_SUPPORTED_PROVIDERS = frozenset({"anthropic", "ollama", "openai", "gemini"})

# Takes a configured client and the subject under validation. Returns either
# the verdict alone, or ``(verdict, detail)`` to record the reasoning.
EvaluateFn = Callable[[Any, Any], "bool | tuple[bool, str]"]


@dataclass(frozen=True)
class QuorumSpec:
    provider: str
    model: str

    @property
    def family(self) -> str:
        return model_family(self.model)


def parse_quorum_models(raw: str) -> tuple[QuorumSpec, ...]:
    """Parse a ``provider:model,provider:model`` list into specs.

    Rejects repeats and same-family members: two checkpoints of one family are
    one opinion counted twice, and a quorum that double-counts a family is not
    measuring independent agreement.
    """
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    specs: list[QuorumSpec] = []
    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                f"quorum entry {entry!r} must be written as provider:model"
            )
        provider, _, model = entry.partition(":")
        provider, model = provider.strip().lower(), model.strip()
        if not model:
            raise ValueError(f"quorum entry {entry!r} has a blank model")
        if provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"unknown provider {provider!r} in quorum entry {entry!r}; "
                f"valid: {sorted(_SUPPORTED_PROVIDERS)}"
            )
        spec = QuorumSpec(provider=provider, model=model)
        if spec in specs:
            raise ValueError(f"duplicate quorum member: {entry!r}")
        clash = next((s for s in specs if s.family == spec.family), None)
        if clash is not None:
            raise ValueError(
                f"quorum members {clash.model!r} and {spec.model!r} share family "
                f"{spec.family!r}; members must come from different families to "
                "count as independent opinions"
            )
        specs.append(spec)
    return tuple(specs)


def configured_quorum() -> tuple[QuorumSpec, ...]:
    """The quorum declared by the environment, empty when none is configured."""
    return parse_quorum_models(os.environ.get(QUORUM_MODELS_VAR, ""))


class _LLMMember:
    """One LLM in a quorum. Abstains rather than failing the run on error."""

    def __init__(self, spec: QuorumSpec, client: Any, evaluate: EvaluateFn) -> None:
        self.member_id = spec.model
        self.family = spec.family
        self._client = client
        self._evaluate = evaluate

    def evaluate(self, subject: Any) -> MemberVerdict:
        try:
            result = self._evaluate(self._client, subject)
        except Exception as exc:
            # A provider outage is missing evidence, not evidence of failure.
            # Abstaining keeps it out of the agreement denominator instead of
            # silently counting as a vote against.
            return MemberVerdict(
                member_id=self.member_id, family=self.family,
                passed=None, detail=f"abstained: {exc}",
            )
        # An evaluate function may return the verdict alone or paired with the
        # reasoning behind it. Unpacking is not optional: a (False, "...") tuple
        # is truthy, so treating the result as a bare bool would silently turn
        # every failing verdict into a pass.
        if isinstance(result, tuple):
            passed, detail = result
        else:
            passed, detail = result, ""
        return MemberVerdict(
            member_id=self.member_id, family=self.family,
            passed=bool(passed), detail=detail,
        )


def _default_client_factory(spec: QuorumSpec):
    from forge.extraction.llm_client import get_client

    return get_client(model=spec.model, provider=spec.provider)


def quorum_members(
    specs: Sequence[QuorumSpec],
    *,
    generator_families: tuple[str, ...],
    evaluate: EvaluateFn,
    client_factory: Callable[[QuorumSpec], Any] = _default_client_factory,
) -> tuple[_LLMMember, ...]:
    """Build quorum members, refusing any that share the generator's family."""
    if not specs:
        raise ValueError(
            f"no quorum members configured; set {QUORUM_MODELS_VAR} to a "
            "provider:model list"
        )
    generators = set(generator_families)
    contaminated = [spec for spec in specs if spec.family in generators]
    if contaminated:
        raise GraderContaminationError(
            f"quorum members {[s.model for s in contaminated]} share family "
            f"{sorted({s.family for s in contaminated})} with the generating "
            "models; a quorum drawn from the generating family is not "
            "independent evidence"
        )
    return tuple(_LLMMember(spec, client_factory(spec), evaluate) for spec in specs)
