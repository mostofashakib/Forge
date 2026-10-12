"""Model families, for keeping generator and grader models independent.

A model grading environments authored by its own family is not independent
evidence. `model_family` collapses tiers of one vendor (Haiku and Sonnet, 4o and
4o-mini) so callers can compare provenance, and `GraderContaminationError` is
what they raise when a grader shares the generator's family.
"""
from __future__ import annotations

import re

# Models are named ``<family><variant>`` in every provider we support:
# ``claude-sonnet-4-6``, ``gpt-4o``, ``gemma4:26b``, ``llama-3.1-8b``. The
# leading alphabetic run identifies the family; the rest is a tier or version.
_LEADING_ALPHA = re.compile(r"^[a-z]+")


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
