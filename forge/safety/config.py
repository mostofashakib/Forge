"""Build the dataset safety service from environment variables.

Every misconfiguration raises `SafetyConfigError` naming the variable to fix.
There is no unscreened fallback: an export without a working gate does not run.
"""
from __future__ import annotations

import os
from typing import NamedTuple

from forge.extraction.llm_client import generation_models, get_judge_client
from forge.safety.adult import (
    DEFAULT_ADULT_MODEL,
    DEFAULT_ADULT_THRESHOLD,
    DistilBertAdultDetector,
    load_adult_pipeline,
)
from forge.safety.classifier import LLMSafetyClassifier
from forge.safety.ner import DEFAULT_NER_MODEL, DEFAULT_NER_THRESHOLD, GlinerNameDetector, load_gliner
from forge.safety.pseudonymizer import Pseudonymizer
from forge.safety.service import DatasetSafetyService, PiiScrubbing
from forge.safety.types import DataSource


class SafetyConfigError(RuntimeError):
    """The safety gate cannot be built, so nothing may be exported."""


class ConfiguredSafety(NamedTuple):
    service: DatasetSafetyService
    # "rules" or "rules+llm:<judge model>", recorded in the dataset manifest.
    mode: str


def safety_service_from_env(source: DataSource) -> ConfiguredSafety:
    """The gate for one pipeline. Only internet and human data load PII scrubbing."""
    adult = _adult_from_env()
    pii = _pii_from_env() if source.scrubs_pii else None
    if os.environ.get("FORGE_SAFETY_LLM", "0").strip() != "1":
        return ConfiguredSafety(DatasetSafetyService(adult, pii), "rules")
    try:
        classifier = LLMSafetyClassifier(get_judge_client())
    except Exception as exc:
        raise SafetyConfigError(f"FORGE_SAFETY_LLM=1 but the judge client could not be built: {exc}") from exc
    judge_model = os.environ.get("FORGE_JUDGE_MODEL") or generation_models()[0]
    return ConfiguredSafety(DatasetSafetyService(adult, pii, classifier), f"rules+llm:{judge_model}")


def _adult_from_env() -> DistilBertAdultDetector:
    threshold = _unit_float("FORGE_ADULT_THRESHOLD", DEFAULT_ADULT_THRESHOLD)
    model_id = os.environ.get("FORGE_ADULT_MODEL") or DEFAULT_ADULT_MODEL
    try:
        return DistilBertAdultDetector(load_adult_pipeline(model_id), threshold)
    except Exception as exc:
        raise SafetyConfigError(f"FORGE_ADULT_MODEL={model_id!r} could not be used: {exc}") from exc


def _pii_from_env() -> PiiScrubbing:
    key = os.environ.get("FORGE_SCRUB_KEY", "").strip()
    if not key:
        raise SafetyConfigError(
            "FORGE_SCRUB_KEY is not set. It keys PII pseudonyms and is required for internet and human data."
        )
    threshold = _unit_float("FORGE_NER_THRESHOLD", DEFAULT_NER_THRESHOLD)
    model_id = os.environ.get("FORGE_NER_MODEL") or DEFAULT_NER_MODEL
    try:
        model = load_gliner(model_id)
    except Exception as exc:
        raise SafetyConfigError(f"FORGE_NER_MODEL={model_id!r} could not be loaded: {exc}") from exc
    return PiiScrubbing(Pseudonymizer(key.encode()), GlinerNameDetector(model, threshold))


def _unit_float(name: str, default: float) -> float:
    raw = os.environ.get(name) or str(default)
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not 0.0 < value < 1.0:
        raise SafetyConfigError(f"{name} must be a number between 0 and 1, got {raw!r}")
    return value
