"""Build the dataset safety service from environment variables.

Every misconfiguration raises `SafetyConfigError` naming the variable to fix.
There is no unscrubbed fallback: an export without a working gate does not run.
"""
from __future__ import annotations

import os
from typing import NamedTuple

from forge.extraction.llm_client import generation_models, get_judge_client
from forge.safety.classifier import LLMSafetyClassifier
from forge.safety.ner import DEFAULT_NER_MODEL, GlinerNameDetector, load_gliner
from forge.safety.pseudonymizer import Pseudonymizer
from forge.safety.service import DatasetSafetyService

_DEFAULT_THRESHOLD = 0.5


class SafetyConfigError(RuntimeError):
    """The safety gate cannot be built, so nothing may be exported."""


class ConfiguredSafety(NamedTuple):
    service: DatasetSafetyService
    # "rules" or "rules+llm:<judge model>", recorded in the dataset manifest.
    mode: str


def safety_service_from_env() -> ConfiguredSafety:
    pseudonymizer = Pseudonymizer(_scrub_key())
    names = GlinerNameDetector(_load_model(), _threshold())
    if os.environ.get("FORGE_SAFETY_LLM", "0").strip() != "1":
        return ConfiguredSafety(DatasetSafetyService(pseudonymizer, names), "rules")
    try:
        classifier = LLMSafetyClassifier(get_judge_client())
    except Exception as exc:
        raise SafetyConfigError(f"FORGE_SAFETY_LLM=1 but the judge client could not be built: {exc}") from exc
    judge_model = os.environ.get("FORGE_JUDGE_MODEL") or generation_models()[0]
    return ConfiguredSafety(DatasetSafetyService(pseudonymizer, names, classifier), f"rules+llm:{judge_model}")


def _scrub_key() -> bytes:
    key = os.environ.get("FORGE_SCRUB_KEY", "").strip()
    if not key:
        raise SafetyConfigError("FORGE_SCRUB_KEY is not set. It keys PII pseudonyms and is required to export data.")
    return key.encode()


def _load_model():
    model_id = os.environ.get("FORGE_NER_MODEL") or DEFAULT_NER_MODEL
    try:
        return load_gliner(model_id)
    except Exception as exc:
        raise SafetyConfigError(f"FORGE_NER_MODEL={model_id!r} could not be loaded: {exc}") from exc


def _threshold() -> float:
    raw = os.environ.get("FORGE_NER_THRESHOLD") or str(_DEFAULT_THRESHOLD)
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not 0.0 < value < 1.0:
        raise SafetyConfigError(f"FORGE_NER_THRESHOLD must be a number between 0 and 1, got {raw!r}")
    return value
