from __future__ import annotations
import logging
from dataclasses import dataclass, field

import httpx

from forge.schema.state_schema import StateSchemaManifest

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    missing_fields: list[str] = field(default_factory=list)
    coverage_score: float = 0.0


class PostGenerationValidator:
    """Boots the container, exercises declared actions, verifies manifest contract."""

    def __init__(self, base_url: str, http_timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = http_timeout

    def validate(self, manifest: StateSchemaManifest) -> ValidationResult:
        with httpx.Client(base_url=self._base_url, timeout=self._timeout) as client:
            # Step 1: reset and check base state
            client.post("/forge/reset")
            state = client.get("/forge/state").json()
            missing = manifest.missing_fields(state)

            # Step 2: exercise derived_from actions; check derived fields appear
            derived_fields_to_check: dict[str, list[str]] = {}
            for fname, fspec in manifest.fields.items():
                for action in fspec.derived_from:
                    derived_fields_to_check.setdefault(action, []).append(fname)

            for action_endpoint, field_names in derived_fields_to_check.items():
                try:
                    client.post(f"/{action_endpoint}", json={})
                    state_after = client.get("/forge/state").json()
                    for fname in field_names:
                        if fname in state_after:
                            missing = [f for f in missing if f != fname]
                except Exception as exc:
                    logger.warning("[validator] could not exercise /%s: %s", action_endpoint, exc)

            coverage = manifest.coverage_score(state)
            passed = len(missing) == 0
            return ValidationResult(passed=passed, missing_fields=missing, coverage_score=coverage)
