"""Computer-Use Agent (CUA) verification adapter.

Verifies UI, browser, and GUI workflows: DOM element presence, URL navigation,
visual checkpoints, and forbidden UI interactions.
"""
from __future__ import annotations

from typing import Any

from forge.contracts.types import CheckResult, Task
from forge.contracts.verification import GroupVerificationSpec, VerifierGroup
from forge.runtime.trust_boundary.adapters.base import BaseVerifierGroupAdapter


class CUAAdapter(BaseVerifierGroupAdapter):
    """Adapter verifying Computer-Use and browser agent trajectories."""

    @property
    def group_type(self) -> VerifierGroup:
        return VerifierGroup.CUA

    def verify(
        self,
        spec: GroupVerificationSpec,
        state: dict[str, Any],
        trajectory: Any,
        task: Task | dict[str, Any] | None,
    ) -> list[CheckResult]:
        checks: list[CheckResult] = []
        rules = spec.rules
        events = self.extract_events(trajectory)
        actions = self.extract_actions(trajectory)

        # 1. URL Navigation Check
        expected_url = rules.get("expected_url") or rules.get("url")
        if expected_url:
            current_url = state.get("url") or state.get("current_url", "")
            passed = expected_url in current_url if isinstance(current_url, str) else False
            checks.append(CheckResult(
                name="cua:url_navigation",
                passed=passed,
                score=1.0 if passed else 0.0,
                evidence=f"Expected URL '{expected_url}', observed '{current_url}'",
            ))

        # 2. DOM Elements / Text Content
        expected_elements = rules.get("dom_elements", {})
        if isinstance(expected_elements, dict) and expected_elements:
            dom_state = state.get("dom", {}) or state.get("elements", {})
            for selector, expected_val in expected_elements.items():
                actual_val = dom_state.get(selector)
                if expected_val is True:
                    passed = selector in dom_state or actual_val is not None
                else:
                    passed = actual_val == expected_val or (
                        isinstance(actual_val, str) and str(expected_val) in actual_val
                    )
                checks.append(CheckResult(
                    name=f"cua:dom_element:{selector}",
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    evidence=f"Selector '{selector}': expected {expected_val!r}, found {actual_val!r}",
                ))

        # 3. Required UI Milestones (Events / Checkpoints)
        for milestone in spec.milestones:
            found = any(
                e.get("type") == milestone or e.get("name") == milestone or e.get("action") == milestone
                for e in events
            ) or any(
                a.get("type") == milestone or a.get("action") == milestone
                for a in actions
            )
            checks.append(CheckResult(
                name=f"cua:milestone:{milestone}",
                passed=found,
                score=1.0 if found else 0.0,
                evidence=f"UI milestone '{milestone}' {'reached' if found else 'missing'}",
            ))

        # 4. Forbidden UI Actions (Errors, Cancel, Popups)
        for forbidden in spec.forbidden:
            triggered = any(
                e.get("type") == forbidden or e.get("name") == forbidden for e in events
            ) or any(
                a.get("type") == forbidden for a in actions
            )
            checks.append(CheckResult(
                name=f"cua:forbidden:{forbidden}",
                passed=not triggered,
                score=1.0 if not triggered else 0.0,
                evidence=f"Forbidden action '{forbidden}' {'NOT triggered' if not triggered else 'TRIGGERED'}",
            ))

        # 5. Expected State Keys
        for key, expected_val in spec.expected_state.items():
            actual = state.get(key)
            passed = actual == expected_val
            checks.append(CheckResult(
                name=f"cua:state:{key}",
                passed=passed,
                score=1.0 if passed else 0.0,
                evidence=f"State '{key}': expected {expected_val!r}, got {actual!r}",
            ))

        # Default fallback check if spec is empty
        if not checks:
            checks.append(CheckResult(
                name="cua:basic_completion",
                passed=True,
                score=1.0,
                evidence="CUA interaction finished with clean state",
            ))

        return checks
