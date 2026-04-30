from __future__ import annotations
import logging
from dataclasses import dataclass

from forge.extraction.schemas import PolicyRule

logger = logging.getLogger(__name__)


@dataclass
class PolicyViolationResult:
    rule_id: str
    condition: str
    forbidden_action: str
    severity: str
    description: str


def _derive_severity(description: str) -> str:
    desc_lower = description.lower()
    if "high" in desc_lower:
        return "high"
    if "medium" in desc_lower:
        return "medium"
    return "low"


class PolicyEngine:
    def __init__(self, rules: list[PolicyRule]) -> None:
        self._rules = rules

    def check(self, state: dict, action: dict) -> list[PolicyViolationResult]:
        action_type = action.get("type", "")
        violations: list[PolicyViolationResult] = []
        for rule in self._rules:
            if action_type not in rule.forbidden_actions:
                continue
            try:
                triggered = eval(rule.condition, {"__builtins__": {}}, {"state": state})
            except Exception as exc:
                logger.warning("Policy rule %r condition eval failed: %s", rule.id, exc)
                continue
            if triggered:
                violations.append(
                    PolicyViolationResult(
                        rule_id=rule.id,
                        condition=rule.condition,
                        forbidden_action=action_type,
                        severity=_derive_severity(rule.description),
                        description=rule.description,
                    )
                )
        return violations
