"""Tool-Use Agent (TU) verification adapter.

Verifies OpenAPI, REST, and function-calling workflows: tool invocation
coverage, parameter validation, execution sequencing, and forbidden calls.
"""
from __future__ import annotations

from typing import Any

from forge.contracts.types import CheckResult, Task
from forge.contracts.verification import GroupVerificationSpec, VerifierGroup
from forge.runtime.trust_boundary.adapters.base import BaseVerifierGroupAdapter


class TUAdapter(BaseVerifierGroupAdapter):
    """Adapter verifying Tool-Use and API-calling agent trajectories."""

    @property
    def group_type(self) -> VerifierGroup:
        return VerifierGroup.TU

    def verify(
        self,
        spec: GroupVerificationSpec,
        state: dict[str, Any],
        trajectory: Any,
        task: Task | dict[str, Any] | None,
    ) -> list[CheckResult]:
        checks: list[CheckResult] = []
        rules = spec.rules
        actions = self.extract_actions(trajectory)
        called_tools = [a.get("type", "") for a in actions]

        # 1. Required Tools
        required_tools = rules.get("required_tools", []) or spec.milestones
        for tool in required_tools:
            passed = tool in called_tools
            checks.append(CheckResult(
                name=f"tu:required_tool:{tool}",
                passed=passed,
                score=1.0 if passed else 0.0,
                evidence=f"Tool '{tool}' was {'called' if passed else 'NOT called'}",
            ))

        # 2. Tool Parameter Validation
        param_rules = rules.get("param_constraints", {})
        if isinstance(param_rules, dict):
            for tool_name, constraints in param_rules.items():
                matching_calls = [a for a in actions if a.get("type") == tool_name]
                if not matching_calls:
                    checks.append(CheckResult(
                        name=f"tu:params:{tool_name}",
                        passed=False,
                        score=0.0,
                        evidence=f"Tool '{tool_name}' was never called to validate parameters",
                    ))
                    continue

                all_valid = True
                failure_reason = ""
                for call in matching_calls:
                    params = call.get("params", {})
                    for param_key, expected_val in constraints.items():
                        if param_key not in params:
                            all_valid = False
                            failure_reason = f"Missing required parameter '{param_key}'"
                            break
                        if expected_val is not None and params[param_key] != expected_val:
                            all_valid = False
                            failure_reason = f"Parameter '{param_key}' got {params[param_key]!r}, expected {expected_val!r}"
                            break
                    if not all_valid:
                        break

                checks.append(CheckResult(
                    name=f"tu:params:{tool_name}",
                    passed=all_valid,
                    score=1.0 if all_valid else 0.0,
                    evidence=failure_reason if not all_valid else f"Parameters for '{tool_name}' valid",
                ))

        # 3. Tool Ordering / Sequence
        expected_sequence = rules.get("tool_sequence", [])
        if expected_sequence:
            # Check subsequence in order
            idx = 0
            seq_matched = True
            for expected_tool in expected_sequence:
                try:
                    found_at = called_tools.index(expected_tool, idx)
                    idx = found_at + 1
                except ValueError:
                    seq_matched = False
                    break
            checks.append(CheckResult(
                name="tu:tool_sequence",
                passed=seq_matched,
                score=1.0 if seq_matched else 0.0,
                evidence=f"Sequence {expected_sequence} {'strictly maintained' if seq_matched else 'violated'}",
            ))

        # 4. Forbidden Tool Invocations
        for forbidden in spec.forbidden:
            called = forbidden in called_tools
            checks.append(CheckResult(
                name=f"tu:forbidden:{forbidden}",
                passed=not called,
                score=1.0 if not called else 0.0,
                evidence=f"Forbidden tool '{forbidden}' {'avoided' if not called else 'CALLED'}",
            ))

        # 5. Expected State After Execution
        for key, expected_val in spec.expected_state.items():
            actual = state.get(key)
            passed = actual == expected_val
            checks.append(CheckResult(
                name=f"tu:state:{key}",
                passed=passed,
                score=1.0 if passed else 0.0,
                evidence=f"State '{key}': expected {expected_val!r}, got {actual!r}",
            ))

        if not checks:
            checks.append(CheckResult(
                name="tu:basic_completion",
                passed=True,
                score=1.0,
                evidence="Tool interactions completed cleanly",
            ))

        return checks
