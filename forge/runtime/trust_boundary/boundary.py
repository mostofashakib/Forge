"""Verifier Trust Boundary.

Acts as an isolated security gate between untrusted agent execution trajectories
and authoritative task evaluation. Replaces one-off per-task verifiers with
standardized domain group adapters (CUA, TU, Code, Data) while enforcing
anti-tampering checks on trajectory integrity.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from forge.contracts.reward import Verifier
from forge.contracts.types import CheckResult, Task, VerificationResult
from forge.contracts.verification import (
    GroupVerificationSpec,
    TrustBoundaryReport,
    TrustLevel,
    VerifierGroup,
    VerifierGroupAdapter,
)
from forge.runtime.trust_boundary.adapters.code import CodeAdapter
from forge.runtime.trust_boundary.adapters.cua import CUAAdapter
from forge.runtime.trust_boundary.adapters.data import DataAdapter
from forge.runtime.trust_boundary.adapters.tu import TUAdapter

logger = logging.getLogger(__name__)


class VerifierTrustBoundary(Verifier):
    """Isolated Trust Boundary and evaluation router for agent rollouts."""

    def __init__(self, verifier_id: str = "trust_boundary") -> None:
        self.verifier_id = verifier_id
        self._adapters: dict[VerifierGroup, VerifierGroupAdapter] = {}

        # Register standard default domain adapters
        self.register_adapter(VerifierGroup.CUA, CUAAdapter())
        self.register_adapter(VerifierGroup.TU, TUAdapter())
        self.register_adapter(VerifierGroup.CODE, CodeAdapter())
        self.register_adapter(VerifierGroup.DATA, DataAdapter())

    def register_adapter(self, group: VerifierGroup | str, adapter: VerifierGroupAdapter) -> None:
        """Register a domain adapter for a verifier group."""
        grp = VerifierGroup(group) if isinstance(group, str) else group
        self._adapters[grp] = adapter

    def get_adapter(self, group: VerifierGroup | str) -> VerifierGroupAdapter | None:
        """Retrieve adapter for a specific group."""
        grp = VerifierGroup(group) if isinstance(group, str) else group
        return self._adapters.get(grp)

    # ------------------------------------------------------------------
    # Anti-Tampering & Integrity Analysis
    # ------------------------------------------------------------------

    def audit_trajectory_integrity(self, trajectory: Any) -> list[str]:
        """Detect trajectory tampering, step index regression, or state hash drift."""
        tamper_flags: list[str] = []
        if not hasattr(trajectory, "steps"):
            return tamper_flags

        steps = list(trajectory.steps)
        if not steps:
            return tamper_flags

        # 1. Step Index Continuity
        for i, step in enumerate(steps):
            step_idx = getattr(step, "step_index", i)
            if step_idx != i:
                tamper_flags.append(
                    f"Non-monotonic step sequence: expected index {i}, found {step_idx}"
                )

        # 2. State Hash Chain Continuity
        for i in range(len(steps) - 1):
            after_hash = getattr(steps[i], "state_hash_after", None)
            before_next_hash = getattr(steps[i + 1], "state_hash_before", None)
            if after_hash and before_next_hash and after_hash != before_next_hash:
                tamper_flags.append(
                    f"State hash drift between step {i} and {i + 1}: '{after_hash}' != '{before_next_hash}'"
                )

        # 3. Policy Violations
        has_violation = getattr(trajectory, "has_policy_violation", False)
        if has_violation:
            tamper_flags.append("Agent triggered one or more policy violations during execution")

        return tamper_flags

    # ------------------------------------------------------------------
    # Spec Resolution
    # ------------------------------------------------------------------

    def resolve_group_spec(self, task: Task | dict[str, Any] | None) -> GroupVerificationSpec:
        """Extract or infer domain group and verification rules from task metadata."""
        if task is None:
            return GroupVerificationSpec(group=VerifierGroup.TU)

        metadata: dict[str, Any] = {}
        if isinstance(task, dict):
            metadata = task.get("metadata", {})
            task_name = task.get("name", "")
            task_obj = task.get("objective", "")
        else:
            metadata = task.metadata if hasattr(task, "metadata") else {}
            task_name = getattr(task, "id", "")
            task_obj = getattr(task, "objective", "")

        # Direct explicit spec provided
        if "group_spec" in metadata and isinstance(metadata["group_spec"], dict):
            return GroupVerificationSpec(**metadata["group_spec"])
        if "verifier_spec" in metadata and isinstance(metadata["verifier_spec"], dict):
            return GroupVerificationSpec(**metadata["verifier_spec"])

        # Determine target group
        raw_group = metadata.get("verifier_group") or metadata.get("group")
        group = VerifierGroup.TU
        if raw_group:
            try:
                group = VerifierGroup(raw_group.lower())
            except ValueError:
                group = VerifierGroup.CUSTOM
        else:
            # Heuristic inference from environment type or objectives
            env_type = metadata.get("env_type", "").lower()
            if "browser" in env_type or "cua" in env_type or "ui" in env_type:
                group = VerifierGroup.CUA
            elif "cli" in env_type or "code" in env_type or "terminal" in env_type:
                group = VerifierGroup.CODE
            elif "data" in env_type or "db" in env_type or "sql" in env_type:
                group = VerifierGroup.DATA

        # Build rules from success_conditions if available
        success_conds: list[dict[str, Any]] = []
        if isinstance(task, dict):
            success_conds = task.get("success_conditions", [])
        elif hasattr(task, "success_conditions"):
            success_conds = list(task.success_conditions)

        rules: dict[str, Any] = metadata.get("rules", {})
        expected_state: dict[str, Any] = {}
        for sc in success_conds:
            if isinstance(sc, dict) and "key" in sc and "value" in sc:
                expected_state[sc["key"]] = sc["value"]

        milestones: list[str] = metadata.get("milestones", [])
        forbidden: list[str] = metadata.get("forbidden", [])

        return GroupVerificationSpec(
            group=group,
            rules=rules,
            milestones=milestones,
            forbidden=forbidden,
            expected_state=expected_state,
            metadata={"task_name": task_name, "objective": task_obj},
        )

    # ------------------------------------------------------------------
    # Evaluation Boundary Execution
    # ------------------------------------------------------------------

    def evaluate(
        self,
        state: dict[str, Any],
        trajectory: Any,
        task: Task | dict[str, Any] | None,
        spec: GroupVerificationSpec | None = None,
    ) -> TrustBoundaryReport:
        """Run verification inside the trust boundary and produce an auditable report."""
        # 1. Defensive deepcopy of untrusted state to prevent verifier side-effects
        safe_state = copy.deepcopy(state)

        # 2. Audit Trajectory Integrity
        tamper_flags = self.audit_trajectory_integrity(trajectory)

        # 3. Resolve Group Spec
        active_spec = spec or self.resolve_group_spec(task)

        # 4. Dispatch to Group Adapter
        adapter = self.get_adapter(active_spec.group)
        if adapter is None:
            # Fallback to TU adapter
            adapter = self._adapters.get(VerifierGroup.TU, TUAdapter())

        checks = adapter.verify(active_spec, safe_state, trajectory, task)

        # 5. Synthesize Verdict & Trust Level
        if tamper_flags:
            # Integrity breach detected
            has_policy_violation = any("policy violation" in f for f in tamper_flags)
            trust_level = TrustLevel.REJECTED if has_policy_violation else TrustLevel.TAMPERED
            passed = False
            score = 0.0
            checks.append(CheckResult(
                name="trust_boundary:anti_tampering",
                passed=False,
                score=0.0,
                evidence="; ".join(tamper_flags),
            ))
        else:
            all_passed = all(c.passed for c in checks) if checks else False
            score = sum(c.score for c in checks) / len(checks) if checks else 0.0
            passed = all_passed
            trust_level = TrustLevel.VERIFIED if passed else TrustLevel.UNVERIFIED

        return TrustBoundaryReport(
            trust_level=trust_level,
            passed=passed,
            score=score,
            checks=checks,
            tamper_evidence=tamper_flags,
            group=active_spec.group,
            metadata=active_spec.metadata,
        )

    def verify(
        self,
        state: dict[str, Any],
        trajectory: Any,
        task: Task | dict[str, Any] | None,
    ) -> VerificationResult:
        """Standard Verifier interface implementation."""
        report = self.evaluate(state, trajectory, task)
        return report.to_verification_result(self.verifier_id)

    def __call__(
        self,
        state: dict[str, Any],
        trajectory: Any,
        task: Task | dict[str, Any] | None,
    ) -> VerificationResult:
        """Callable convenience for function-like verifier invocations."""
        return self.verify(state, trajectory, task)
