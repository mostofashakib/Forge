"""Base class and common helpers for Verifier Group Adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from forge.contracts.types import CheckResult, Task
from forge.contracts.verification import GroupVerificationSpec, VerifierGroup


class BaseVerifierGroupAdapter(ABC):
    """Abstract base adapter for domain-specific verification groups."""

    @property
    @abstractmethod
    def group_type(self) -> VerifierGroup:
        """The specific evaluation group this adapter handles."""

    @abstractmethod
    def verify(
        self,
        spec: GroupVerificationSpec,
        state: dict[str, Any],
        trajectory: Any,
        task: Task | dict[str, Any] | None,
    ) -> list[CheckResult]:
        """Execute domain-specific verification checks inside the trust boundary."""

    @staticmethod
    def extract_steps(trajectory: Any) -> list[Any]:
        """Extract step objects from trajectory if present."""
        if hasattr(trajectory, "steps"):
            return list(trajectory.steps)
        if isinstance(trajectory, list):
            return trajectory
        return []

    @staticmethod
    def extract_actions(trajectory: Any) -> list[dict[str, Any]]:
        """Extract action dictionaries from trajectory steps."""
        actions: list[dict[str, Any]] = []
        for step in BaseVerifierGroupAdapter.extract_steps(trajectory):
            if hasattr(step, "action"):
                action = step.action
                if isinstance(action, dict):
                    actions.append(action)
                elif hasattr(action, "to_dict"):
                    actions.append(action.to_dict())
            elif isinstance(step, dict) and "action" in step:
                actions.append(step["action"])
        return actions

    @staticmethod
    def extract_events(trajectory: Any) -> list[dict[str, Any]]:
        """Extract event dictionaries from trajectory."""
        if hasattr(trajectory, "events") and isinstance(trajectory.events, list):
            return list(trajectory.events)
        events: list[dict[str, Any]] = []
        for step in BaseVerifierGroupAdapter.extract_steps(trajectory):
            if hasattr(step, "events") and isinstance(step.events, list):
                events.extend(step.events)
            elif isinstance(step, dict) and "events" in step:
                events.extend(step["events"])
        return events
