"""Test Isolation Boundary.

Enforces complete, strict isolation between agent execution and test suites.
The agent must never have access to unit tests, test suites, grading rubrics,
or ground-truth assertions in order to eliminate test leakage and reward hacking.
"""
from __future__ import annotations

import copy
import re
from typing import Any

from forge.contracts.types import Task


# Sensitive keys that must NEVER be surfaced to the agent
TEST_KEYS_TO_SCRUB = frozenset({
    "tests",
    "test_suite",
    "unit_tests",
    "test_cases",
    "test_results",
    "pytest_passed",
    "tests_passed",
    "hidden_tests",
    "grading_rubric",
    "ground_truth",
    "evaluation_suite",
    "verifier_spec",
    "initial_test_suite",
})

# Patterns matching test file inspection or tampering in agent actions
DISALLOWED_TEST_PATTERNS = [
    re.compile(r"(?:cat|head|tail|less|more|nano|vim|grep|python|pytest|unittest)\s+.*test.*", re.IGNORECASE),
    re.compile(r"(?:tests/|test_.*\.py|.*_test\.py)", re.IGNORECASE),
]


class TestIsolationBoundary:
    """Enforces zero-test-leakage isolation for agent observation and action spaces."""

    @classmethod
    def sanitize_observation(cls, observation: dict[str, Any]) -> dict[str, Any]:
        """Deeply scrub any test suites, ground-truth assertions, or grading keys from agent observation."""
        if not isinstance(observation, dict):
            return observation

        sanitized: dict[str, Any] = {}
        for key, value in observation.items():
            if key.lower() in TEST_KEYS_TO_SCRUB:
                continue

            if isinstance(value, dict):
                sanitized[key] = cls.sanitize_observation(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    cls.sanitize_observation(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    @classmethod
    def sanitize_task_for_agent(cls, task: Task | dict[str, Any] | None) -> dict[str, Any]:
        """Render a task representation strictly stripped of internal test/verification criteria."""
        if task is None:
            return {}

        if isinstance(task, dict):
            task_id = task.get("id") or task.get("name", "")
            objective = task.get("objective", "")
            inputs = task.get("inputs", {})
            metadata = {k: v for k, v in task.get("metadata", {}).items() if k.lower() not in TEST_KEYS_TO_SCRUB}
        else:
            task_id = task.id
            objective = task.objective
            inputs = getattr(task, "inputs", {})
            metadata = {k: v for k, v in task.metadata.items() if k.lower() not in TEST_KEYS_TO_SCRUB}

        return {
            "id": task_id,
            "objective": objective,
            "inputs": inputs,
            "metadata": metadata,
        }

    @classmethod
    def audit_agent_isolation(cls, trajectory: Any) -> list[str]:
        """Detect whether the agent attempted to access, read, or modify forbidden test files."""
        breach_evidence: list[str] = []
        if not hasattr(trajectory, "steps"):
            return breach_evidence

        for i, step in enumerate(trajectory.steps):
            action = getattr(step, "action", {})
            if isinstance(action, dict):
                # Check action command strings (e.g. bash commands or file paths)
                cmd = str(action.get("params", {}).get("command") or action.get("command") or "")
                for pattern in DISALLOWED_TEST_PATTERNS:
                    if pattern.search(cmd):
                        breach_evidence.append(
                            f"Step {i}: Agent attempted unauthorized inspection of test suite: {cmd!r}"
                        )
                        break

        return breach_evidence
