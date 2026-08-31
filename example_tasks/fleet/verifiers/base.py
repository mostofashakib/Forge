"""Deterministic verifier building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fleet.core.models import VerificationResult

StatePredicate = Callable[[dict[str, Any]], tuple[bool, str, dict[str, Any]]]
TrajectoryPredicate = Callable[[dict[str, Any]], tuple[bool, str, dict[str, Any]]]


@dataclass(frozen=True)
class VerificationSpec:
    state_checks: list[StatePredicate]
    invariant_checks: list[TrajectoryPredicate]
    trajectory_checks: list[TrajectoryPredicate]
    negative_checks: list[TrajectoryPredicate]


class LayeredVerifier:
    """Five-layer verifier with deterministic binary scoring.

    The LLM judge layer is intentionally represented as cached external input
    and excluded from authoritative scoring unless a caller injects stable
    cached results into the trajectory before verification.
    """

    def __init__(self, spec: VerificationSpec) -> None:
        self._spec = spec

    def verify(self, final_state: dict[str, Any], trajectory: dict[str, Any]) -> list[VerificationResult]:
        results: list[VerificationResult] = []
        results.extend(self._run_state_layer("state", final_state, self._spec.state_checks))
        results.extend(self._run_trajectory_layer("invariant", trajectory, self._spec.invariant_checks))
        results.extend(self._run_trajectory_layer("trajectory", trajectory, self._spec.trajectory_checks))
        results.extend(self._run_llm_cache_layer(trajectory))
        results.extend(self._run_trajectory_layer("negative", trajectory, self._spec.negative_checks))
        return results

    def _run_state_layer(
        self,
        layer: str,
        state: dict[str, Any],
        checks: list[StatePredicate],
    ) -> list[VerificationResult]:
        return [self._result(layer, *check(state)) for check in checks]

    def _run_trajectory_layer(
        self,
        layer: str,
        trajectory: dict[str, Any],
        checks: list[TrajectoryPredicate],
    ) -> list[VerificationResult]:
        return [self._result(layer, *check(trajectory)) for check in checks]

    def _run_llm_cache_layer(self, trajectory: dict[str, Any]) -> list[VerificationResult]:
        cached = trajectory.get("cached_llm_judgments", [])
        return [
            VerificationResult(
                score=1 if bool(item.get("passed")) else 0,
                layer="llm_judge",
                passed=bool(item.get("passed")),
                message=str(item.get("message", "")),
                details=dict(item.get("details", {})),
            )
            for item in cached
        ]

    def _result(self, layer: str, passed: bool, message: str, details: dict[str, Any]) -> VerificationResult:
        return VerificationResult(score=1 if passed else 0, layer=layer, passed=passed, message=message, details=details)


def message_exists(channel_name: str, body: str, author_id: str | None = None) -> StatePredicate:
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        channel_ids = {
            channel.get("channel_id")
            for channel in state.get("channels", [])
            if channel.get("name") == channel_name
        }
        matches = [
            message
            for message in state.get("messages", [])
            if message.get("channel_id") in channel_ids
            and message.get("body") == body
            and (author_id is None or message.get("author_id") == author_id)
            and not message.get("deleted", False)
        ]
        return bool(matches), "Expected matching Slack message.", {"matches": len(matches)}

    return check


def task_has_status(title: str, status: str) -> StatePredicate:
    def check(state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        matches = [
            task
            for task in state.get("tasks", [])
            if task.get("title") == title and task.get("status") == status and not task.get("deleted", False)
        ]
        return bool(matches), "Expected task with status.", {"matches": len(matches), "title": title, "status": status}

    return check


def required_tool_called(tool_name: str) -> TrajectoryPredicate:
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        calls = [call for call in trajectory_tool_calls(trajectory) if call.get("tool_name") == tool_name]
        return bool(calls), "Expected required tool call.", {"tool_name": tool_name, "count": len(calls)}

    return check


def forbidden_tool_not_called(tool_name: str) -> TrajectoryPredicate:
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        calls = [call for call in trajectory_tool_calls(trajectory) if call.get("tool_name") == tool_name]
        return not calls, "Expected forbidden tool to be absent.", {"tool_name": tool_name, "count": len(calls)}

    return check


def trajectory_tool_calls(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    legacy = trajectory.get("extra", {}).get("legacy")
    if isinstance(legacy, dict):
        artifacts = legacy.get("artifacts", {})
        calls = artifacts.get("tool_calls", legacy.get("tool_calls", []))
        return [dict(call) for call in calls]

    calls = trajectory.get("tool_calls", [])
    if calls:
        return [dict(call) for call in calls]

    atif_calls = []
    for step in trajectory.get("steps", []):
        for call in step.get("tool_calls", []):
            normalized = dict(call)
            if "tool_name" not in normalized and "function_name" in normalized:
                normalized["tool_name"] = normalized["function_name"]
            if "input_payload" not in normalized and "arguments" in normalized:
                normalized["input_payload"] = normalized["arguments"]
            atif_calls.append(normalized)
    return atif_calls


def final_answer_equals(expected: str) -> TrajectoryPredicate:
    def check(trajectory: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        actual = str(trajectory.get("extra", {}).get("final_answer", ""))
        if not actual:
            agent_steps = [step for step in trajectory.get("steps", []) if step.get("source") == "agent"]
            actual = str(agent_steps[-1].get("message", "")) if agent_steps else ""
        cleaned = actual.strip().strip("'\"`").strip()
        return cleaned == expected, "Expected final answer.", {"expected": expected, "actual": actual}

    return check
