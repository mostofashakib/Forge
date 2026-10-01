"""Classify every failed episode as an infrastructure crash or an agent failure."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("forge.reliability")


# ---------------------------------------------------------------------------
# Failure Types & Reasons
# ---------------------------------------------------------------------------

FAILURE_TYPE_INFRASTRUCTURE = "infrastructure"
FAILURE_TYPE_AGENT = "agent"

# Specific Infrastructure Failure Reasons
REASON_CONTAINER_UNREACHABLE = "container_unreachable"
REASON_RESET_FAILED = "reset_failed"
REASON_STATE_READ_FAILED = "state_read_failed"
REASON_TRANSPORT_TIMEOUT = "transport_timeout"
REASON_BROWSER_DISCONNECTED = "browser_disconnected"
REASON_GRADING_SANDBOX_FAILED = "grading_sandbox_failed"
REASON_PROVIDER_ERROR = "provider_error"

INFRASTRUCTURE_REASONS = frozenset({
    REASON_CONTAINER_UNREACHABLE,
    REASON_RESET_FAILED,
    REASON_STATE_READ_FAILED,
    REASON_TRANSPORT_TIMEOUT,
    REASON_BROWSER_DISCONNECTED,
    REASON_GRADING_SANDBOX_FAILED,
    REASON_PROVIDER_ERROR,
})

# Specific Agent Failure Reasons
REASON_INVALID_ACTION = "invalid_action"
REASON_POLICY_VIOLATION = "policy_violation"
REASON_CHECK_FAILED = "check_failed"

AGENT_REASONS = frozenset({
    REASON_INVALID_ACTION,
    REASON_POLICY_VIOLATION,
    REASON_CHECK_FAILED,
})



class InfrastructureCrash(RuntimeError):
    """An infrastructure failure that should not count against the agent."""

    def __init__(
        self,
        reason: str,
        detail: str = "",
        original_exc: Exception | None = None,
        step: int = 0,
    ) -> None:
        super().__init__(f"Infrastructure crash [{reason}]: {detail}")
        self.reason = reason
        self.detail = detail
        self.original_exc = original_exc
        self.step = step


class AgentFailure(RuntimeError):
    """An agent failure (invalid action, policy violation, check failed)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"Agent failure [{reason}]: {detail}")
        self.reason = reason
        self.detail = detail


class DivergenceError(InfrastructureCrash):
    """Raised when replaying steps reaches a different state hash from original."""

    def __init__(self, step: int, expected_hash: str, actual_hash: str) -> None:
        super().__init__(
            reason="diverged",
            detail=f"Replay diverged at step {step}: expected {expected_hash}, got {actual_hash}",
            step=step,
        )
        self.step = step
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


def classify_failure(exc_or_reason: Any) -> tuple[str, str]:
    """Classify every failed episode as either an agent failure or an infrastructure failure.

    Returns:
        (failure_type, specific_reason)
    """
    if isinstance(exc_or_reason, InfrastructureCrash):
        return FAILURE_TYPE_INFRASTRUCTURE, exc_or_reason.reason
    if isinstance(exc_or_reason, AgentFailure):
        return FAILURE_TYPE_AGENT, exc_or_reason.reason

    text = str(exc_or_reason or "").lower()

    # Known infrastructure matches
    if any(k in text for k in ("container_unreachable", "econnrefused", "connection refused", "not become healthy", "no running container")):
        return FAILURE_TYPE_INFRASTRUCTURE, REASON_CONTAINER_UNREACHABLE
    if any(k in text for k in ("reset_failed", "reset failed", "/forge/reset")):
        return FAILURE_TYPE_INFRASTRUCTURE, REASON_RESET_FAILED
    if any(k in text for k in ("state_read_failed", "get_state failed", "/forge/state failed", "failed state read")):
        return FAILURE_TYPE_INFRASTRUCTURE, REASON_STATE_READ_FAILED
    if any(k in text for k in ("transport_timeout", "timed out", "timeout", "transport error", "readtimeout", "connecttimeout")):
        return FAILURE_TYPE_INFRASTRUCTURE, REASON_TRANSPORT_TIMEOUT
    if any(k in text for k in ("browser_disconnected", "devtools", "cdp", "browser connection", "playwright")):
        return FAILURE_TYPE_INFRASTRUCTURE, REASON_BROWSER_DISCONNECTED
    if any(k in text for k in ("grading_sandbox_failed", "grading sandbox", "sandbox will not start")):
        return FAILURE_TYPE_INFRASTRUCTURE, REASON_GRADING_SANDBOX_FAILED
    if any(k in text for k in ("provider_error", "ratelimit", "rate limit", "503", "429", "apierror", "overloaded", "quota")):
        return FAILURE_TYPE_INFRASTRUCTURE, REASON_PROVIDER_ERROR

    # Known agent matches
    if any(k in text for k in ("invalid_action", "unknown endpoint", "bad action", "schema mismatch")):
        return FAILURE_TYPE_AGENT, REASON_INVALID_ACTION
    if any(k in text for k in ("policy_violation", "forbidden", "policy violation", "policy check")):
        return FAILURE_TYPE_AGENT, REASON_POLICY_VIOLATION
    if any(k in text for k in ("check_failed", "verifier failed", "objective score", "dead_end", "diverged")):
        return FAILURE_TYPE_AGENT, REASON_CHECK_FAILED

    # Check for httpx / docker / system network exceptions
    if any(k in text for k in ("http", "connection", "socket", "network", "errno")):
        return FAILURE_TYPE_INFRASTRUCTURE, REASON_TRANSPORT_TIMEOUT

    # Default to agent failure for unknown execution issues
    return FAILURE_TYPE_AGENT, REASON_CHECK_FAILED


def log_infrastructure_failure(
    log: logging.Logger | None = None,
    *,
    episode_id: str,
    attempt: int,
    step: int,
    environment_version: str,
    error: Any,
) -> None:
    """Surface every infrastructure failure in developer logs with full context."""
    target_logger = log or logger
    target_logger.error(
        "[INFRASTRUCTURE FAILURE] episode=%s attempt=%d step=%d env_version=%s error=%s",
        episode_id,
        attempt,
        step,
        environment_version,
        str(error),
        exc_info=isinstance(error, BaseException),
    )
