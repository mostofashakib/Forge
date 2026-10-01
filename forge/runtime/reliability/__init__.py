"""Reliability subsystem for the Forge platform.

  - failures:   classify a failure as an infrastructure crash or an agent failure
  - settings:   retry cap, short-episode threshold, snapshot interval
  - snapshots:  periodic per-environment snapshots and divergence checks
  - versioning: the exact environment an episode ran against
  - quarantine: quarantine always-crashing tasks, flag suspect versions
  - executor:   the retry loop that ties them together
"""
from forge.runtime.reliability.executor import (
    AttemptRecord,
    ReliableEpisodeExecution,
    execute_reliable_episode,
)
from forge.runtime.reliability.failures import (
    AGENT_REASONS,
    FAILURE_TYPE_AGENT,
    FAILURE_TYPE_INFRASTRUCTURE,
    INFRASTRUCTURE_REASONS,
    REASON_BROWSER_DISCONNECTED,
    REASON_CHECK_FAILED,
    REASON_CONTAINER_UNREACHABLE,
    REASON_GRADING_SANDBOX_FAILED,
    REASON_INVALID_ACTION,
    REASON_POLICY_VIOLATION,
    REASON_PROVIDER_ERROR,
    REASON_RESET_FAILED,
    REASON_STATE_READ_FAILED,
    REASON_TRANSPORT_TIMEOUT,
    AgentFailure,
    DivergenceError,
    InfrastructureCrash,
    classify_failure,
    log_infrastructure_failure,
)
from forge.runtime.reliability.quarantine import (
    record_flagged_version_if_needed,
    record_task_quarantine_if_needed,
    release_quarantined_task,
    resolve_flagged_version,
)
from forge.runtime.reliability.settings import (
    DEFAULT_RETRY_CAP,
    DEFAULT_SHORT_EPISODE_THRESHOLD,
    DEFAULT_SNAPSHOT_INTERVAL,
    ReliabilitySettings,
    get_reliability_settings,
)
from forge.runtime.reliability.snapshots import SnapshotManager, SnapshotRecord
from forge.runtime.reliability.versioning import compute_environment_version

__all__ = [
    "AGENT_REASONS", "FAILURE_TYPE_AGENT", "FAILURE_TYPE_INFRASTRUCTURE", "INFRASTRUCTURE_REASONS",
    "REASON_BROWSER_DISCONNECTED", "REASON_CHECK_FAILED", "REASON_CONTAINER_UNREACHABLE",
    "REASON_GRADING_SANDBOX_FAILED", "REASON_INVALID_ACTION", "REASON_POLICY_VIOLATION",
    "REASON_PROVIDER_ERROR", "REASON_RESET_FAILED", "REASON_STATE_READ_FAILED", "REASON_TRANSPORT_TIMEOUT",
    "DEFAULT_RETRY_CAP", "DEFAULT_SHORT_EPISODE_THRESHOLD", "DEFAULT_SNAPSHOT_INTERVAL",
    "AgentFailure", "AttemptRecord", "DivergenceError", "InfrastructureCrash", "ReliabilitySettings",
    "ReliableEpisodeExecution", "SnapshotManager", "SnapshotRecord",
    "classify_failure", "compute_environment_version", "execute_reliable_episode", "get_reliability_settings",
    "log_infrastructure_failure", "record_flagged_version_if_needed", "record_task_quarantine_if_needed",
    "release_quarantined_task", "resolve_flagged_version",
]
