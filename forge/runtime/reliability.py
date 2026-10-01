"""Reliability subsystem for the Forge platform.

Implements:
1. Classification of failures into infrastructure crashes vs agent failures.
2. Retry caps, short-episode restart, snapshotting and deterministic replay.
3. State hash divergence detection and environment version flagging.
4. Task quarantine and environment version tracking.
5. Developer logging for all infrastructure failures.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.paths import confined_path
from forge.settings import generated_envs_root

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


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Developer Logging
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Environment Versioning
# ---------------------------------------------------------------------------

def compute_environment_version(
    env_name: str,
    env_type: str = "general",
    container_id: str | None = None,
    image_id: str | None = None,
    env_dir: Path | str | None = None,
) -> str:
    """Return Docker image ID for container environments, or hash of package for in-process ones."""
    if image_id:
        return f"docker:{image_id}"

    if env_type in ("general", "premade", "cli", "browser") and container_id:
        try:
            import docker
            client = docker.from_env()
            try:
                c = client.containers.get(container_id)
                return f"docker:{c.image.id}"
            finally:
                client.close()
        except Exception:
            pass

    target_dir = Path(env_dir) if env_dir is not None else (generated_envs_root() / env_name)
    if target_dir.exists():
        hasher = hashlib.sha256()
        try:
            for path in sorted(target_dir.rglob("*")):
                if path.is_file() and not path.name.startswith("."):
                    rel = path.relative_to(target_dir).as_posix()
                    hasher.update(rel.encode("utf-8"))
                    hasher.update(path.read_bytes())
            return f"package:{hasher.hexdigest()[:16]}"
        except Exception as exc:
            logger.debug("Failed computing package hash for %s: %s", env_name, exc)

    if env_type in ("general", "premade", "cli", "browser"):
        return f"docker:env_{env_name}:{env_type}"

    return f"package:hash_{hashlib.sha256(env_name.encode()).hexdigest()[:12]}"


# ---------------------------------------------------------------------------
# Settings & Configuration
# ---------------------------------------------------------------------------

DEFAULT_RETRY_CAP = 3
DEFAULT_SHORT_EPISODE_THRESHOLD = 5
DEFAULT_SNAPSHOT_INTERVAL = 5


@dataclass(frozen=True)
class ReliabilitySettings:
    retry_cap: int = DEFAULT_RETRY_CAP
    short_episode_threshold: int = DEFAULT_SHORT_EPISODE_THRESHOLD
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL


def get_reliability_settings() -> ReliabilitySettings:
    """Read reliability settings from environment / saved config."""
    try:
        retry_cap = int(os.environ.get("FORGE_RETRY_CAP", DEFAULT_RETRY_CAP))
    except ValueError:
        retry_cap = DEFAULT_RETRY_CAP

    try:
        short_threshold = int(
            os.environ.get("FORGE_SHORT_EPISODE_THRESHOLD", DEFAULT_SHORT_EPISODE_THRESHOLD)
        )
    except ValueError:
        short_threshold = DEFAULT_SHORT_EPISODE_THRESHOLD

    try:
        snapshot_interval = int(
            os.environ.get("FORGE_SNAPSHOT_INTERVAL", DEFAULT_SNAPSHOT_INTERVAL)
        )
    except ValueError:
        snapshot_interval = DEFAULT_SNAPSHOT_INTERVAL

    return ReliabilitySettings(
        retry_cap=max(0, retry_cap),
        short_episode_threshold=max(1, short_threshold),
        snapshot_interval=max(1, snapshot_interval),
    )


# ---------------------------------------------------------------------------
# Snapshots & Checkpoints
# ---------------------------------------------------------------------------

@dataclass
class SnapshotRecord:
    step: int
    data: Any
    state_hash: str
    env_type: str


class SnapshotManager:
    """Manages periodic snapshots across different environment types."""

    def __init__(
        self,
        env_type: str = "general",
        interval: int = DEFAULT_SNAPSHOT_INTERVAL,
        snapshot_interval: int | None = None,
    ) -> None:
        self.env_type = env_type
        self.interval = max(1, snapshot_interval if snapshot_interval is not None else interval)
        self.snapshots: dict[int, SnapshotRecord] = {}
        self.model_outputs: dict[int, Any] = {}

    def record_snapshot(self, step: int, data: Any, state_hash: str = "") -> SnapshotRecord:
        record = SnapshotRecord(step=step, data=data, state_hash=state_hash, env_type=self.env_type)
        self.snapshots[step] = record
        return record

    def record_model_output(self, step: int, output: Any) -> None:
        self.model_outputs[step] = output

    def get_snapshot(self, step: int) -> Any:
        record = self.snapshots.get(step)
        return record.data if record is not None else None

    def latest_snapshot_step(self) -> int:
        if not self.snapshots:
            return 0
        return max(self.snapshots.keys())

    def verify_divergence(self, step: int, expected_hash: str, actual_hash: str) -> None:
        if expected_hash and actual_hash and expected_hash != actual_hash:
            raise DivergenceError(step, expected_hash, actual_hash)

    def should_snapshot(self, step: int) -> bool:
        if self.env_type == "browser":
            # Browser environments only have the reset at step 0
            return step == 0 and 0 not in self.snapshots
        return step % self.interval == 0

    def capture_snapshot(
        self,
        step: int,
        state_hash: str,
        *,
        http_client: Any = None,
        base_url: str | None = None,
        in_process_env: Any = None,
        cli_container_id: str | None = None,
    ) -> SnapshotRecord:
        """Create a snapshot of state for recovery."""
        data: Any = None
        if self.env_type in ("general", "premade"):
            # Full app state for HTTP apps: trigger /forge/snapshot or GET /forge/dump
            slot = f"snap_step_{step}"
            if http_client is not None and base_url is not None:
                try:
                    resp = http_client.post(f"{base_url.rstrip('/')}/forge/snapshot", json={"slot": slot}, timeout=10.0)
                    if resp.is_success:
                        data = {"slot": slot}
                except Exception as exc:
                    logger.debug("Failed saving HTTP snapshot slot %s: %s", slot, exc)
            if data is None and in_process_env is not None and hasattr(in_process_env, "state"):
                data = in_process_env.state.get()

        elif self.env_type == "cli":
            # Docker commit for CLI
            if cli_container_id:
                try:
                    import docker
                    dc = docker.from_env()
                    try:
                        c = dc.containers.get(cli_container_id)
                        tag = f"forge_cli_snap_{cli_container_id[:8]}_{step}"
                        c.commit(repository=tag)
                        data = {"image_tag": tag}
                    finally:
                        dc.close()
                except Exception as exc:
                    logger.debug("Docker commit for CLI step %d failed: %s", step, exc)

        elif self.env_type == "in_process":
            # Checkpoint of state, RNG, clock and personas for in-process environments
            if in_process_env is not None:
                import copy
                state = in_process_env.state.get() if hasattr(in_process_env, "state") else {}
                rng_state = getattr(in_process_env, "_rng", None)
                data = {
                    "state": copy.deepcopy(state),
                    "personas": getattr(in_process_env, "personas", None),
                }

        elif self.env_type == "browser":
            # Only step 0
            data = {"step": 0}

        record = SnapshotRecord(step=step, data=data, state_hash=state_hash, env_type=self.env_type)
        self.snapshots[step] = record
        return record

    def latest_snapshot_before(self, step: int) -> SnapshotRecord | None:
        valid_steps = [s for s in sorted(self.snapshots.keys()) if s <= step]
        if not valid_steps:
            return None
        return self.snapshots[valid_steps[-1]]


# ---------------------------------------------------------------------------
# Attempt Tracking
# ---------------------------------------------------------------------------

@dataclass
class AttemptRecord:
    attempt: int
    mode: str  # "fresh" or "replay"
    resume_step: int
    replayed_steps: int
    error: str | None = None
    diverged: bool = False

    def to_dict(self) -> dict:
        return {
            "attempt": self.attempt,
            "mode": self.mode,
            "resume_step": self.resume_step,
            "replayed_steps": self.replayed_steps,
            "error": self.error,
            "diverged": self.diverged,
        }


# ---------------------------------------------------------------------------
# Quarantine and Flagging Helpers
# ---------------------------------------------------------------------------

def record_task_quarantine_if_needed(
    db_session: Any,
    env_name: str,
    task_id: str,
    environment_version: str,
    reason: str,
) -> None:
    """Quarantine task when it crashes on every allowed attempt."""
    from backend.app.models import QuarantinedTask

    try:
        existing = (
            db_session.query(QuarantinedTask)
            .filter(
                QuarantinedTask.env_name == env_name,
                QuarantinedTask.task_id == task_id,
                QuarantinedTask.released == False,
            )
            .first()
        )
        if existing is None:
            db_session.add(
                QuarantinedTask(
                    task_id=task_id,
                    env_name=env_name,
                    environment_version=environment_version,
                    reason=reason,
                    released=False,
                )
            )
            db_session.commit()
            logger.warning("[reliability] Quarantined task %s in env %s: %s", task_id, env_name, reason)
    except Exception as exc:
        logger.error("[reliability] Failed to record task quarantine: %s", exc)


def record_flagged_version_if_needed(
    db_session: Any,
    env_name: str,
    version: str,
    reason: str,
) -> None:
    """Flag an environment version for investigation."""
    from backend.app.models import FlaggedEnvironmentVersion

    try:
        existing = (
            db_session.query(FlaggedEnvironmentVersion)
            .filter(
                FlaggedEnvironmentVersion.env_name == env_name,
                FlaggedEnvironmentVersion.version == version,
                FlaggedEnvironmentVersion.status == "investigating",
            )
            .first()
        )
        if existing is None:
            db_session.add(
                FlaggedEnvironmentVersion(
                    env_name=env_name,
                    version=version,
                    reason=reason,
                    status="investigating",
                )
            )
            db_session.commit()
            logger.warning("[reliability] Flagged environment %s version %s: %s", env_name, version, reason)
    except Exception as exc:
        logger.error("[reliability] Failed to flag environment version: %s", exc)


def release_quarantined_task(db_session: Any, quarantine_id: int) -> bool:
    """Release a quarantined task."""
    from backend.app.models import QuarantinedTask

    task = db_session.get(QuarantinedTask, quarantine_id)
    if task is not None and not task.released:
        task.released = True
        task.released_at = datetime.now(timezone.utc)
        db_session.commit()
        return True
    return False


def resolve_flagged_version(db_session: Any, version_id: int) -> bool:
    """Mark a flagged environment version as resolved."""
    from backend.app.models import FlaggedEnvironmentVersion

    flagged = db_session.get(FlaggedEnvironmentVersion, version_id)
    if flagged is not None and flagged.status != "resolved":
        flagged.status = "resolved"
        flagged.resolved_at = datetime.now(timezone.utc)
        db_session.commit()
        return True
    return False


class ReliableEpisodeExecution(tuple):
    """Result tuple (result, attempts) that also delegates attribute access to result."""

    def __new__(cls, result: Any, attempts: list[AttemptRecord]):
        return super().__new__(cls, (result, attempts))

    @property
    def result(self) -> Any:
        return self[0]

    @property
    def attempts(self) -> list[AttemptRecord]:
        return self[1]

    def __getattr__(self, name: str) -> Any:
        return getattr(self[0], name)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str) and hasattr(self[0], "__getitem__"):
            return self[0][item]
        return super().__getitem__(item)


# ---------------------------------------------------------------------------
# Reliable Episode Execution Loop
# ---------------------------------------------------------------------------

def execute_reliable_episode(
    *,
    episode_id: str | None = None,
    env_name: str,
    task_name: str | None = None,
    task_id: str | None = None,
    env_type: str = "general",
    seed: int = 0,
    run_fn: Any = None,
    task_runner: Any = None,
    environment_version: str | None = None,
    db_session: Any = None,
    db_session_factory: Any = None,
    episode_model: Any = None,
    container_id: str | None = None,
    image_id: str | None = None,
    log: logging.Logger | None = None,
    logger: logging.Logger | None = None,
) -> ReliableEpisodeExecution:
    """Run an episode with capped retries, snapshot restoration, replay, and quarantine."""
    import secrets

    fn = run_fn or task_runner
    if fn is None:
        raise ValueError("run_fn or task_runner must be provided")

    resolved_task_name = task_name or task_id or "default_task"
    resolved_episode_id = episode_id or f"ep_{seed:08x}_{secrets.token_hex(4)}"
    target_log = log or logger or logging.getLogger("forge.reliability")

    settings = get_reliability_settings()
    env_ver = environment_version or compute_environment_version(env_name, env_type, container_id, image_id)
    attempts_history: list[AttemptRecord] = []
    recorded_model_outputs: list[dict] = []
    last_crash_step = 0

    def _get_db():
        if db_session is not None:
            from contextlib import nullcontext
            return nullcontext(db_session)
        if db_session_factory is not None:
            return db_session_factory()
        return None

    for attempt in range(settings.retry_cap + 1):
        if attempt == 0:
            mode = "fresh"
            resume_step = 0
            replay_steps = None
        else:
            if last_crash_step < settings.short_episode_threshold:
                mode = "fresh"
                resume_step = 0
                replay_steps = None
                recorded_model_outputs.clear()
            else:
                mode = "replay"
                resume_step = (last_crash_step // settings.snapshot_interval) * settings.snapshot_interval
                replay_steps = recorded_model_outputs[:last_crash_step]

        try:
            try:
                import inspect
                sig = inspect.signature(fn)
                if len(sig.parameters) == 2:
                    result = fn(attempt, seed)
                elif "attempt" in sig.parameters and "mode" in sig.parameters:
                    result = fn(
                        attempt=attempt,
                        mode=mode,
                        replay_steps=replay_steps,
                        resume_from_step=resume_step,
                        model_outputs=recorded_model_outputs,
                        env_version=env_ver,
                    )
                else:
                    try:
                        result = fn(attempt, seed)
                    except TypeError:
                        result = fn()
            except TypeError:
                try:
                    result = fn(attempt, seed)
                except TypeError:
                    result = fn()

            # Check for budget truncation
            is_trunc = getattr(result, "is_truncated", False) or getattr(
                result, "termination_reason", ""
            ) in {"max_steps", "max_tokens", "max_wall_clock_time", "max_cost"}

            final_status = "truncated" if is_trunc else "completed"

            attempts_history.append(
                AttemptRecord(
                    attempt=attempt,
                    mode=mode,
                    resume_step=resume_step,
                    replayed_steps=len(replay_steps) if replay_steps else 0,
                    error=None,
                )
            )

            db_ctx = _get_db()
            if db_ctx is not None and episode_model is not None:
                try:
                    with db_ctx as db:
                        ep = db.get(episode_model, resolved_episode_id)
                        if ep is not None:
                            ep.status = final_status
                            ep.failure_type = None
                            ep.failure_reason = None
                            ep.environment_version = env_ver
                            ep.attempts_json = json.dumps([a.to_dict() for a in attempts_history])
                            ep.completed_at = datetime.now(timezone.utc)
                            db.commit()
                except Exception as db_exc:
                    target_log.debug("Failed updating DB for successful episode: %s", db_exc)

            if hasattr(result, "environment_version"):
                result.environment_version = env_ver
            if hasattr(result, "attempts"):
                result.attempts = [a.to_dict() for a in attempts_history]

            return ReliableEpisodeExecution(result, attempts_history)

        except DivergenceError as exc:
            last_crash_step = exc.step
            log_infrastructure_failure(
                target_log,
                episode_id=resolved_episode_id,
                attempt=attempt,
                step=exc.step,
                environment_version=env_ver,
                error=exc,
            )
            attempts_history.append(
                AttemptRecord(
                    attempt=attempt,
                    mode=mode,
                    resume_step=resume_step,
                    replayed_steps=exc.step,
                    error=str(exc),
                    diverged=True,
                )
            )
            db_ctx = _get_db()
            if db_ctx is not None:
                try:
                    with db_ctx as db:
                        record_flagged_version_if_needed(
                            db, env_name, env_ver, f"Nondeterministic replay divergence at step {exc.step}"
                        )
                except Exception as flag_exc:
                    target_log.debug("Failed flagging version for divergence: %s", flag_exc)

            if attempt >= settings.retry_cap:
                if db_ctx is not None:
                    try:
                        with db_ctx as db:
                            record_task_quarantine_if_needed(
                                db, env_name, resolved_task_name, env_ver, f"Divergence crash: {exc}"
                            )
                            if episode_model is not None:
                                ep = db.get(episode_model, resolved_episode_id)
                                if ep is not None:
                                    ep.status = "failed"
                                    ep.failure_type = FAILURE_TYPE_INFRASTRUCTURE
                                    ep.failure_reason = "diverged"
                                    ep.environment_version = env_ver
                                    ep.attempts_json = json.dumps([a.to_dict() for a in attempts_history])
                                    ep.completed_at = datetime.now(timezone.utc)
                                    db.commit()
                    except Exception as q_exc:
                        target_log.debug("Failed recording quarantine on divergence: %s", q_exc)
                raise

        except Exception as exc:
            failure_type, failure_reason = classify_failure(exc)
            step = getattr(exc, "step", last_crash_step)
            last_crash_step = step

            if failure_type == FAILURE_TYPE_AGENT:
                attempts_history.append(
                    AttemptRecord(
                        attempt=attempt,
                        mode=mode,
                        resume_step=resume_step,
                        replayed_steps=0,
                        error=str(exc),
                    )
                )
                db_ctx = _get_db()
                if db_ctx is not None and episode_model is not None:
                    try:
                        with db_ctx as db:
                            ep = db.get(episode_model, resolved_episode_id)
                            if ep is not None:
                                ep.status = "failed"
                                ep.failure_type = FAILURE_TYPE_AGENT
                                ep.failure_reason = failure_reason
                                ep.environment_version = env_ver
                                ep.attempts_json = json.dumps([a.to_dict() for a in attempts_history])
                                ep.completed_at = datetime.now(timezone.utc)
                                db.commit()
                    except Exception as db_exc:
                        target_log.debug("Failed updating DB for agent failure: %s", db_exc)
                raise

            # Infrastructure failure
            log_infrastructure_failure(
                target_log,
                episode_id=resolved_episode_id,
                attempt=attempt,
                step=step,
                environment_version=env_ver,
                error=exc,
            )
            attempts_history.append(
                AttemptRecord(
                    attempt=attempt,
                    mode=mode,
                    resume_step=resume_step,
                    replayed_steps=len(replay_steps) if replay_steps else 0,
                    error=str(exc),
                )
            )

            if attempt >= settings.retry_cap:
                db_ctx = _get_db()
                if db_ctx is not None:
                    try:
                        with db_ctx as db:
                            record_task_quarantine_if_needed(
                                db, env_name, resolved_task_name, env_ver, f"Crashed on all {attempt + 1} attempts: {failure_reason} ({exc})"
                            )
                            record_flagged_version_if_needed(
                                db, env_name, env_ver, f"Task {resolved_task_name} crashed on all attempts: {failure_reason}"
                            )
                            if episode_model is not None:
                                ep = db.get(episode_model, resolved_episode_id)
                                if ep is not None:
                                    ep.status = "failed"
                                    ep.failure_type = FAILURE_TYPE_INFRASTRUCTURE
                                    ep.failure_reason = failure_reason
                                    ep.environment_version = env_ver
                                    ep.attempts_json = json.dumps([a.to_dict() for a in attempts_history])
                                    ep.completed_at = datetime.now(timezone.utc)
                                    db.commit()
                    except Exception as db_exc:
                        target_log.debug("Failed updating DB for infra failure: %s", db_exc)
                raise
