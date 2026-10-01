"""Periodic snapshots an episode can resume from, for each environment type."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from forge.runtime.reliability.failures import DivergenceError
from forge.runtime.reliability.settings import DEFAULT_SNAPSHOT_INTERVAL

logger = logging.getLogger("forge.reliability")


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
            # The environment's own checkpoint: state, RNG, clock, counters,
            # trajectory and personas. `in_process_env.restore(data)` resumes it.
            if in_process_env is not None:
                data = in_process_env.checkpoint()

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
