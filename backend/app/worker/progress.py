"""Progress reporting for long-running jobs: Redis pub/sub messages and BenchmarkRun status."""
from __future__ import annotations

import json
import logging

from forge.settings import redis_url

logger = logging.getLogger("backend.app.worker.tasks")


def connect_progress_redis():
    """A Redis client for progress messages, verified with a ping. Raises when unreachable."""
    import redis as _redis

    client = _redis.from_url(redis_url(), socket_connect_timeout=3, socket_timeout=3)
    client.ping()
    return client

def progress_publisher(client, channel: str, tag: str):
    """Publish JSON progress messages. A lost message is logged, never fatal to the run."""

    def publish(message: dict) -> None:
        if client is None:
            return
        try:
            client.publish(channel, json.dumps(message))
        except Exception:
            logger.debug("[task:%s] progress publish failed", tag, exc_info=True)

    return publish

def update_benchmark_run(
    run_id: str,
    status: str,
    error: str | None = None,
    report_json: str | None = None,
) -> None:
    from backend.app.database import get_session_factory
    from backend.app.models import BenchmarkRun
    from datetime import datetime, timezone

    try:
        SessionLocal = get_session_factory()
        with SessionLocal() as db:
            run = db.get(BenchmarkRun, run_id)
            if run:
                run.status = status
                if error is not None:
                    run.error = error
                if report_json is not None:
                    run.report_json = report_json
                if status in ("done", "failed"):
                    run.completed_at = datetime.now(timezone.utc)
                db.commit()
    except Exception as exc:
        logger.error("[task:benchmark] DB update failed for %s: %s", run_id, exc)
