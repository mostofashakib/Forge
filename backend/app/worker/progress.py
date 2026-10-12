"""Progress reporting for long-running jobs over Redis pub/sub."""
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
