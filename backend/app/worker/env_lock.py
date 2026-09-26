"""Per-environment mutual exclusion for container episodes.

A container environment is one live container. Every episode resets it and
then steps it, so two episodes on the same environment would interleave
their resets and actions. Workers run several tasks at once, so episodes on
one environment take a Redis lock for the whole episode.
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

import redis

logger = logging.getLogger(__name__)

# Short, so a worker that dies mid-episode frees its environment in minutes.
# A live episode renews it every third of this, however long it runs.
DEFAULT_LOCK_TTL_S = 300.0


@contextmanager
def exclusive_environment(
    redis_client: redis.Redis, env_name: str, *, ttl_s: float = DEFAULT_LOCK_TTL_S
) -> Iterator[None]:
    """Block until no other episode holds `env_name`, then hold it for the block."""
    lock = redis_client.lock(f"forge:env-lock:{env_name}", timeout=ttl_s)
    lock.acquire()
    stop = threading.Event()

    def renew() -> None:
        while not stop.wait(ttl_s / 3):
            lock.reacquire()

    renewer = threading.Thread(target=renew, name=f"env-lock-{env_name}", daemon=True)
    renewer.start()
    try:
        yield
    finally:
        stop.set()
        renewer.join()
        try:
            lock.release()
        except redis.exceptions.LockError:
            # Only possible if renewal failed and the TTL lapsed, which means
            # another episode may already have overlapped this one.
            logger.warning("[env-lock] lock for %s expired before release", env_name)
