"""Two episodes must never drive the same environment container at once.

Every container episode resets and steps the one container its environment
owns. Overlapping episodes interleave resets and actions, so each sees the
other's writes. `exclusive_environment` serializes them per environment.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from backend.app.worker.env_lock import exclusive_environment


def test_the_lock_is_keyed_by_environment_and_held_for_the_whole_block():
    redis_client = MagicMock()
    lock = redis_client.lock.return_value
    events: list[str] = []
    lock.acquire.side_effect = lambda *a, **k: events.append("acquire") or True
    lock.release.side_effect = lambda: events.append("release")

    with exclusive_environment(redis_client, "gmail_env", ttl_s=30):
        events.append("episode")

    assert redis_client.lock.call_args.args[0] == "forge:env-lock:gmail_env"
    assert redis_client.lock.call_args.kwargs["timeout"] == 30
    assert events == ["acquire", "episode", "release"]


def test_the_lock_is_released_when_the_episode_raises():
    redis_client = MagicMock()
    lock = redis_client.lock.return_value

    try:
        with exclusive_environment(redis_client, "gmail_env", ttl_s=30):
            raise RuntimeError("runner crashed")
    except RuntimeError:
        pass

    lock.release.assert_called_once()


def test_a_long_episode_keeps_renewing_the_lock():
    # The TTL only exists so a dead worker frees the environment. A live
    # episode that outlasts it must keep the lock, or a second one starts.
    redis_client = MagicMock()
    lock = redis_client.lock.return_value
    renewed = threading.Event()
    lock.reacquire.side_effect = lambda: renewed.set()

    with exclusive_environment(redis_client, "gmail_env", ttl_s=0.3):
        assert renewed.wait(timeout=2.0)

    renew_count = lock.reacquire.call_count
    time.sleep(0.3)
    # False-positive guard: renewal stops once the episode is over.
    assert lock.reacquire.call_count == renew_count


def test_different_environments_use_different_locks():
    redis_client = MagicMock()

    with exclusive_environment(redis_client, "env_a", ttl_s=30):
        pass
    with exclusive_environment(redis_client, "env_b", ttl_s=30):
        pass

    names = [c.args[0] for c in redis_client.lock.call_args_list]
    assert names == ["forge:env-lock:env_a", "forge:env-lock:env_b"]
