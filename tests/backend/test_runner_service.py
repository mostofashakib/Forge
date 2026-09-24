from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from backend.app.services import runner_service as rs

STEP_SECONDS = 0.05
EPISODE_STEPS = 5


class _BlockingEnv:
    """Minimal ForgeEnv stand-in whose step() blocks like real env work does."""

    action_types = frozenset({"noop"})

    def __init__(self, steps: int = EPISODE_STEPS) -> None:
        self._limit = steps
        self.steps_taken = 0
        self.thread_ids: set[int] = set()

    def reset(self, seed=None):
        return {}, {}

    def step(self, action):
        self.thread_ids.add(threading.get_ident())
        time.sleep(STEP_SECONDS)
        self.steps_taken += 1
        done = self.steps_taken >= self._limit
        return {}, (1.5 if done else 0.0), done, False, {}

    def finalize_episode(self, reason="external"):
        return SimpleNamespace(total_reward=1.5)


class _Policy:
    def __init__(self, *_args, **_kwargs):
        pass

    def act(self, _obs):
        return {"type": "noop"}


@pytest.fixture
def blocking_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "envs"))
    from backend.app import database
    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_SessionLocal", None)
    database.init_db()
    env = _BlockingEnv()
    monkeypatch.setattr(rs, "load_forge_env", lambda _name, _telemetry: env)
    monkeypatch.setattr(rs, "RandomPolicy", _Policy)
    return env


async def _drain(queue: asyncio.Queue) -> list[dict]:
    events = []
    while True:
        event = await asyncio.wait_for(queue.get(), timeout=5)
        events.append(event)
        if event["type"] in ("complete", "error"):
            return events


async def test_running_episode_keeps_event_loop_responsive(blocking_env):
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    ticking = asyncio.create_task(ticker())
    episode_id = await rs.start_episode("env", "task", 1, "random")
    events = await _drain(rs.episode_queues[episode_id])
    ticking.cancel()

    assert events[-1]["type"] == "complete"
    # A blocked loop gets one tick per step. A free loop gets dozens.
    assert ticks > EPISODE_STEPS * 3
    assert threading.get_ident() not in blocking_env.thread_ids


async def test_episode_events_report_steps_and_final_reward(blocking_env):
    episode_id = await rs.start_episode("env", "task", 1, "random")
    events = await _drain(rs.episode_queues[episode_id])

    steps = [e for e in events if e["type"] == "step"]
    assert [e["step_index"] for e in steps] == list(range(EPISODE_STEPS))
    assert events[-1] == {
        "type": "complete",
        "total_reward": 1.5,
        "passed": True,
        "total_steps": EPISODE_STEPS,
    }


async def test_cancelling_an_episode_stops_the_env(blocking_env):
    blocking_env._limit = 1_000
    episode_id = await rs.start_episode("env", "task", 1, "random")
    await asyncio.sleep(STEP_SECONDS * 2)
    rs.episode_tasks[episode_id].cancel()
    await asyncio.sleep(STEP_SECONDS * 4)
    stopped_at = blocking_env.steps_taken
    await asyncio.sleep(STEP_SECONDS * 4)
    assert blocking_env.steps_taken == stopped_at
    assert stopped_at < 20
