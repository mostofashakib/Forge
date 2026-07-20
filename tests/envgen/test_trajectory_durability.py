"""Crash-durable trajectory logging for container-based episode runners.

The trajectory JSONL must be written incrementally, so a run that crashes
mid-episode still leaves a replayable partial trace on disk.
"""
from __future__ import annotations

import json

import httpx
import pytest

from forge.envgen.episode_base import BaseEpisodeResult, TrajectoryWriter
from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig


# ---------------------------------------------------------------------------
# TrajectoryWriter unit behavior
# ---------------------------------------------------------------------------

def test_writer_persists_each_step_before_close(tmp_path):
    result = BaseEpisodeResult()  # default: steps are plain dicts
    path = tmp_path / "ep.jsonl"
    writer = TrajectoryWriter(path, result)
    writer.record({"step_index": 0, "action": {"type": "a"}})
    # No close yet — the step must already be durable on disk.
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["step_index"] == 0


def test_writer_appends_summary_on_close(tmp_path):
    result = BaseEpisodeResult(total_reward=1.0, termination_reason="success")
    path = tmp_path / "ep.jsonl"
    with TrajectoryWriter(path, result) as writer:
        writer.record({"step_index": 0})
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["type"] == "episode_summary"
    assert json.loads(lines[1])["termination_reason"] == "success"


def test_writer_closes_and_flushes_on_exception(tmp_path):
    result = BaseEpisodeResult()
    path = tmp_path / "ep.jsonl"
    with pytest.raises(RuntimeError):
        with TrajectoryWriter(path, result) as writer:
            writer.record({"step_index": 0})
            raise RuntimeError("boom")
    # Step recorded before the crash survives; the summary line is still written.
    lines = path.read_text().splitlines()
    assert json.loads(lines[0])["step_index"] == 0
    assert json.loads(lines[-1])["type"] == "episode_summary"


# ---------------------------------------------------------------------------
# Container runner leaves a partial, replayable trace on a mid-episode crash
# ---------------------------------------------------------------------------

def _counter_app():
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/forge/reset":
            state["n"] = 0
            return httpx.Response(200, json={"ok": True})
        if path == "/forge/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/forge/state":
            return httpx.Response(200, json={"n": state["n"]})
        if path == "/openapi.json":
            return httpx.Response(200, json={
                "paths": {"/inc": {"post": {"summary": "increment"}}},
                "components": {"schemas": {}},
            })
        if path == "/inc":
            state["n"] += 1
            return httpx.Response(200, json={"n": state["n"]})
        return httpx.Response(404, json={})

    return handler


class _ExplodingScorer:
    """Scores normally, then raises on the Nth step to simulate a crash."""

    def __init__(self, explode_on: int) -> None:
        self._calls = 0
        self._explode_on = explode_on

    def score(self, *args, **kwargs) -> float:
        self._calls += 1
        if self._calls >= self._explode_on:
            raise RuntimeError("scorer exploded")
        return 0.1


class _IncAgent:
    def act(self, state, objective, actions):
        return {"endpoint": "/inc", "payload": {}}


def _runner(scorer, tmp_path) -> ContainerEpisodeRunner:
    runner = ContainerEpisodeRunner(
        EpisodeConfig(base_url="http://c", objective="increment n", max_steps=10),
        scorer=scorer,
    )
    runner._http = httpx.Client(base_url="http://c", transport=httpx.MockTransport(_counter_app()))
    return runner


def test_crash_mid_episode_leaves_replayable_partial_trace(tmp_path):
    jsonl = tmp_path / "ep.jsonl"
    runner = _runner(_ExplodingScorer(explode_on=3), tmp_path)
    with pytest.raises(RuntimeError):
        runner.run_episode(_IncAgent(), episode_id="ep1", jsonl_path=jsonl)

    # Two steps completed before the 3rd scoring call blew up — both are on disk.
    lines = [json.loads(x) for x in jsonl.read_text().splitlines()]
    steps = [x for x in lines if x.get("type") != "episode_summary"]
    assert len(steps) == 2
    assert [s["step_index"] for s in steps] == [0, 1]
    assert steps[0]["action"] == {"endpoint": "/inc", "payload": {}}
    # The state change is captured so the trajectory can be replayed.
    assert steps[0]["state_after"] == {"n": 1}


def test_normal_run_writes_steps_incrementally_and_a_summary(tmp_path):
    jsonl = tmp_path / "ep.jsonl"
    runner = _runner(_ExplodingScorer(explode_on=999), tmp_path)
    runner.run_episode(_IncAgent(), episode_id="ep2", jsonl_path=jsonl)
    lines = [json.loads(x) for x in jsonl.read_text().splitlines()]
    assert lines[-1]["type"] == "episode_summary"
    steps = [x for x in lines if x.get("type") != "episode_summary"]
    assert len(steps) >= 1
