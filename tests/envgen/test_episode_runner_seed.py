"""Seed threading through the container episode runner's reset."""
from __future__ import annotations

import json

import httpx

from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig


def _runner_with_capture() -> tuple[ContainerEpisodeRunner, dict]:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/forge/reset":
            captured["reset_body"] = json.loads(request.content) if request.content else None
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/forge/state":
            return httpx.Response(200, json={"todos": []})
        return httpx.Response(404, json={"error": "unknown"})

    runner = ContainerEpisodeRunner(EpisodeConfig(base_url="http://c", objective="do it"))
    runner._http = httpx.Client(
        base_url="http://c", transport=httpx.MockTransport(handler)
    )
    return runner, captured


def test_reset_forwards_seed():
    runner, captured = _runner_with_capture()
    runner._reset(seed=7)
    assert captured["reset_body"] == {"seed": 7}


def test_reset_without_seed_sends_no_body():
    runner, captured = _runner_with_capture()
    runner._reset()
    assert captured["reset_body"] in (None, {})
