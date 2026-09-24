"""Sandbox API behavior under load: event-loop blocking, Docker client
lifetime, and the progress-stream race."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
from sqlalchemy.orm import Session

from backend.app.models import SandboxEnvironment
from tests.backend.test_sandbox_e2e import _add_sandbox, _mock_create_deps, client  # noqa: F401


def _set_container(env_name: str, **fields) -> None:
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sandbox = db.get(SandboxEnvironment, env_name)
        for name, value in fields.items():
            setattr(sandbox, name, value)
        db.commit()
    finally:
        db.close()


async def test_slow_create_does_not_stall_other_requests(client, monkeypatch):
    from backend.app.api import sandbox as sandbox_api

    def slow_limit() -> int:
        time.sleep(0.4)
        return 10

    monkeypatch.setattr(sandbox_api, "sandbox_limit", slow_limit)
    redis_patch, task_patch = _mock_create_deps()
    transport = httpx.ASGITransport(app=client.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        with redis_patch, task_patch:
            create = asyncio.create_task(http.post("/api/sandbox/", json={"env_name": "slow_env"}))
            started = time.perf_counter()
            await asyncio.sleep(0.05)
            loop_lag = time.perf_counter() - started
            created = await create

    assert created.status_code == 202
    # A handler that blocks the loop delays this 50 ms sleep by the full 400 ms.
    assert loop_lag < 0.25

def test_dispatch_timeout_marks_sandbox_error(client, monkeypatch):
    from backend.app.api import sandbox as sandbox_api

    monkeypatch.setattr(sandbox_api, "DISPATCH_TIMEOUT_S", 0.1)
    redis_patch, task_patch = _mock_create_deps()
    with redis_patch, task_patch as delay:
        delay.side_effect = lambda **_kw: time.sleep(0.5)
        response = client.post("/api/sandbox/", json={"env_name": "stuck_env"})

    assert response.status_code == 503
    assert client.get("/api/sandbox/stuck_env").json()["status"] == "error"


@pytest.mark.parametrize("path", ["/api/sandbox/leak_env", "/api/sandbox/leak_env/logs"])
def test_docker_client_is_closed_after_each_request(client, path):
    _add_sandbox(client, "leak_env", status="running")
    _set_container("leak_env", container_id="cid", container_port=32000, image_tag="img:latest")
    container = MagicMock(status="running", attrs={"RestartCount": 0, "State": {}})
    container.ports = {"8000/tcp": [{"HostPort": "32000"}]}
    container.logs.return_value = b"hello"
    docker_client = MagicMock()
    docker_client.containers.get.return_value = container

    with patch("docker.from_env", return_value=docker_client):
        assert client.get(path).status_code == 200

    docker_client.close.assert_called_once()


class _SilentPubSub:
    async def subscribe(self, _channel):
        pass

    async def unsubscribe(self, _channel):
        pass

    async def listen(self):
        return
        yield


def _silent_redis():
    redis_client = MagicMock()
    redis_client.pubsub.return_value = _SilentPubSub()
    redis_client.aclose = MagicMock(side_effect=lambda: asyncio.sleep(0))
    return redis_client


@pytest.mark.parametrize(("status", "expected"), [
    ("running", {"done": True}),
    ("error", {"done": True, "error": "Build failed — check the worker logs for details."}),
])
def test_progress_stream_reports_a_build_that_already_finished(client, status, expected):
    _add_sandbox(client, "finished_env", status=status)
    with patch("backend.app.api.sandbox.redis.asyncio.from_url", return_value=_silent_redis()):
        with client.websocket_connect("/api/sandbox/ws/progress/finished_env") as ws:
            assert ws.receive_json() == expected


async def test_relay_stops_when_the_client_disconnects():
    from fastapi import WebSocketDisconnect
    from backend.app.api._pubsub_relay import relay_pubsub

    class _HangingPubSub:
        async def listen(self):
            await asyncio.Event().wait()
            yield

    websocket = MagicMock()

    async def receive():
        raise WebSocketDisconnect()

    websocket.receive = receive
    await asyncio.wait_for(
        relay_pubsub(websocket, _HangingPubSub(), is_final=lambda _msg: False),
        timeout=1,
    )


def test_activity_stream_reaps_the_docker_logs_process(client):
    _add_sandbox(client, "logs_env", status="running")
    _set_container("logs_env", container_id="cid")
    proc = MagicMock()
    proc.stdout.readline = MagicMock(side_effect=lambda: asyncio.sleep(0, result=b""))
    proc.wait = MagicMock(side_effect=lambda: asyncio.sleep(0, result=0))

    async def fake_exec(*_args, **_kwargs):
        return proc

    with patch("backend.app.api.sandbox.asyncio.create_subprocess_exec", fake_exec):
        with client.websocket_connect("/api/sandbox/ws/activity/logs_env"):
            pass

    proc.terminate.assert_called_once()
    proc.wait.assert_called_once()


def test_exec_stream_reaps_the_shell_process(client):
    _add_sandbox(client, "shell_env", status="running")
    _set_container("shell_env", container_id="cid")
    proc = MagicMock()

    with patch("subprocess.Popen", return_value=proc):
        with client.websocket_connect("/api/sandbox/ws/exec/shell_env"):
            pass

    proc.terminate.assert_called_once()
    proc.wait.assert_called_once()
