"""The browser sandbox proxy: HTTP and WebSocket traffic to one browser sandbox, by name.

The browser desktop (Selkies) streams over a WebSocket at a path relative to the
page, so a proxy that only carries HTTP leaves the desktop blank.
"""
from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.server import serve

from backend.app.models import SandboxEnvironment


class _EchoHandler(BaseHTTPRequestHandler):
    """Answers with what it received, so tests see exactly what the proxy sent."""

    def _answer(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length).decode() if length else ""
        payload = json.dumps({
            "method": self.command, "path": self.path, "body": body,
            "actor": self.headers.get("x-forge-actor"),
        }).encode()
        self.send_response(201 if self.command == "POST" else 200)
        self.send_header("content-type", "application/json")
        self.send_header("x-upstream", "yes")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = _answer

    def log_message(self, *_args):
        pass


@pytest.fixture(scope="module")
def http_upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


@pytest.fixture(scope="module")
def ws_upstream():
    """A WebSocket server that echoes each message and reports its path once."""
    ready = threading.Event()
    state: dict = {}

    async def echo(connection):
        await connection.send(f"path={connection.request.path}")
        async for message in connection:
            await connection.send(message)

    def run():
        async def main():
            async with serve(echo, "127.0.0.1", 0, subprotocols=["selkies"]) as server:
                state["port"] = server.sockets[0].getsockname()[1]
                state["stop"] = asyncio.get_running_loop().create_future()
                state["loop"] = asyncio.get_running_loop()
                ready.set()
                await state["stop"]
        asyncio.run(main())

    threading.Thread(target=run, daemon=True).start()
    ready.wait(5)
    yield state["port"]
    state["loop"].call_soon_threadsafe(state["stop"].set_result, None)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.delenv("FORGE_SANDBOX_HOST", raising=False)
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    from backend.app.main import app
    yield TestClient(app)
    database._engine = None
    database._SessionLocal = None


def _add(env_name: str, *, port: int | None, env_type: str = "browser", status: str = "running") -> None:
    from backend.app import database
    with database.get_session_factory()() as db:
        db.add(SandboxEnvironment(
            id=env_name, status=status, env_type=env_type, container_port=port,
            container_id="c1", ttl_days=30, expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        ))
        db.commit()


def test_get_reaches_the_sandbox_with_its_path_and_query(client, http_upstream):
    _add("web", port=http_upstream)

    response = client.get("/api/sandbox/web/browser/assets/index.js?v=2")

    assert response.status_code == 200
    assert response.json()["path"] == "/assets/index.js?v=2"
    assert response.json()["actor"] == "user"
    assert response.headers["x-upstream"] == "yes"


def test_the_root_page_reaches_the_sandbox_root(client, http_upstream):
    _add("web", port=http_upstream)

    assert client.get("/api/sandbox/web/browser/").json()["path"] == "/"


def test_post_forwards_the_body_and_status(client, http_upstream):
    _add("web", port=http_upstream)

    response = client.post("/api/sandbox/web/browser/api/settings", content=b'{"x":1}')

    assert response.status_code == 201
    assert response.json() == {"method": "POST", "path": "/api/settings", "body": '{"x":1}', "actor": "user"}


def test_unknown_sandbox_is_404(client):
    assert client.get("/api/sandbox/missing/browser/").status_code == 404


def test_a_general_sandbox_is_rejected(client, http_upstream):
    _add("app", port=http_upstream, env_type="general")

    response = client.get("/api/sandbox/app/browser/")

    assert response.status_code == 409
    assert "not a browser sandbox" in response.json()["detail"]


def test_a_stopped_sandbox_is_rejected(client, http_upstream):
    _add("web", port=http_upstream, status="stopped")

    assert client.get("/api/sandbox/web/browser/").status_code == 409


def test_a_dot_dot_segment_is_rejected(client, http_upstream):
    _add("web", port=http_upstream)

    assert client.get("/api/sandbox/web/browser/a/%2E%2E/b").status_code == 400


def test_an_unreachable_sandbox_is_502(client):
    _add("web", port=1)

    response = client.get("/api/sandbox/web/browser/")

    assert response.status_code == 502
    assert "unreachable" in response.json()["detail"]


def test_websocket_messages_flow_both_ways_on_the_page_relative_path(client, ws_upstream):
    _add("web", port=ws_upstream)

    with client.websocket_connect("/api/sandbox/web/browser/api/websockets?role=viewer", subprotocols=["selkies"]) as ws:
        assert ws.receive_text() == "path=/api/websockets?role=viewer"
        ws.send_text("hello")
        assert ws.receive_text() == "hello"
        ws.send_bytes(b"\x00\x01frame")
        assert ws.receive_bytes() == b"\x00\x01frame"


def test_websocket_to_a_non_browser_sandbox_is_refused(client, ws_upstream):
    _add("app", port=ws_upstream, env_type="general")

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/sandbox/app/browser/api/websockets") as ws:
            ws.receive_text()
