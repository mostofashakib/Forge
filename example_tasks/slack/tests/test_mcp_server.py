#!/usr/bin/env python3
"""The MCP server that hands the Slack tools to the agent's harness.

The agent is not told in prose that a `slack` command exists. Harbor registers
this stdio server from `task.toml`, so the workspace arrives as native tools
with their own schemas -- the same way a real Slack integration would.

What these checks protect:

  * the handshake an MCP client performs before it will call anything;
  * that the tool list is the workspace's own list, not a curated subset;
  * that a refusal from the workspace stays a refusal instead of becoming a
    crash or, worse, a silent success;
  * that this module remains client-only, so shipping it into the agent's
    image cannot leak the seed or the database.
"""

from __future__ import annotations

import ast
import json
import socket
import sys
import tempfile
import threading
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slack_sim import mcp_server
from slack_sim.service import TOOL_NAMES


class _Workspace:
    """A stand-in for the environment's container, one socket, canned answers."""

    def __init__(self, path: Path, responder: Callable[[dict[str, Any]], dict[str, Any]]):
        self.path = str(path)
        self.requests: list[dict[str, Any]] = []
        self._responder = responder
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.path)
        self._server.listen(8)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                client, _ = self._server.accept()
            except OSError:
                return
            with client:
                chunks: list[bytes] = []
                while not chunks or not chunks[-1].endswith(b"\n"):
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                payload = json.loads(b"".join(chunks).strip() or b"{}")
                self.requests.append(payload)
                client.sendall(json.dumps(self._responder(payload)).encode() + b"\n")

    def close(self) -> None:
        self._server.close()


def _ok(result: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda _request: {"ok": True, "result": result}


def _initialize(protocol: str = "2025-06-18") -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": protocol,
            "capabilities": {},
            "clientInfo": {"name": "test-harness", "version": "0"},
        },
    }


def test_initialize_advertises_tools(root: Path) -> None:
    response = mcp_server.dispatch(_initialize(), socket_path="/nonexistent")
    assert response is not None
    result = response["result"]
    assert response["jsonrpc"] == "2.0" and response["id"] == 1
    assert "tools" in result["capabilities"], "a client that sees no tool capability never lists tools"
    assert result["serverInfo"]["name"] == "slack"


def test_initialize_answers_in_the_clients_protocol_version(root: Path) -> None:
    for version in mcp_server.SUPPORTED_PROTOCOL_VERSIONS:
        response = mcp_server.dispatch(_initialize(version), socket_path="/nonexistent")
        assert response["result"]["protocolVersion"] == version


def test_initialize_falls_back_when_the_client_speaks_something_else(root: Path) -> None:
    response = mcp_server.dispatch(_initialize("1999-01-01"), socket_path="/nonexistent")
    assert response["result"]["protocolVersion"] == mcp_server.PROTOCOL_VERSION


def test_a_notification_is_never_answered(root: Path) -> None:
    notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert mcp_server.dispatch(notification, socket_path="/nonexistent") is None


def test_tools_list_is_the_whole_workspace_surface(root: Path) -> None:
    response = mcp_server.dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, socket_path="/nonexistent"
    )
    tools = response["result"]["tools"]
    assert {tool["name"] for tool in tools} == set(TOOL_NAMES)
    for tool in tools:
        assert tool["description"].strip(), f"{tool['name']} has no description"
        assert tool["inputSchema"]["type"] == "object", f"{tool['name']} has no object schema"
        assert "input_schema" not in tool, "MCP spells it inputSchema; a client ignores the other"


def test_tools_call_reaches_the_workspace_and_returns_its_result(root: Path) -> None:
    workspace = _Workspace(root / "agent.sock", _ok({"channels": [{"channel_id": "C001"}]}))
    try:
        response = mcp_server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "list_channels", "arguments": {"limit": 5}},
            },
            socket_path=workspace.path,
        )
    finally:
        workspace.close()

    assert workspace.requests == [
        {"op": "call_tool", "tool": "list_channels", "input": {"limit": 5}}
    ]
    result = response["result"]
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["result"]["channels"][0]["channel_id"] == "C001"


def test_a_tool_result_is_serialized_without_pretty_printing(root: Path) -> None:
    """Indentation is context the agent pays for and cannot read.

    A workspace page is deeply nested, so `indent=2` inflated every result by
    a third before it reached the model -- roughly 75,000 tokens across one
    sweep of the workspace, spent entirely on leading spaces and newlines. The
    payload is parsed, never eyeballed, so the whitespace bought nothing.
    """
    envelope = {"ok": True, "result": {"messages": [{"id": "M1", "text": "hello", "nested": {"a": [1, 2]}}]}}
    wrapped = mcp_server._tool_result(envelope)
    text = wrapped["content"][0]["text"]

    assert json.loads(text) == envelope, "compacting must not change what is said"
    assert "\n" not in text, "the result is still pretty-printed"
    assert ": " not in text and ", " not in text, "the result still carries filler spaces"


def test_tools_call_defaults_missing_arguments_to_an_empty_input(root: Path) -> None:
    workspace = _Workspace(root / "agent.sock", _ok({"channels": []}))
    try:
        mcp_server.dispatch(
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
             "params": {"name": "list_channels"}},
            socket_path=workspace.path,
        )
    finally:
        workspace.close()
    assert workspace.requests[0]["input"] == {}


def test_a_refusal_from_the_workspace_stays_a_refusal(root: Path) -> None:
    refusal = {
        "ok": False,
        "error": {"code": "permission_denied", "type": "permission",
                  "message": "You are not a member of that channel."},
    }
    workspace = _Workspace(root / "agent.sock", lambda _request: refusal)
    try:
        response = mcp_server.dispatch(
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
             "params": {"name": "get_channel_messages", "arguments": {"channel_id": "C024"}}},
            socket_path=workspace.path,
        )
    finally:
        workspace.close()

    result = response["result"]
    assert result["isError"] is True, "a denial reported as success teaches the agent a lie"
    assert json.loads(result["content"][0]["text"])["error"]["code"] == "permission_denied"


def test_an_unreachable_workspace_is_reported_not_raised(root: Path) -> None:
    response = mcp_server.dispatch(
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
         "params": {"name": "list_channels", "arguments": {}}},
        socket_path=str(root / "absent.sock"),
    )
    result = response["result"]
    assert result["isError"] is True
    assert json.loads(result["content"][0]["text"])["error"]["code"] == "workspace_unavailable"


def test_a_call_without_a_tool_name_is_an_invalid_request(root: Path) -> None:
    response = mcp_server.dispatch(
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"arguments": {}}},
        socket_path="/nonexistent",
    )
    assert response["error"]["code"] == mcp_server.INVALID_PARAMS


def test_an_unknown_method_is_an_unknown_method(root: Path) -> None:
    response = mcp_server.dispatch(
        {"jsonrpc": "2.0", "id": 8, "method": "resources/list"}, socket_path="/nonexistent"
    )
    assert response["error"]["code"] == mcp_server.METHOD_NOT_FOUND


def test_ping_is_answered_so_a_client_does_not_time_out(root: Path) -> None:
    response = mcp_server.dispatch(
        {"jsonrpc": "2.0", "id": 9, "method": "ping"}, socket_path="/nonexistent"
    )
    assert response["result"] == {}


def test_the_stream_answers_requests_and_skips_notifications(root: Path) -> None:
    stream = "\n".join(
        [
            json.dumps(_initialize()),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        ]
    )
    out = StringIO()
    mcp_server.serve(StringIO(stream), out, socket_path="/nonexistent")

    lines = out.getvalue().splitlines()
    assert len(lines) == 2, "one response per request, and nothing for a notification"
    assert [json.loads(line)["id"] for line in lines] == [1, 2]


def test_a_malformed_line_is_a_parse_error_and_the_server_keeps_going(root: Path) -> None:
    stream = "{not json\n" + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    out = StringIO()
    mcp_server.serve(StringIO(stream), out, socket_path="/nonexistent")

    first, second = (json.loads(line) for line in out.getvalue().splitlines())
    assert first["error"]["code"] == mcp_server.PARSE_ERROR
    assert second["id"] == 2, "one bad frame must not end the session"


def test_the_server_stays_client_only(root: Path) -> None:
    """This module ships in the agent's image, so what it imports is a boundary.

    `slack_sim.service` and `slack_sim.seed` carry the database and the answer
    key. If this module ever imports one, the agent's container gains the means
    to build a private copy of the workspace and read channels it was refused.
    """
    source = Path(mcp_server.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {name for name in imported
                 if name.startswith("slack_sim.")
                 and name.split(".")[1] not in {"protocol", "tool_definitions"}}
    assert not forbidden, f"the agent's image must not gain {sorted(forbidden)}"


def main() -> None:
    tests = sorted(
        (value for name, value in globals().items()
         if name.startswith("test_") and callable(value)),
        key=lambda fn: fn.__name__,
    )
    with tempfile.TemporaryDirectory() as directory:
        for index, test in enumerate(tests):
            root = Path(directory) / str(index)
            root.mkdir()
            test(root)
            print(f"  {test.__name__}: ok")
    print(f"mcp server: ok ({len(tests)} tests)")


if __name__ == "__main__":
    main()
