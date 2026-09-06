#!/usr/bin/env python3
"""The MCP server is a client: it speaks the protocol and owns nothing."""

from __future__ import annotations

import ast
import io
import json
import sys
from pathlib import Path

from task_sim import mcp_server
from task_sim.tool_definitions import get_tool_definitions

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def test_the_handshake() -> None:
    reply = mcp_server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                 "params": {"protocolVersion": "2025-06-18"}})
    check("initialize is answered", reply is not None and "result" in reply)
    check("the requested version is echoed",
          reply["result"]["protocolVersion"] == "2025-06-18")
    check("the server names itself", reply["result"]["serverInfo"]["name"] == "tasks")

    older = mcp_server.dispatch({"jsonrpc": "2.0", "id": 2, "method": "initialize",
                                 "params": {"protocolVersion": "2024-11-05"}})
    check("a pinned older revision is honoured",
          older["result"]["protocolVersion"] == "2024-11-05")

    unknown = mcp_server.dispatch({"jsonrpc": "2.0", "id": 3, "method": "initialize",
                                   "params": {"protocolVersion": "1999-01-01"}})
    check("an unknown revision falls back to ours",
          unknown["result"]["protocolVersion"] == mcp_server.PROTOCOL_VERSION)


def test_the_tool_list_is_generated_not_written_down() -> None:
    reply = mcp_server.dispatch({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
    listed = {tool["name"] for tool in reply["result"]["tools"]}
    declared = {tool["name"] for tool in get_tool_definitions()}
    check("every workspace tool reaches the agent", listed == declared,
          f"missing={declared - listed} extra={listed - declared}")
    check("each carries its schema",
          all("inputSchema" in tool for tool in reply["result"]["tools"]))


def test_protocol_errors_are_reported_as_protocol_errors() -> None:
    check("an unknown method is METHOD_NOT_FOUND",
          mcp_server.dispatch({"jsonrpc": "2.0", "id": 5, "method": "wat"})["error"]["code"]
          == mcp_server.METHOD_NOT_FOUND)
    check("tools/call with no name is INVALID_PARAMS",
          mcp_server.dispatch({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                               "params": {}})["error"]["code"] == mcp_server.INVALID_PARAMS)
    check("non-object params is INVALID_PARAMS",
          mcp_server.dispatch({"jsonrpc": "2.0", "id": 7, "method": "tools/list",
                               "params": []})["error"]["code"] == mcp_server.INVALID_PARAMS)


def test_a_notification_is_never_answered() -> None:
    # No id means there is nowhere to send a reply; answering one is itself a
    # protocol error.
    check("a notification gets no reply",
          mcp_server.dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None)


def test_an_unreachable_workspace_is_reported_not_fatal() -> None:
    reply = mcp_server.dispatch(
        {"jsonrpc": "2.0", "id": 8, "method": "tools/call",
         "params": {"name": "list_tasks", "arguments": {}}},
        socket_path="/nonexistent/agent.sock",
    )
    check("the server survives a workspace that is not up", "result" in reply)
    check("and reports it as an error result", reply["result"]["isError"] is True)
    envelope = json.loads(reply["result"]["content"][0]["text"])
    check("naming the condition", envelope["error"]["code"] == "workspace_unavailable",
          str(envelope))


def test_a_refusal_reaches_the_agent_as_a_refusal() -> None:
    """A denial is information. An empty success is a lie about the workspace."""
    envelope = {"ok": False, "error": {"code": "task_not_found", "type": "not_found",
                                       "message": "Task was not found."}}
    wrapped = mcp_server._tool_result(envelope)
    check("isError is set", wrapped["isError"] is True)
    check("the whole envelope is passed through",
          json.loads(wrapped["content"][0]["text"]) == envelope)


def test_results_are_serialized_compactly() -> None:
    wrapped = mcp_server._tool_result({"ok": True, "result": {"tasks": [{"task_id": "TASK001"}]}})
    text = wrapped["content"][0]["text"]
    check("no indentation is spent on a parser", "\n" not in text and ": " not in text, text)


def test_the_frame_loop_answers_each_line() -> None:
    stdin = io.StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n"
        + "\n"
        + "not json\n"
        + json.dumps({"jsonrpc": "2.0", "method": "notifications/x"}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n"
    )
    stdout = io.StringIO()
    mcp_server.serve(stdin, stdout)
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    check("blank lines and notifications produce no frames", len(replies) == 3, str(len(replies)))
    check("ping is answered", replies[0]["id"] == 1)
    check("malformed JSON is a parse error", replies[1]["error"]["code"] == mcp_server.PARSE_ERROR)
    check("the tool list still comes back", replies[2]["id"] == 2)


def test_the_client_imports_nothing_the_agent_must_not_have() -> None:
    """This module ships in the agent's image, where the seed must never be."""
    source = Path(mcp_server.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden = {
        name for name in imported
        if name.startswith("task_sim.")
        and name not in ("task_sim.protocol", "task_sim.tool_definitions")
    }
    check("it imports only the wire format and the schemas", not forbidden, str(forbidden))


def main() -> int:
    print(__doc__)
    for test in (
        test_the_handshake,
        test_the_tool_list_is_generated_not_written_down,
        test_protocol_errors_are_reported_as_protocol_errors,
        test_a_notification_is_never_answered,
        test_an_unreachable_workspace_is_reported_not_fatal,
        test_a_refusal_reaches_the_agent_as_a_refusal,
        test_results_are_serialized_compactly,
        test_the_frame_loop_answers_each_line,
        test_the_client_imports_nothing_the_agent_must_not_have,
    ):
        print(f"\n{test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all MCP checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
