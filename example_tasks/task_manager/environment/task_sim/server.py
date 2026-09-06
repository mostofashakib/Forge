"""The tracker world server: the only way in from outside the environment.

This process owns the workspace database. It runs in the environment's own
container as the ``tasksd`` user, and nothing outside that container has a
filesystem path to the state. Callers reach it through two Unix sockets whose
file permissions carry the privilege boundary:

``agent.sock`` (0666)
    The tracker tool surface, nothing else. This is what the agent is given.

``admin.sock`` (0600, root)
    The tools plus lifecycle and inspection operations used to build the image
    and to grade a finished episode. The agent runs as an unprivileged user and
    cannot open it.

The split is by socket rather than by a flag on one endpoint so that reaching an
admin operation from the agent's socket is not a check that could be bypassed --
the dispatch table simply does not contain it.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import socketserver
import threading
from pathlib import Path
from typing import Any

from task_sim.identity import LOGGED_IN_USER
from task_sim.protocol import ADMIN_SOCKET, AGENT_SOCKET, ProtocolError, decode, encode
from task_sim.service import (
    DEFAULT_DB_PATH,
    DEFAULT_SNAPSHOT_PATH,
    TOOL_NAMES,
    execute_tool,
    export_state,
    record_integrity_violation,
    seed_database,
    teardown_database,
)
from task_sim.sqlite_common import (
    InternalError,
    UnknownToolError,
    WorldError,
    storage_errors,
)

LOGGER = logging.getLogger("tasksd")


class WorldService:
    """Applies requests to the workspace. One instance guards one database."""

    def __init__(self, db_path: Path, snapshot_path: Path) -> None:
        self.db_path = db_path
        self.snapshot_path = snapshot_path
        # SQLite handles its own locking, but serializing here keeps the
        # COUNT(*)-based id allocation in service.py free of races.
        self._lock = threading.Lock()

    # -- agent surface ---------------------------------------------------
    def call_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in TOOL_NAMES:
            return _from_error(UnknownToolError(tool_name))
        with self._lock:
            try:
                result = execute_tool(
                    self.db_path, tool_name, payload, actor_id=LOGGED_IN_USER.user_id
                )
            # Every deliberate failure of the world is a WorldError carrying its
            # own code, so one clause reports a rule violation, a corrupt
            # database and a bad tool name each under its own name.
            except WorldError as exc:
                return _from_error(exc)
        return {"ok": True, "result": result}

    def list_tools(self) -> dict[str, Any]:
        from task_sim.tool_definitions import get_tool_definitions

        return {"ok": True, "result": {"tools": get_tool_definitions()}}

    # -- admin surface ---------------------------------------------------
    # These read and rebuild the database directly, so they meet the same
    # storage faults the tool surface does and are given the same name for them.
    # A grader told `storage_error` knows the export is untrustworthy; one told
    # `internal_error` cannot tell that from a bug in the exporter.
    def export(self) -> dict[str, Any]:
        with self._lock, storage_errors():
            return {"ok": True, "result": export_state(self.db_path)}

    def seed(self) -> dict[str, Any]:
        with self._lock, storage_errors():
            seed_database(self.db_path, self.snapshot_path)
        return {"ok": True, "result": {"seeded": True}}

    def teardown(self) -> dict[str, Any]:
        with self._lock, storage_errors():
            teardown_database(self.db_path, self.snapshot_path)
        return {"ok": True, "result": {"teardown": True}}


def _error(code: str, kind: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "type": kind, "message": message}}


def _from_error(exc: WorldError) -> dict[str, Any]:
    """The only place a `WorldError` becomes a response.

    Reading the code off the exception rather than off the catch site keeps the
    taxonomy from drifting: a new subclass is reportable the moment it exists,
    and no handler can quietly relabel one kind of failure as another.
    """
    return _error(exc.error_code, exc.error_type, str(exc))


AGENT_OPS = ("call_tool", "list_tools", "ping")
ADMIN_OPS = AGENT_OPS + ("export_state", "seed", "teardown")


def _dispatch(
    world: WorldService, allowed: tuple[str, ...], request: dict[str, Any]
) -> dict[str, Any]:
    op = request.get("op", "call_tool")
    if op not in allowed:
        # An admin op named on the agent socket is reported as unknown; the
        # agent surface must not disclose that a privileged surface exists.
        # It is still written down. Guessing `list_toolz` is a typo; asking the
        # unprivileged socket to export the answer key is not, and the refusal
        # is the only trace the episode would otherwise keep of it.
        if op in ADMIN_OPS:
            record_integrity_violation(
                world.db_path, "privileged_operation", f"op={op!r} on the agent socket"
            )
        return _error("unknown_operation", "validation_error", f"Unknown operation: {op}")
    if op == "ping":
        return {"ok": True, "result": {"pong": True}}
    if op == "list_tools":
        return world.list_tools()
    if op == "call_tool":
        tool_name = request.get("tool")
        if not isinstance(tool_name, str) or not tool_name:
            return _error("invalid_request", "validation_error", "A 'tool' name is required.")
        payload = request.get("input") or {}
        if not isinstance(payload, dict):
            return _error("invalid_request", "validation_error", "'input' must be an object.")
        return world.call_tool(tool_name, payload)
    if op == "export_state":
        return world.export()
    if op == "seed":
        return world.seed()
    if op == "teardown":
        return world.teardown()
    return _error("unknown_operation", "validation_error", f"Unknown operation: {op}")


class _Handler(socketserver.StreamRequestHandler):
    timeout = 60

    def handle(self) -> None:  # pragma: no cover - exercised over a real socket
        try:
            line = self.rfile.readline()
            if not line:
                return
            try:
                request = decode(line.strip())
            except ProtocolError as exc:
                response = _error("malformed_request", "validation_error", str(exc))
            else:
                try:
                    response = _dispatch(self.server.world, self.server.allowed_ops, request)
                except WorldError as exc:
                    LOGGER.warning("%s while dispatching: %s", exc.error_code, exc)
                    response = _from_error(exc)
                except Exception as exc:  # noqa: BLE001 - never kill the server
                    # The residue, and now only the residue. Anything reaching
                    # here is a defect rather than a condition, which is why it
                    # is the one branch that logs a traceback.
                    LOGGER.exception("Unhandled error while dispatching")
                    response = _from_error(InternalError(f"{type(exc).__name__}: {exc}"))
            self.wfile.write(encode(response))
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            return


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, path: str, world: WorldService, allowed_ops: tuple[str, ...], mode: int
    ) -> None:
        socket_path = Path(path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists():
            socket_path.unlink()
        super().__init__(str(socket_path), _Handler)
        os.chmod(socket_path, mode)
        self.world = world
        self.allowed_ops = allowed_ops


def serve(
    db_path: Path = DEFAULT_DB_PATH,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    agent_socket: str = AGENT_SOCKET,
    admin_socket: str = ADMIN_SOCKET,
) -> None:
    world = WorldService(db_path, snapshot_path)
    # Every run starts from the same initial workspace. The server process
    # starting is the start of a run, so seed unconditionally rather than only
    # when the file happens to be missing: an episode must never inherit another
    # episode's rows or half-fired events, and relying on container freshness
    # for that makes it an accident rather than a rule.
    world.seed()
    # 0666: the socket is the sanctioned interface, so any local uid may speak
    # it. Authority comes from what the surface exposes, not from who connects.
    agent = _Server(agent_socket, world, AGENT_OPS, 0o666)
    # 0600 and owned by the server's uid: the unprivileged agent user cannot
    # open this at all.
    admin = _Server(admin_socket, world, ADMIN_OPS, 0o600)
    LOGGER.info("tasksd listening on %s (agent) and %s (admin)", agent_socket, admin_socket)
    threading.Thread(target=admin.serve_forever, daemon=True).start()
    agent.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-tracker world server.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT_PATH))
    parser.add_argument("--agent-socket", default=AGENT_SOCKET)
    parser.add_argument("--admin-socket", default=ADMIN_SOCKET)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve(Path(args.db), Path(args.snapshot), args.agent_socket, args.admin_socket)


if __name__ == "__main__":
    main()
