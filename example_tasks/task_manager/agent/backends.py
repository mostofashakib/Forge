"""Where tool calls are actually executed.

The provider adapters decide which model answers; these decide where its
answers land. The two axes are independent on purpose -- the same loop drives
Ollama against a Harbor container, or a hosted model against a throwaway
workspace on this machine, with no change to either half.

Every backend answers the same two questions: what tools exist, and what
happened when one was called. The reply is the world's own envelope,
``{"ok": true, "result": ...}`` or ``{"ok": false, "error": {...}}``, passed
through unflattened -- a refusal is information the model needs, not an
exception the loop should swallow.

The methods are async because one of the backends is: Harbor's ``environment``
is an async API, and making the other two await trivially is cheaper than
bridging an event loop from inside a synchronous one.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class BackendError(RuntimeError):
    """The tool surface itself could not be reached or did not answer in the
    agreed shape. A refused *tool call* is not this: that comes back as
    ``{"ok": false}`` and is shown to the model."""


class ToolBackend(ABC):
    """A place where the tracker's tools can be run."""

    @abstractmethod
    async def list_tools(self) -> list[dict[str, Any]]:
        """The tool schemas, in the environment's own shape."""

    @abstractmethod
    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one tool and return the world's envelope."""

    async def aclose(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"backend": type(self).__name__}


class SubprocessBackend(ToolBackend):
    """Drive the ``tasks`` CLI as a child process.

    ``prefix`` is what goes in front of it, which is the whole of the
    configuration: empty runs the CLI on this machine (inside the agent
    container), and ``["docker", "exec", "-u", "agent", "<container>"]`` runs it
    in a compose stack from outside. Either way the workspace stays on the far
    side of the socket -- this holds no state.
    """

    def __init__(
        self,
        prefix: Sequence[str] = (),
        *,
        executable: str = "tasks",
        timeout_sec: float = 120.0,
    ) -> None:
        self.prefix = tuple(prefix)
        self.executable = executable
        self.timeout_sec = timeout_sec

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "subprocess",
            "command": " ".join([*self.prefix, self.executable]),
        }

    async def _run(self, argv: Sequence[str]) -> dict[str, Any]:
        command = [*self.prefix, self.executable, *argv]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise BackendError(f"Could not run {shlex.join(command)}: {exc}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_sec
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise BackendError(
                f"{shlex.join(command)} did not finish within {self.timeout_sec}s"
            ) from None
        # The CLI prints the envelope on stdout for a success or a refusal, and
        # on stderr only when it could not reach the socket at all. Both are
        # JSON, so both are read the same way.
        text = (stdout or b"").decode("utf-8", "replace").strip()
        if not text:
            text = (stderr or b"").decode("utf-8", "replace").strip()
        if not text:
            raise BackendError(
                f"{shlex.join(command)} exited {process.returncode} with no output."
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise BackendError(
                f"{shlex.join(command)} did not return JSON: {text[:500]}"
            ) from exc

    async def list_tools(self) -> list[dict[str, Any]]:
        response = await self._run(["list_tools"])
        return list((response.get("result") or {}).get("tools") or [])

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._run(
            [name, "--input-payload", json.dumps(arguments, sort_keys=True)]
        )


class HarborBackend(ToolBackend):
    """Run the ``tasks`` CLI inside a Harbor-managed environment.

    Harbor already owns the container and the user the agent runs as, so this
    hands the command back to it rather than reaching for Docker directly.
    """

    def __init__(self, environment: Any, *, executable: str = "tasks", timeout_sec: int = 120) -> None:
        self.environment = environment
        self.executable = executable
        self.timeout_sec = timeout_sec

    def describe(self) -> dict[str, Any]:
        return {"backend": "harbor", "command": self.executable}

    async def _run(self, argv: Sequence[str]) -> dict[str, Any]:
        command = shlex.join([self.executable, *argv])
        result = await self.environment.exec(command=command, timeout_sec=self.timeout_sec)
        text = (result.stdout or "").strip() or (result.stderr or "").strip()
        if not text:
            raise BackendError(f"`{command}` exited {result.return_code} with no output.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise BackendError(f"`{command}` did not return JSON: {text[:500]}") from exc

    async def list_tools(self) -> list[dict[str, Any]]:
        response = await self._run(["list_tools"])
        return list((response.get("result") or {}).get("tools") or [])

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._run(
            [name, "--input-payload", json.dumps(arguments, sort_keys=True)]
        )


class InProcessBackend(ToolBackend):
    """Call the world directly, with no container and no socket.

    For trying a model against the task without Docker, and for tests that need
    a real workspace and a real refusal without one. It imports the world's own
    modules, so it can only be used where those are present -- which is the
    repository and the graded image, never the agent's container.
    """

    def __init__(self, db_path: Path | str, *, actor_id: str | None = None) -> None:
        self.db_path = Path(db_path)
        self.actor_id = actor_id

    def describe(self) -> dict[str, Any]:
        return {"backend": "in-process", "db": str(self.db_path)}

    async def list_tools(self) -> list[dict[str, Any]]:
        from task_sim.tool_definitions import get_tool_definitions

        return list(get_tool_definitions())

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from task_sim.server import _from_error
        from task_sim.service import execute_tool
        from task_sim.sqlite_common import WorldError

        kwargs = {"actor_id": self.actor_id} if self.actor_id else {}
        try:
            return {"ok": True, "result": execute_tool(self.db_path, name, arguments, **kwargs)}
        except WorldError as exc:
            # Borrowed rather than reimplemented: the server documents itself as
            # the only place a WorldError becomes a response, and a second
            # spelling of the refusal envelope here would let a model tell the
            # backends apart -- and would drift.
            return _from_error(exc)
