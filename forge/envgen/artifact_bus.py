from __future__ import annotations
import asyncio
from typing import Any, Callable, Awaitable, Iterable, Mapping

from forge.envgen.a2a import A2AProtocol, AgentMessage, MessageKind


class ArtifactBus:
    def __init__(self, protocol: A2AProtocol | None = None) -> None:
        self._events: dict[str, asyncio.Event] = {}
        self._values: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._callbacks: list[Callable[[str, Any], Awaitable[None]]] = []
        self._log_callbacks: list[Callable[[str], Awaitable[None]]] = []
        self.protocol = protocol or A2AProtocol()

    def on_publish(self, callback: Callable[[str, Any], Awaitable[None]]) -> None:
        self._callbacks.append(callback)

    def on_log(self, callback: Callable[[str], Awaitable[None]]) -> None:
        self._log_callbacks.append(callback)

    async def log(self, message: str) -> None:
        for cb in self._log_callbacks:
            await cb(message)

    async def publish(self, name: str, value: Any) -> None:
        async with self._lock:
            self._values[name] = value
            if name not in self._events:
                self._events[name] = asyncio.Event()
            self._events[name].set()
        for cb in self._callbacks:
            await cb(name, value)

    async def wait_for(self, name: str) -> Any:
        async with self._lock:
            if name not in self._events:
                self._events[name] = asyncio.Event()
            event = self._events[name]
        await event.wait()
        return self._values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def invalidate(self, names: Iterable[str]) -> None:
        """Drop cached artifacts so consumers re-block until they are republished.

        Used by the repair loop to force downstream specialists and the reviewers
        to wait for freshly regenerated code instead of reading stale values.
        """
        for name in names:
            self._values.pop(name, None)
            self._events.pop(name, None)

    def scoped(
        self,
        *,
        agent_id: str,
        task_id: str,
        readable: set[str],
        writable: set[str],
    ) -> AgentChannel:
        self.protocol.register_scope(task_id, readable)
        return AgentChannel(
            bus=self,
            agent_id=agent_id,
            task_id=task_id,
            readable=frozenset(readable),
            writable=frozenset(writable),
        )

    def snapshot(self) -> Mapping[str, Any]:
        return dict(self._values)


class AgentChannel:
    """Least-privilege view of the artifact bus for one agent task."""

    def __init__(
        self,
        *,
        bus: ArtifactBus,
        agent_id: str,
        task_id: str,
        readable: frozenset[str],
        writable: frozenset[str],
    ) -> None:
        self._bus = bus
        self.agent_id = agent_id
        self.task_id = task_id
        self._readable = readable
        self._writable = writable

    async def log(self, message: str) -> None:
        await self._bus.log(message)

    async def publish(self, name: str, value: Any) -> None:
        if name not in self._writable:
            raise PermissionError(
                f"Agent {self.agent_id!r} cannot publish undeclared artifact {name!r}"
            )
        await self._bus.publish(name, value)
        self._bus.protocol.send(AgentMessage(
            sender=self.agent_id,
            recipient="orchestrator",
            kind=MessageKind.ARTIFACT_AVAILABLE,
            task_id=self.task_id,
            payload={"artifact": name},
        ))

    async def wait_for(self, name: str) -> Any:
        self._assert_readable(name)
        return await self._bus.wait_for(name)

    def get(self, name: str, default: Any = None) -> Any:
        self._assert_readable(name)
        return self._bus.get(name, default)

    def relevant_context(self) -> Mapping[str, Any]:
        return self._bus.protocol.context_for(self.task_id, self._bus.snapshot())

    def _assert_readable(self, name: str) -> None:
        if name not in self._readable:
            raise PermissionError(
                f"Agent {self.agent_id!r} cannot read undeclared artifact {name!r}"
            )
