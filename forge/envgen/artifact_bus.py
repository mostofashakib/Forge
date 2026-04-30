from __future__ import annotations
import asyncio
from typing import Any, Callable, Awaitable


class ArtifactBus:
    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}
        self._values: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._callbacks: list[Callable[[str, Any], Awaitable[None]]] = []

    def on_publish(self, callback: Callable[[str, Any], Awaitable[None]]) -> None:
        self._callbacks.append(callback)

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
