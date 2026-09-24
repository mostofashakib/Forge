from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from fastapi import WebSocket, WebSocketDisconnect


async def relay_pubsub(
    websocket: WebSocket,
    pubsub,
    is_final: Callable[[dict], bool],
) -> None:
    """Forward JSON pub/sub messages to the socket until a final one or a disconnect.

    Watching the socket matters: a quiet channel never wakes `listen()`, so
    without it a closed tab keeps its Redis subscription until the job ends.
    """

    async def forward() -> None:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            data = json.loads(message["data"])
            await websocket.send_json(data)
            if is_final(data):
                return

    async def wait_for_disconnect() -> None:
        try:
            while (await websocket.receive())["type"] != "websocket.disconnect":
                pass
        except WebSocketDisconnect:
            pass

    tasks = [asyncio.create_task(forward()), asyncio.create_task(wait_for_disconnect())]
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    for task in done:
        task.result()
