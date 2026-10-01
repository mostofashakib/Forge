from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


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


async def stream_channel(
    websocket: WebSocket,
    *,
    redis_url: str,
    channel: str,
    finished_message: Callable[[], dict | None],
    is_final: Callable[[dict], bool],
    log_tag: str,
) -> None:
    """Accept the socket and stream a job's pub/sub channel until it finishes.

    `finished_message` runs in a thread after subscribing, so a job that ends
    between the client connecting and the subscription landing still reaches
    the client: either through the channel or as the stored final message.
    """
    import redis

    await websocket.accept()
    try:
        connection = redis.asyncio.from_url(redis_url)
        pubsub = connection.pubsub()
        await pubsub.subscribe(channel)
        logger.info("[ws:%s] subscribed to %s", log_tag, channel)
    except Exception:
        logger.exception("[ws:%s] could not connect to Redis", log_tag)
        await websocket.close(code=1011)
        return

    try:
        finished = await run_in_threadpool(finished_message)
        if finished is not None:
            await websocket.send_json(finished)
        else:
            await relay_pubsub(websocket, pubsub, is_final=is_final)
    except WebSocketDisconnect:
        logger.info("[ws:%s] client disconnected from %s", log_tag, channel)
    except Exception:
        logger.exception("[ws:%s] unexpected error on %s", log_tag, channel)
    finally:
        await pubsub.unsubscribe(channel)
        await connection.aclose()
        try:
            await websocket.close()
        except RuntimeError:
            pass  # client already closed the connection
