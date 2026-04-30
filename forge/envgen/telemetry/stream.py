from __future__ import annotations
from typing import AsyncIterator
import redis


class StreamConsumer:
    def __init__(self, redis_url: str, env_name: str) -> None:
        self._redis = redis.asyncio.from_url(redis_url)
        self._key = f"forge:events:{env_name}"

    async def tail(self, last_id: str = "$") -> AsyncIterator[dict]:
        current_id = last_id
        while True:
            results = await self._redis.xread({self._key: current_id}, block=1000, count=10)
            for _, messages in (results or []):
                for msg_id, fields in messages:
                    current_id = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                    yield {
                        "id": current_id,
                        **{
                            (k.decode() if isinstance(k, bytes) else k):
                            (v.decode() if isinstance(v, bytes) else v)
                            for k, v in fields.items()
                        }
                    }

    async def read_range(self, start: str = "-", end: str = "+", count: int = 100) -> list[dict]:
        messages = await self._redis.xrange(self._key, start, end, count=count)
        return [
            {
                "id": (msg_id.decode() if isinstance(msg_id, bytes) else msg_id),
                **{
                    (k.decode() if isinstance(k, bytes) else k):
                    (v.decode() if isinstance(v, bytes) else v)
                    for k, v in fields.items()
                }
            }
            for msg_id, fields in messages
        ]

    async def close(self) -> None:
        await self._redis.aclose()
