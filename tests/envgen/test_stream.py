import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_stream_consumer_read_range_returns_decoded_events():
    from forge.envgen.telemetry.stream import StreamConsumer

    mock_redis = AsyncMock()
    mock_redis.xrange.return_value = [
        (b"1234-0", {b"timestamp": b"2026-01-01T00:00:00", b"actor": b"user", b"action_type": b"close_ticket", b"payload": b"{}", b"state_before": b"{}", b"state_after": b"{}"}),
    ]

    with patch("forge.envgen.telemetry.stream.redis") as mock_redis_module:
        mock_redis_module.asyncio.from_url.return_value = mock_redis
        consumer = StreamConsumer(redis_url="redis://localhost:6379", env_name="test_env")
        events = await consumer.read_range()

    assert len(events) == 1
    assert events[0]["actor"] == "user"
    assert events[0]["action_type"] == "close_ticket"
    assert events[0]["id"] == "1234-0"


@pytest.mark.asyncio
async def test_read_range_with_no_entries_returns_empty():
    # Boundary: an empty stream must decode to no events, not a phantom entry.
    from forge.envgen.telemetry.stream import StreamConsumer

    mock_redis = AsyncMock()
    mock_redis.xrange.return_value = []

    with patch("forge.envgen.telemetry.stream.redis") as mock_redis_module:
        mock_redis_module.asyncio.from_url.return_value = mock_redis
        consumer = StreamConsumer(redis_url="redis://localhost:6379", env_name="test_env")
        events = await consumer.read_range()

    assert events == []
