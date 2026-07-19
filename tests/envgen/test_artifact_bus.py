import asyncio
import pytest
from forge.envgen.artifact_bus import ArtifactBus


@pytest.mark.asyncio
async def test_publish_then_wait_for_returns_value():
    bus = ArtifactBus()
    await bus.publish("app_code", {"main.py": "# code"})
    result = await bus.wait_for("app_code")
    assert result == {"main.py": "# code"}


@pytest.mark.asyncio
async def test_wait_for_unblocks_when_published():
    bus = ArtifactBus()
    received = []

    async def waiter():
        val = await bus.wait_for("x")
        received.append(val)

    async def publisher():
        await asyncio.sleep(0.01)
        await bus.publish("x", 42)

    await asyncio.gather(waiter(), publisher())
    assert received == [42]


@pytest.mark.asyncio
async def test_on_publish_callback_is_called():
    bus = ArtifactBus()
    calls = []

    async def cb(name, value):
        calls.append((name, value))

    bus.on_publish(cb)
    await bus.publish("policy_dsl", "content")
    assert calls == [("policy_dsl", "content")]


@pytest.mark.asyncio
async def test_get_returns_value_or_default():
    bus = ArtifactBus()
    await bus.publish("reward_fn_code", "fn")
    assert bus.get("reward_fn_code") == "fn"
    assert bus.get("missing", "default") == "default"


@pytest.mark.asyncio
async def test_invalidate_clears_value_and_reblocks_wait_for():
    bus = ArtifactBus()
    await bus.publish("app_code", {"main.py": "v1"})

    bus.invalidate(["app_code"])
    assert bus.get("app_code") is None

    received: list = []

    async def waiter():
        received.append(await bus.wait_for("app_code"))

    async def republisher():
        await asyncio.sleep(0.01)
        await bus.publish("app_code", {"main.py": "v2"})

    await asyncio.gather(waiter(), republisher())
    assert received == [{"main.py": "v2"}]


@pytest.mark.asyncio
async def test_invalidate_ignores_unknown_names():
    bus = ArtifactBus()
    bus.invalidate(["never_published"])  # must not raise
    assert bus.get("never_published") is None
