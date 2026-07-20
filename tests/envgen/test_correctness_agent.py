import pytest

from forge.envgen.agents.correctness import audit_determinism, EnvironmentCorrectnessAgent
from forge.envgen.agents.reviewer import ReviewSeverity
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.extraction.schemas import ActionDef, CompilerInput


_COMPLIANT_MAIN = '''
_FORGE_CLOCK = 0
_ID_COUNTERS: dict[str, int] = {}


def forge_now() -> int:
    global _FORGE_CLOCK
    v = _FORGE_CLOCK
    _FORGE_CLOCK += 1
    return v


def _next_id(entity: str) -> int:
    _ID_COUNTERS[entity] = _ID_COUNTERS.get(entity, 0) + 1
    return _ID_COUNTERS[entity]


def seed():
    return {"id": _next_id("todo"), "created_at": forge_now()}


def forge_reset():
    global _FORGE_CLOCK
    _FORGE_CLOCK = 0
    _ID_COUNTERS.clear()
    seed()


class EnvState:
    """Centralized state store satisfying the authoring contract."""

    def seed_state(self, seed_value: int) -> None:
        seed()

    def reset_state(self) -> None:
        forge_reset()
'''


def _categories(issues):
    return {i.category for i in issues}


def test_clean_contract_app_has_no_findings():
    assert audit_determinism({"main.py": _COMPLIANT_MAIN}) == []


def test_wall_clock_is_flagged():
    src = _COMPLIANT_MAIN + "\ndef bad():\n    return datetime.utcnow()\n"
    issues = audit_determinism({"main.py": src})
    assert "wall_clock" in _categories(issues)
    assert all(i.severity == ReviewSeverity.ERROR for i in issues)


def test_uuid_is_flagged():
    src = _COMPLIANT_MAIN + "\ndef bad():\n    return uuid.uuid4()\n"
    assert "nondeterministic_id" in _categories(audit_determinism({"main.py": src}))


def test_unseeded_random_is_flagged():
    src = _COMPLIANT_MAIN + "\ndef bad():\n    return random.random()\n"
    assert "unseeded_randomness" in _categories(audit_determinism({"main.py": src}))


def test_seeded_random_is_allowed():
    src = _COMPLIANT_MAIN + "\nrandom.seed(0)\n\ndef ok():\n    return random.random()\n"
    assert "unseeded_randomness" not in _categories(audit_determinism({"main.py": src}))


def test_telemetry_envelope_timestamp_is_exempt():
    src = _COMPLIANT_MAIN + (
        "\ndef emit_event(name):\n"
        "    return {'ts': datetime.utcnow().isoformat(), 'name': name}\n"
    )
    assert "wall_clock" not in _categories(audit_determinism({"main.py": src}))


def test_missing_contract_is_flagged():
    src = "def seed():\n    return {'id': 1}\n"
    assert "contract_missing" in _categories(audit_determinism({"main.py": src}))


def test_reset_without_reinitialization_is_flagged():
    src = _COMPLIANT_MAIN.replace(
        "    global _FORGE_CLOCK\n    _FORGE_CLOCK = 0\n    _ID_COUNTERS.clear()\n", ""
    )
    assert "reset_not_reinitialized" in _categories(audit_determinism({"main.py": src}))


def _ctx() -> EnvGenContext:
    return EnvGenContext(
        env_name="todo_env",
        description="todo tracker",
        compiler_input=CompilerInput(
            project_name="todo_env", domain="todo",
            entities=[], actions=[ActionDef(name="add_todo", params=[])], tasks=[],
        ),
    )


@pytest.mark.asyncio
async def test_agent_publishes_approved_report_for_clean_app():
    bus = ArtifactBus()
    await bus.publish("app_code", {"main.py": _COMPLIANT_MAIN, "ui.html": "<html></html>"})
    await bus.publish("instrumented_code", {"main.py": _COMPLIANT_MAIN})
    await bus.publish("state_bridge_code", "class E:\n    pass\n")
    await bus.publish("reward_fn_code", "def compute_reward(*a):\n    return 0.0\n")
    await EnvironmentCorrectnessAgent().run(_ctx(), bus)
    report = bus.get("correctness_report")
    assert report.approved is True


@pytest.mark.asyncio
async def test_agent_rejects_wall_clock_app():
    bad = _COMPLIANT_MAIN + "\ndef bad():\n    return datetime.utcnow()\n"
    bus = ArtifactBus()
    await bus.publish("app_code", {"main.py": bad, "ui.html": "<html></html>"})
    await bus.publish("instrumented_code", {"main.py": bad})
    await bus.publish("state_bridge_code", "class E:\n    pass\n")
    await bus.publish("reward_fn_code", "def compute_reward(*a):\n    return 0.0\n")
    await EnvironmentCorrectnessAgent().run(_ctx(), bus)
    report = bus.get("correctness_report")
    assert report.approved is False
    assert "wall_clock" in {i.category for i in report.issues}


@pytest.mark.asyncio
async def test_agent_rejects_app_without_a_state_class():
    # _COMPLIANT_MAIN minus the EnvState class — determinism is fine, but the
    # authoring contract (centralized state class) is not satisfied.
    without_class = _COMPLIANT_MAIN.split("class EnvState:")[0]
    bus = ArtifactBus()
    await bus.publish("app_code", {"main.py": without_class, "ui.html": "<html></html>"})
    await bus.publish("instrumented_code", {"main.py": without_class})
    await bus.publish("state_bridge_code", "class E:\n    pass\n")
    await bus.publish("reward_fn_code", "def compute_reward(*a):\n    return 0.0\n")
    await EnvironmentCorrectnessAgent().run(_ctx(), bus)
    report = bus.get("correctness_report")
    assert report.approved is False
    assert "state_class_missing" in {i.category for i in report.issues}


@pytest.mark.asyncio
async def test_agent_rejects_app_with_a_string_returning_endpoint():
    bad = _COMPLIANT_MAIN + (
        "\nfrom fastapi import FastAPI\n"
        "app = FastAPI()\n\n"
        "@app.post('/close_ticket')\n"
        "def close_ticket():\n"
        "    return 'closed'\n"
    )
    bus = ArtifactBus()
    await bus.publish("app_code", {"main.py": bad, "ui.html": "<html></html>"})
    await bus.publish("instrumented_code", {"main.py": bad})
    await bus.publish("state_bridge_code", "class E:\n    pass\n")
    await bus.publish("reward_fn_code", "def compute_reward(*a):\n    return 0.0\n")
    await EnvironmentCorrectnessAgent().run(_ctx(), bus)
    report = bus.get("correctness_report")
    assert report.approved is False
    assert "untyped_return" in {i.category for i in report.issues}
