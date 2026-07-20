import pytest
from forge.envgen.agents.app_generator import AppGeneratorAgent, AppGeneratorPrompts
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.schemas import AppPlan, FilePlan, GeneratedFile
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.schemas import (
    CompilerInput, EntityDef, FieldDef, ActionDef, TaskTemplate, SuccessCondition,
)


def _ctx() -> EnvGenContext:
    return EnvGenContext(
        env_name="ticket_env",
        description="A support ticket system",
        compiler_input=CompilerInput(
            project_name="ticket_env",
            domain="support",
            entities=[EntityDef(name="ticket", fields=[
                FieldDef(name="id", type="string"),
                FieldDef(name="status", type="enum", values=["open", "closed"]),
            ])],
            actions=[ActionDef(name="close_ticket", params=[])],
            tasks=[TaskTemplate(
                name="resolve",
                description="Resolve ticket",
                success_conditions=[SuccessCondition(type="state_check", expression="ticket.status=='closed'")],
            )],
        ),
    )


def _mock_client(files: list[tuple[str, str]]) -> MockLLMClient:
    """
    Build a MockLLMClient that handles both LLM calls the AppGeneratorAgent makes:
    - Phase 1: extract(AppPlan) → returns a plan with the given file paths
    - Phase 2: extract(GeneratedFile) → returns the matching file content

    MockLLMClient keys on schema.__name__, so both calls share the same key space.
    We use MockRetryClient override: since the same key is looked up multiple times
    for GeneratedFile (once per file), we supply the last GeneratedFile response only,
    which the mock returns for every call with that key.
    """
    plan = AppPlan(files=[FilePlan(path=p, description=f"file {p}") for p, _ in files])
    # For simplicity, return the same GeneratedFile for all files in phase 2.
    generated = GeneratedFile(content="from fastapi import FastAPI\napp = FastAPI()\n@app.get('/forge/health')\ndef health(): return {'status': 'ok'}")
    return MockLLMClient({"AppPlan": plan, "GeneratedFile": generated})


@pytest.mark.asyncio
async def test_app_generator_publishes_app_code():
    client = _mock_client([("main.py", "# main"), ("models.py", "# models")])
    agent = AppGeneratorAgent(client=client)
    bus = ArtifactBus()
    await agent.run(_ctx(), bus)

    result = bus.get("app_code")
    assert result is not None
    assert "main.py" in result
    assert "models.py" in result


@pytest.mark.asyncio
async def test_app_generator_has_no_dependencies():
    agent = AppGeneratorAgent(client=_mock_client([("main.py", "# app")]))
    assert agent.depends_on == []
    assert agent.produces == ["app_code"]


@pytest.mark.asyncio
async def test_app_generator_files_contain_generated_content():
    content = "from fastapi import FastAPI\napp = FastAPI()"
    client = MockLLMClient({
        "AppPlan": AppPlan(files=[FilePlan(path="main.py", description="entry point")]),
        "GeneratedFile": GeneratedFile(content=content),
    })
    agent = AppGeneratorAgent(client=client)
    bus = ArtifactBus()
    await agent.run(_ctx(), bus)

    files = bus.get("app_code")
    assert files["main.py"] == content


def test_backend_prompt_mandates_determinism_contract():
    prompt = AppGeneratorPrompts.BACKEND
    # Counter-based logical clock
    assert "_FORGE_CLOCK" in prompt
    assert "forge_now()" in prompt
    # Sequential identifiers
    assert "_ID_COUNTERS" in prompt
    assert "_next_id(" in prompt
    # Reset must re-initialize both counters
    assert "reset" in prompt.lower()
    assert "re-initialize" in prompt.lower() or "reinitialize" in prompt.lower()
    # Wall-clock / random ids are banned
    assert "utcnow" in prompt.lower()
    assert "uuid" in prompt.lower()


def test_backend_prompt_mandates_state_management_class():
    prompt = AppGeneratorPrompts.BACKEND
    # A single centralized state class with the two contract methods.
    assert "reset_state" in prompt
    assert "seed_state" in prompt
    assert "seed_state(self, seed" in prompt  # seed-driven, reproducible
    # Reset delegates to the class rather than inlining query logic per endpoint.
    assert "STATE.reset_state()" in prompt


def test_backend_prompt_mandates_seeded_reset():
    prompt = AppGeneratorPrompts.BACKEND
    lower = prompt.lower()
    # /forge/reset accepts an optional seed in the request body.
    assert '"seed"' in prompt or "'seed'" in prompt
    # A seeded reset delegates to STATE.seed_state(seed); unseeded → baseline.
    assert "STATE.seed_state(seed)" in prompt
    # seed_state must draw from a seeded RNG so distinct seeds diverge but the
    # same seed reproduces.
    assert "random.Random(seed)" in prompt
    assert "same seed" in lower and "different seed" in lower


def test_backend_prompt_mandates_typed_dict_returns():
    prompt = AppGeneratorPrompts.BACKEND.lower()
    assert "typed" in prompt
    # Both success and error paths must be dicts, never bare strings.
    assert "never a bare string" in prompt
    assert '"error"' in prompt or "'error'" in prompt
