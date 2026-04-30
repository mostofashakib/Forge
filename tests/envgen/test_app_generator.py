import pytest
from forge.envgen.agents.app_generator import AppGeneratorAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.schemas import GeneratedApp, FileContent
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.schemas import CompilerInput, EntityDef, FieldDef, ActionDef, TaskTemplate, SuccessCondition


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


@pytest.mark.asyncio
async def test_app_generator_publishes_app_code():
    mock = MockLLMClient({
        "GeneratedApp": GeneratedApp(files=[
            FileContent(path="main.py", content="from fastapi import FastAPI\napp = FastAPI()\n@app.get('/forge/health')\ndef health(): return {'status': 'ok'}"),
        ])
    })
    agent = AppGeneratorAgent(client=mock)
    bus = ArtifactBus()
    await agent.run(_ctx(), bus)
    assert bus.get("app_code") is not None
    assert "main.py" in bus.get("app_code")


@pytest.mark.asyncio
async def test_app_generator_has_no_dependencies():
    agent = AppGeneratorAgent(client=MockLLMClient({
        "GeneratedApp": GeneratedApp(files=[FileContent(path="main.py", content="# app")])
    }))
    assert agent.depends_on == []
    assert agent.produces == "app_code"
