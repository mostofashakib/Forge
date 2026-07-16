from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge.envgen.a2a import MessageKind
from forge.envgen.agents.app_generator import AppAssemblyAgent
from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.agents.reviewer import ReviewerAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.executor import TaskExecutor
from forge.envgen.error_handling import AgentExecutionError, GenerationErrorHandler
from forge.envgen.planning import AgentTask, GenerationPlan, PromptPlannerAgent
from forge.envgen.research import SpecialistResearchContext
from forge.extraction.schemas import ActionDef, CompilerInput


def _ctx() -> EnvGenContext:
    return EnvGenContext(
        env_name="tasks_env",
        description="A task tracker where users can complete tasks",
        compiler_input=CompilerInput(
            project_name="tasks_env",
            domain="task_management",
            entities=[],
            actions=[ActionDef(name="complete_task", params=[])],
            tasks=[],
        ),
    )


class _Producer(EnvGenAgent):
    agent_id = "producer"
    produces = ["source"]

    async def run(self, ctx, bus) -> None:
        await bus.publish("source", ctx.description)


class _Consumer(EnvGenAgent):
    agent_id = "consumer"
    depends_on = ["source"]
    produces = ["result"]

    async def run(self, ctx, bus) -> None:
        assert set(bus.relevant_context()) == {"source"}
        with pytest.raises(PermissionError):
            bus.get("unrelated")
        await bus.publish("result", (await bus.wait_for("source")).upper())


class _FailingAgent(EnvGenAgent):
    agent_id = "failing"
    produces = ["never"]

    async def run(self, ctx, bus) -> None:
        raise ValueError("bad specialist output")


def test_prompt_planner_creates_dependency_aware_todos():
    plan = PromptPlannerAgent().create_plan(_ctx(), [_Producer(), _Consumer()])
    consumer = next(task for task in plan.tasks if task.id == "consumer")
    assert consumer.dependencies == ["producer"]
    assert consumer.context_keys == ["source"]
    assert consumer.outputs == ["result"]
    assert plan.user_request == _ctx().description


@pytest.mark.asyncio
async def test_executor_uses_scoped_a2a_context_and_records_messages():
    agents = [_Producer(), _Consumer()]
    plan = PromptPlannerAgent().create_plan(_ctx(), agents)
    bus = ArtifactBus()

    await TaskExecutor().execute(plan, agents, _ctx(), bus)

    assert bus.get("result") == _ctx().description.upper()
    kinds = [message.kind for message in bus.protocol.history]
    assert MessageKind.TASK_ASSIGNED in kinds
    assert MessageKind.ARTIFACT_AVAILABLE in kinds
    assert MessageKind.TASK_COMPLETED in kinds


@pytest.mark.asyncio
async def test_executor_tracks_and_normalizes_agent_failures():
    handler = GenerationErrorHandler()
    agents = [_FailingAgent()]
    plan = PromptPlannerAgent().create_plan(_ctx(), agents)

    with pytest.raises(AgentExecutionError, match="bad specialist output"):
        await TaskExecutor(error_handler=handler).execute(plan, agents, _ctx(), ArtifactBus())

    assert len(handler.records) == 1
    assert handler.records[0].agent_id == "failing"
    assert handler.records[0].error_type == "ValueError"


def test_generation_plan_rejects_dependency_cycles():
    with pytest.raises(ValidationError, match="dependency cycle"):
        GenerationPlan(
            user_request="cycle",
            tasks=[
                AgentTask(id="a", agent_id="a", description="a", dependencies=["b"]),
                AgentTask(id="b", agent_id="b", description="b", dependencies=["a"]),
            ],
        )


@pytest.mark.asyncio
async def test_app_assembler_keeps_backend_and_ui_separate():
    bus = ArtifactBus()
    await bus.publish("backend_code", {"main.py": "app = object()"})
    await bus.publish("ui_code", {"ui.html": "<html></html>"})
    await AppAssemblyAgent().run(_ctx(), bus)
    assert bus.get("app_code") == {
        "main.py": "app = object()",
        "ui.html": "<html></html>",
    }


async def _review_bus(main_py: str) -> ArtifactBus:
    bus = ArtifactBus()
    app_code = {
        "main.py": main_py,
        "ui.html": "<html><body><script>complete_task()</script></body></html>",
        "requirements.txt": "fastapi\n",
        "Dockerfile": "FROM python:3.12-slim\n",
    }
    await bus.publish("app_code", app_code)
    await bus.publish("instrumented_code", {"main.py": main_py})
    await bus.publish("state_bridge_code", "class ContainerForgeEnv:\n    pass\n")
    await bus.publish("state_schema_manifest", {"fields": {}})
    await bus.publish("policy_dsl", "policies: []\n")
    await bus.publish("reward_fn_code", "def compute_reward(*args):\n    return 0.0\n")
    await bus.publish("reviewer_research", SpecialistResearchContext(
        role="reviewer",
        product_summary="A task tracker",
    ))
    return bus


@pytest.mark.asyncio
async def test_reviewer_approves_complete_generation():
    endpoints = " ".join((
        "/forge/health", "/forge/state", "/forge/reset", "/forge/snapshot",
        "/forge/restore", "/forge/restore-state", "complete_task",
    ))
    bus = await _review_bus(f"ROUTES = {endpoints!r}\n")
    await ReviewerAgent(semantic_review=False).run(_ctx(), bus)
    assert bus.get("review_report").approved is True


@pytest.mark.asyncio
async def test_reviewer_rejects_syntax_and_requirement_failures():
    bus = await _review_bus("def broken(:\n")
    await ReviewerAgent(semantic_review=False).run(_ctx(), bus)
    review = bus.get("review_report")
    assert review.approved is False
    assert {issue.category for issue in review.issues} >= {"syntax", "requirements"}
