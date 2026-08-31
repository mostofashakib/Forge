from __future__ import annotations

import asyncio

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


def _ui_ctx() -> EnvGenContext:
    ctx = _ctx()
    ctx.with_ui = True
    return ctx


class _BackendStub(EnvGenAgent):
    agent_id = "backend_stub"
    produces = ["backend_code"]

    async def run(self, ctx, bus) -> None:
        await bus.publish("backend_code", {"main.py": "app = object()"})


class _UIStub(EnvGenAgent):
    agent_id = "ui_stub"
    produces = ["ui_code"]

    async def run(self, ctx, bus) -> None:
        await bus.publish("ui_code", {"ui.html": "<html></html>"})


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


class _OptionalConsumer(EnvGenAgent):
    agent_id = "optional_consumer"
    optional_depends_on = ["source"]
    produces = ["result"]

    async def run(self, ctx, bus) -> None:
        await bus.publish("result", bus.get("source", ctx.description))


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


def test_prompt_planner_only_scopes_optional_context_when_produced():
    without_source = PromptPlannerAgent().create_plan(_ctx(), [_OptionalConsumer()])
    task = without_source.tasks[0]
    assert task.dependencies == []
    assert task.context_keys == ["source"]

    with_source = PromptPlannerAgent().create_plan(
        _ctx(), [_Producer(), _OptionalConsumer()]
    )
    task = next(item for item in with_source.tasks if item.id == "optional_consumer")
    assert task.dependencies == ["producer"]
    assert task.context_keys == ["source"]


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
async def test_executor_runs_selfcontained_subplan_against_prepopulated_bus():
    # Simulates a repair re-run: only the consumer is re-executed, with its
    # external producer dependency already satisfied on the bus and stripped
    # from the sub-plan (as RepairPlanner produces it).
    agents = [_Producer(), _Consumer()]
    bus = ArtifactBus()
    await bus.publish("source", "already-here")

    subplan = GenerationPlan(
        user_request="repair",
        tasks=[AgentTask(
            id="consumer",
            agent_id="consumer",
            description="consumer",
            dependencies=[],  # external "producer" dep stripped by the planner
            context_keys=["source"],
            outputs=["result"],
        )],
    )

    await TaskExecutor().execute(subplan, agents, _ctx(), bus)

    assert bus.get("result") == "ALREADY-HERE"


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
    await AppAssemblyAgent().run(_ui_ctx(), bus)
    assert bus.get("app_code") == {
        "main.py": "app = object()",
        "ui.html": "<html></html>",
    }


@pytest.mark.asyncio
async def test_app_assembler_emits_backend_only_for_a_headless_environment():
    bus = ArtifactBus()
    await bus.publish("backend_code", {"main.py": "app = object()"})

    # A headless run has no ui_builder, so ui_code is never published. The
    # assembler must not block on an artifact that will never arrive.
    await asyncio.wait_for(AppAssemblyAgent().run(_ctx(), bus), timeout=2)

    assert bus.get("app_code") == {"main.py": "app = object()"}


@pytest.mark.asyncio
async def test_planner_builds_a_valid_graph_when_no_ui_builder_is_present():
    # ui_code has no producer in a headless pipeline; the assembler must
    # tolerate that rather than failing the plan for a missing dependency.
    agents = [_BackendStub(), AppAssemblyAgent()]

    plan = PromptPlannerAgent().create_plan(_ctx(), agents)

    assembler = next(task for task in plan.tasks if task.id == "app_assembler")
    assert assembler.dependencies == ["backend_stub"]
    assert "ui_code" in assembler.context_keys


def test_planner_wires_the_ui_builder_in_when_one_is_present():
    # False-positive guard: making ui_code optional must not sever the real
    # dependency edge when a UI builder is in the pipeline.
    agents = [_BackendStub(), _UIStub(), AppAssemblyAgent()]

    plan = PromptPlannerAgent().create_plan(_ui_ctx(), agents)

    assembler = next(task for task in plan.tasks if task.id == "app_assembler")
    assert set(assembler.dependencies) == {"backend_stub", "ui_stub"}


async def _review_bus(main_py: str, *, include_research: bool = True) -> ArtifactBus:
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
    if include_research:
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
async def test_reviewer_does_not_require_research_context():
    endpoints = " ".join((
        "/forge/health", "/forge/state", "/forge/reset", "/forge/snapshot",
        "/forge/restore", "/forge/restore-state", "complete_task",
    ))
    bus = await _review_bus(f"ROUTES = {endpoints!r}\n", include_research=False)

    await ReviewerAgent(semantic_review=False).run(_ctx(), bus)

    assert bus.get("review_report").approved is True


@pytest.mark.asyncio
async def test_reviewer_rejects_syntax_and_requirement_failures():
    bus = await _review_bus("def broken(:\n")
    await ReviewerAgent(semantic_review=False).run(_ctx(), bus)
    review = bus.get("review_report")
    assert review.approved is False
    assert {issue.category for issue in review.issues} >= {"syntax", "requirements"}


async def _headless_review_bus(main_py: str) -> ArtifactBus:
    """Artifacts a headless generation produces — no ui.html anywhere."""
    bus = ArtifactBus()
    await bus.publish("app_code", {
        "main.py": main_py,
        "requirements.txt": "fastapi\n",
        "Dockerfile": "FROM python:3.12-slim\n",
    })
    await bus.publish("instrumented_code", {"main.py": main_py})
    await bus.publish("state_bridge_code", "class ContainerForgeEnv:\n    pass\n")
    await bus.publish("state_schema_manifest", {"fields": {}})
    await bus.publish("policy_dsl", "policies: []\n")
    await bus.publish("reward_fn_code", "def compute_reward(*args):\n    return 0.0\n")
    return bus


_COMPLETE_ROUTES = " ".join((
    "/forge/health", "/forge/state", "/forge/reset", "/forge/snapshot",
    "/forge/restore", "/forge/restore-state", "complete_task",
))


@pytest.mark.asyncio
async def test_reviewer_approves_a_headless_generation_with_no_ui_html():
    bus = await _headless_review_bus(f"ROUTES = {_COMPLETE_ROUTES!r}\n")

    await ReviewerAgent(semantic_review=False).run(_ctx(), bus)

    assert bus.get("review_report").approved is True


@pytest.mark.asyncio
async def test_reviewer_rejects_a_ui_generation_missing_its_ui_html():
    # False-positive guard: relaxing the requirement for headless runs must not
    # relax it for runs that asked for a UI.
    bus = await _headless_review_bus(f"ROUTES = {_COMPLETE_ROUTES!r}\n")

    await ReviewerAgent(semantic_review=False).run(_ui_ctx(), bus)

    review = bus.get("review_report")
    assert review.approved is False
    assert any(issue.artifact == "ui.html" for issue in review.issues)


@pytest.mark.asyncio
async def test_reviewer_still_requires_the_backend_when_headless():
    # Negative: dropping the UI requirement must not drop the others with it.
    bus = ArtifactBus()
    await bus.publish("app_code", {
        "requirements.txt": "fastapi\n",
        "Dockerfile": "FROM python:3.12-slim\n",
    })
    await bus.publish("instrumented_code", {})
    await bus.publish("state_bridge_code", "class ContainerForgeEnv:\n    pass\n")
    await bus.publish("state_schema_manifest", {"fields": {}})
    await bus.publish("policy_dsl", "policies: []\n")
    await bus.publish("reward_fn_code", "def compute_reward(*args):\n    return 0.0\n")

    await ReviewerAgent(semantic_review=False).run(_ctx(), bus)

    review = bus.get("review_report")
    assert review.approved is False
    assert any(issue.artifact == "main.py" for issue in review.issues)


# ---------------------------------------------------------------------------
# Semantic review independence
# ---------------------------------------------------------------------------

def test_reviewer_semantic_judge_is_not_the_generation_client(monkeypatch):
    """The model that wrote the artifacts must not be the one approving them."""
    from forge.envgen.agents.reviewer import ReviewerAgent

    monkeypatch.setenv("FORGE_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("FORGE_LLM_MODEL", "gemma4:26b")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "gemma4:26b")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "llama3.1:8b")
    monkeypatch.delenv("FORGE_QUORUM_MODELS", raising=False)

    agent = ReviewerAgent()

    assert agent._client._model == "llama3.1:8b"
    assert agent._client._model != "gemma4:26b"


def test_reviewer_builds_a_quorum_panel_when_one_is_configured(monkeypatch):
    from forge.envgen.agents.reviewer import ReviewerAgent

    monkeypatch.setenv("FORGE_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("FORGE_QUORUM_MODELS", "openai:gpt-4o,gemini:gemini-2.0-flash")

    agent = ReviewerAgent()

    assert agent._panel is not None
    assert agent._panel._jury is not None


def test_reviewer_without_a_quorum_uses_a_single_judge(monkeypatch):
    from forge.envgen.agents.reviewer import ReviewerAgent

    monkeypatch.delenv("FORGE_QUORUM_MODELS", raising=False)

    agent = ReviewerAgent()

    assert agent._panel is not None
    assert agent._panel._jury is None


def test_reviewer_with_semantic_review_disabled_builds_no_panel(monkeypatch):
    from forge.envgen.agents.reviewer import ReviewerAgent

    monkeypatch.setenv("FORGE_QUORUM_MODELS", "openai:gpt-4o")

    assert ReviewerAgent(semantic_review=False)._panel is None


def test_reviewer_refuses_a_quorum_drawn_from_the_generating_family(monkeypatch):
    from forge.grading_provenance import GraderContaminationError
    from forge.envgen.agents.reviewer import ReviewerAgent

    monkeypatch.setenv("FORGE_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-4-6")
    monkeypatch.setenv("FORGE_QUORUM_MODELS", "anthropic:claude-haiku-4-5-20251001")

    with pytest.raises(GraderContaminationError):
        ReviewerAgent()


def _complete_routes() -> str:
    endpoints = " ".join((
        "/forge/health", "/forge/state", "/forge/reset", "/forge/snapshot",
        "/forge/restore", "/forge/restore-state", "complete_task",
    ))
    return f"ROUTES = {endpoints!r}\n"


class _PanelStub:
    def __init__(self, requirements_met, findings):
        from forge.envgen.agents.semantic_review import PanelResult

        self._result = PanelResult(requirements_met=requirements_met, findings=findings)

    def assess(self, semantic_input):
        return self._result


@pytest.mark.asyncio
async def test_semantic_panel_agreeing_requirements_are_met_approves():
    bus = await _review_bus(_complete_routes())
    agent = ReviewerAgent(semantic_review=False)
    agent._semantic_review = True
    agent._panel = _PanelStub(True, ["cosmetic nit"])

    await agent.run(_ctx(), bus)

    review = bus.get("review_report")
    assert review.approved is True


@pytest.mark.asyncio
async def test_semantic_panel_agreeing_requirements_are_unmet_blocks():
    bus = await _review_bus(_complete_routes())
    agent = ReviewerAgent(semantic_review=False)
    agent._semantic_review = True
    agent._panel = _PanelStub(False, ["reward never fires"])

    await agent.run(_ctx(), bus)

    review = bus.get("review_report")
    assert review.approved is False
    assert any(issue.category == "semantic_review" for issue in review.issues)


@pytest.mark.asyncio
async def test_a_contested_semantic_panel_blocks_rather_than_approving():
    """A split panel is not approval — it routes into repair with the dissent."""
    bus = await _review_bus(_complete_routes())
    agent = ReviewerAgent(semantic_review=False)
    agent._semantic_review = True
    agent._panel = _PanelStub(None, ["[gpt-4o] ok", "[gemini] state schema is wrong"])

    await agent.run(_ctx(), bus)

    review = bus.get("review_report")
    assert review.approved is False
    messages = " ".join(issue.message for issue in review.issues)
    assert "state schema is wrong" in messages


@pytest.mark.asyncio
async def test_a_contested_panel_is_labelled_distinctly_from_a_clear_rejection():
    """The record must distinguish 'reviewers disagreed' from 'reviewers rejected'."""
    bus = await _review_bus(_complete_routes())
    agent = ReviewerAgent(semantic_review=False)
    agent._semantic_review = True
    agent._panel = _PanelStub(None, ["[gemini] state schema is wrong"])

    await agent.run(_ctx(), bus)

    categories = {issue.category for issue in bus.get("review_report").issues}
    assert "semantic_review_contested" in categories
    assert "semantic_review" not in categories


def test_reviewer_does_not_build_a_judge_client_when_a_quorum_is_configured(monkeypatch):
    """A quorum-only setup must not require the single judge's credentials.

    get_client eagerly constructs the provider SDK, so building an unused judge
    would make an Anthropic API key mandatory even when every quorum member
    comes from another provider.
    """
    import forge.envgen.agents.reviewer as reviewer_module

    def _explode(*args, **kwargs):
        raise AssertionError("judge client must not be built when a quorum exists")

    monkeypatch.setattr(reviewer_module, "get_judge_client", _explode)
    monkeypatch.setenv("FORGE_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("FORGE_QUORUM_MODELS", "openai:gpt-4o,gemini:gemini-2.0-flash")

    agent = reviewer_module.ReviewerAgent()

    assert agent._panel is not None
    assert agent._client is None


def test_reviewer_still_builds_a_judge_client_without_a_quorum(monkeypatch):
    """False-positive guard: the single-judge path must keep its client."""
    import forge.envgen.agents.reviewer as reviewer_module

    monkeypatch.delenv("FORGE_QUORUM_MODELS", raising=False)
    monkeypatch.setenv("FORGE_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "llama3.1:8b")

    assert reviewer_module.ReviewerAgent()._client is not None
