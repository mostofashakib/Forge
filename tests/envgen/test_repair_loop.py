from __future__ import annotations

import pytest

from forge.envgen.a2a import MessageKind
from forge.envgen.agents.base import EnvGenAgent, render_correction_context
from forge.envgen.agents.reviewer import (
    GenerationReview,
    GenerationReviewError,
    ReviewIssue,
    ReviewSeverity,
)
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.executor import TaskExecutor
from forge.envgen.planning import PromptPlannerAgent
from forge.envgen.repair import (
    CorrectionTask,
    FindingRouter,
    RepairLoop,
    RepairPlanner,
    UnrepairableFinding,
    correction_key,
    correction_tasks_for,
)
from forge.extraction.schemas import ActionDef, CompilerInput


def _issue(artifact: str | None, *, category: str = "requirements") -> ReviewIssue:
    return ReviewIssue(
        severity=ReviewSeverity.ERROR,
        category=category,
        message="something is wrong",
        artifact=artifact,
    )


class _Agent(EnvGenAgent):
    def __init__(self, agent_id: str, produces: list[str]) -> None:
        self.agent_id = agent_id
        self.produces = produces

    async def run(self, ctx, bus) -> None:  # pragma: no cover - not executed
        raise NotImplementedError


def _pipeline_agents() -> list[EnvGenAgent]:
    return [
        _Agent("backend_builder", ["backend_code"]),
        _Agent("ui_builder", ["ui_code"]),
        _Agent("app_assembler", ["app_code"]),
        _Agent("telemetry", ["instrumented_code"]),
        _Agent("state_bridge", ["state_bridge_code"]),
        _Agent("policy", ["policy_dsl"]),
        _Agent("reward", ["reward_fn_code"]),
        _Agent("correctness_reviewer", ["correctness_report"]),
        _Agent("reviewer", ["review_report"]),
    ]


class _GraphAgent(EnvGenAgent):
    """Stub carrying the real dependency wiring so the planner builds a true DAG."""

    def __init__(self, agent_id, produces, depends_on=()) -> None:
        self.agent_id = agent_id
        self.produces = list(produces)
        self.depends_on = list(depends_on)

    async def run(self, ctx, bus) -> None:  # pragma: no cover - not executed
        raise NotImplementedError


def _graph_agents() -> list[EnvGenAgent]:
    return [
        _GraphAgent("backend_builder", ["backend_code"]),
        _GraphAgent("ui_builder", ["ui_code"]),
        _GraphAgent("app_assembler", ["app_code"], ["backend_code", "ui_code"]),
        _GraphAgent("telemetry", ["instrumented_code"], ["app_code"]),
        _GraphAgent(
            "state_bridge",
            ["state_bridge_code", "state_schema_manifest"],
            ["instrumented_code"],
        ),
        _GraphAgent("policy", ["policy_dsl"]),
        _GraphAgent("reward", ["reward_fn_code"]),
        _GraphAgent(
            "correctness_reviewer",
            ["correctness_report"],
            ["app_code", "instrumented_code", "state_bridge_code", "reward_fn_code"],
        ),
        _GraphAgent(
            "reviewer",
            ["review_report"],
            [
                "app_code", "instrumented_code", "state_bridge_code",
                "state_schema_manifest", "policy_dsl", "reward_fn_code",
            ],
        ),
    ]


def _ctx() -> EnvGenContext:
    return EnvGenContext(
        env_name="tasks_env",
        description="A task tracker",
        compiler_input=CompilerInput(
            project_name="tasks_env",
            domain="task_management",
            entities=[],
            actions=[ActionDef(name="complete_task", params=[])],
            tasks=[],
        ),
    )


def _base_plan():
    return PromptPlannerAgent().create_plan(_ctx(), _graph_agents())


def _ids(plan) -> set[str]:
    return {task.id for task in plan.tasks}


def test_planner_reruns_target_and_transitive_downstream_only():
    subplan = RepairPlanner(_base_plan()).subplan({"backend_builder"})
    assert _ids(subplan) == {
        "backend_builder", "app_assembler", "telemetry", "state_bridge",
        "correctness_reviewer", "reviewer",
    }
    # Sibling branches that do not consume backend output are untouched.
    assert "ui_builder" not in _ids(subplan)
    assert "policy" not in _ids(subplan)
    assert "reward" not in _ids(subplan)


def test_planner_always_reruns_the_reviewers_for_a_leaf_target():
    subplan = RepairPlanner(_base_plan()).subplan({"reward"})
    assert _ids(subplan) == {"reward", "correctness_reviewer", "reviewer"}


def test_planner_strips_dependencies_outside_the_rerun_set():
    subplan = RepairPlanner(_base_plan()).subplan({"reward"})
    reviewer = next(t for t in subplan.tasks if t.id == "reviewer")
    # reviewer normally depends on many producers; only reward is in the set.
    assert reviewer.dependencies == ["reward"]
    reward = next(t for t in subplan.tasks if t.id == "reward")
    assert reward.dependencies == []


def test_planner_preserves_task_outputs_and_criteria():
    base = _base_plan()
    subplan = RepairPlanner(base).subplan({"reward"})
    reward_base = next(t for t in base.tasks if t.id == "reward")
    reward_sub = next(t for t in subplan.tasks if t.id == "reward")
    assert reward_sub.outputs == reward_base.outputs
    assert reward_sub.acceptance_criteria == reward_base.acceptance_criteria


@pytest.mark.asyncio
async def test_render_correction_context_returns_none_without_a_correction():
    bus = ArtifactBus()
    assert render_correction_context(bus, "reward") is None


@pytest.mark.asyncio
async def test_with_correction_appends_block_only_when_present():
    from forge.envgen.agents.base import with_correction

    bus = ArtifactBus()
    assert with_correction(bus, "reward", "PROMPT") == "PROMPT"

    await bus.publish(correction_key("reward"), {
        "findings": [{"category": "requirements", "artifact": "reward_fn_code",
                      "message": "fix the reward"}],
        "acceptance_criteria": [],
        "prior_output": None,
    })
    combined = with_correction(bus, "reward", "PROMPT")
    assert combined.startswith("PROMPT")
    assert "fix the reward" in combined


@pytest.mark.asyncio
async def test_reward_agent_folds_correction_into_its_prompt():
    from forge.envgen.agents.reward import RewardAgent

    class _FakeClient:
        def __init__(self) -> None:
            self.user = None

        def extract(self, *, system, user, schema):
            self.user = user
            return schema(content="def compute_reward(*a):\n    return 1.0\n")

    client = _FakeClient()
    ctx = _ctx()
    ctx.reward_requirements = "reward on completion"
    bus = ArtifactBus()
    await bus.publish(correction_key("reward"), {
        "findings": [{"category": "requirements", "artifact": "reward_fn_code",
                      "message": "reward ignores completion signal"}],
        "acceptance_criteria": ["Reward reflects task completion"],
        "prior_output": "def compute_reward(*a):\n    return 0.0\n",
    })

    await RewardAgent(client=client).run(ctx, bus)

    assert "reward ignores completion signal" in client.user


@pytest.mark.asyncio
async def test_render_correction_context_includes_findings_criteria_and_prior():
    bus = ArtifactBus()
    await bus.publish(correction_key("reward"), {
        "findings": [
            {"category": "requirements", "artifact": "reward_fn_code",
             "message": "reward ignores the completion signal"},
        ],
        "acceptance_criteria": ["Reward reflects task completion"],
        "prior_output": "def compute_reward(*a):\n    return 0.0\n",
    })

    rendered = render_correction_context(bus, "reward")

    assert rendered is not None
    assert "reward ignores the completion signal" in rendered
    assert "requirements" in rendered
    assert "reward_fn_code" in rendered
    assert "Reward reflects task completion" in rendered
    assert "def compute_reward" in rendered


def test_planner_grants_target_read_access_to_its_correction():
    subplan = RepairPlanner(_base_plan()).subplan({"reward"})
    reward = next(t for t in subplan.tasks if t.id == "reward")
    assert "correction:reward" in reward.context_keys
    # Downstream re-run tasks are not targets and get no correction channel.
    reviewer = next(t for t in subplan.tasks if t.id == "reviewer")
    assert not any(k.startswith("correction:") for k in reviewer.context_keys)


def test_router_maps_backend_files_to_backend_builder():
    router = FindingRouter(_pipeline_agents())
    assert router.route(_issue("main.py")) == "backend_builder"
    assert router.route(_issue("requirements.txt")) == "backend_builder"
    assert router.route(_issue("Dockerfile")) == "backend_builder"


def test_router_maps_ui_html_to_ui_builder():
    router = FindingRouter(_pipeline_agents())
    assert router.route(_issue("ui.html")) == "ui_builder"


def test_router_maps_instrumented_files_to_telemetry():
    router = FindingRouter(_pipeline_agents())
    assert router.route(_issue("instrumented:main.py")) == "telemetry"


def test_router_maps_named_artifacts_to_their_producer():
    router = FindingRouter(_pipeline_agents())
    assert router.route(_issue("state_bridge_code")) == "state_bridge"
    assert router.route(_issue("reward_fn_code")) == "reward"
    assert router.route(_issue("policy_dsl")) == "policy"
    assert router.route(_issue("backend_code")) == "backend_builder"


def test_router_returns_none_for_unmappable_findings():
    router = FindingRouter(_pipeline_agents())
    # Semantic requirement findings carry no artifact.
    assert router.route(_issue(None, category="semantic_review")) is None
    # app_code is produced by the pure-combine assembler, not a fixable source.
    assert router.route(_issue("app_code")) is None
    # An artifact nobody in this pipeline produces.
    assert router.route(_issue("mystery_artifact")) is None


def test_router_returns_none_when_responsible_agent_absent():
    # Pipeline without a telemetry specialist cannot repair instrumented code.
    agents = [a for a in _pipeline_agents() if a.agent_id != "telemetry"]
    router = FindingRouter(agents)
    assert router.route(_issue("instrumented:main.py")) is None


# --------------------------------------------------------------------------
# correction_tasks_for — typed correction tasks
# --------------------------------------------------------------------------

def test_correction_tasks_carry_target_criteria_and_source():
    plan = _base_plan()
    router = FindingRouter(_graph_agents())
    issue = _issue("reward_fn_code")

    tasks = correction_tasks_for([(issue, "review")], router, plan, round_number=1)

    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, CorrectionTask)
    assert task.target_agent_id == "reward"
    assert task.artifact == "reward_fn_code"
    assert task.source_report == "review"
    assert task.round == 1
    reward = next(t for t in plan.tasks if t.id == "reward")
    assert task.acceptance_criteria == reward.acceptance_criteria


def test_correction_tasks_raise_on_unmappable_finding():
    plan = _base_plan()
    router = FindingRouter(_graph_agents())
    with pytest.raises(UnrepairableFinding):
        correction_tasks_for(
            [(_issue(None, category="semantic_review"), "review")],
            router, plan, round_number=1,
        )


# --------------------------------------------------------------------------
# RepairLoop — end-to-end repair harness with scripted reviewers
# --------------------------------------------------------------------------

class _StubBuilder(EnvGenAgent):
    """Publishes its declared artifacts and records runs + corrections seen."""

    def __init__(self, agent_id, produces, depends_on=()) -> None:
        self.agent_id = agent_id
        self.produces = list(produces)
        self.depends_on = list(depends_on)
        self.runs = 0
        self.corrections: list = []

    async def run(self, ctx, bus) -> None:
        self.runs += 1
        self.corrections.append(render_correction_context(bus, self.agent_id))
        for artifact in self.produces:
            await bus.publish(artifact, f"{artifact}-v{self.runs}")


class _ScriptedReporter(EnvGenAgent):
    """Publishes a GenerationReview whose issues come from a per-run script."""

    def __init__(self, agent_id, report_artifact, script, depends_on) -> None:
        self.agent_id = agent_id
        self.produces = [report_artifact]
        self.depends_on = list(depends_on)
        self._report_artifact = report_artifact
        self._script = script
        self.runs = 0

    async def run(self, ctx, bus) -> None:
        issues = list(self._script(self.runs))
        self.runs += 1
        approved = not any(i.severity == ReviewSeverity.ERROR for i in issues)
        await bus.publish(
            self._report_artifact,
            GenerationReview(approved=approved, issues=issues, requirements_checked=[]),
        )


def _always_approve(_run: int) -> list[ReviewIssue]:
    return []


def _build_pipeline(review_script, correctness_script=_always_approve):
    backend = _StubBuilder("backend_builder", ["backend_code"])
    ui = _StubBuilder("ui_builder", ["ui_code"])
    assembler = _StubBuilder("app_assembler", ["app_code"], ["backend_code", "ui_code"])
    telemetry = _StubBuilder("telemetry", ["instrumented_code"], ["app_code"])
    bridge = _StubBuilder(
        "state_bridge", ["state_bridge_code", "state_schema_manifest"],
        ["instrumented_code"],
    )
    policy = _StubBuilder("policy", ["policy_dsl"])
    reward = _StubBuilder("reward", ["reward_fn_code"])
    correctness = _ScriptedReporter(
        "correctness_reviewer", "correctness_report", correctness_script,
        ["app_code", "instrumented_code", "state_bridge_code", "reward_fn_code"],
    )
    reviewer = _ScriptedReporter(
        "reviewer", "review_report", review_script,
        ["app_code", "instrumented_code", "state_bridge_code",
         "state_schema_manifest", "policy_dsl", "reward_fn_code"],
    )
    agents = [backend, ui, assembler, telemetry, bridge, policy, reward,
              correctness, reviewer]
    return agents, {a.agent_id: a for a in agents}


async def _run_initial(agents):
    ctx = _ctx()
    plan = PromptPlannerAgent().create_plan(ctx, agents)
    bus = ArtifactBus()
    executor = TaskExecutor()
    await executor.execute(plan, agents, ctx, bus)
    return plan, bus, executor, ctx


def _kinds(bus) -> list:
    return [m.kind for m in bus.protocol.history]


def _reason(bus) -> str | None:
    for m in bus.protocol.history:
        if m.kind == MessageKind.REPAIR_EXHAUSTED:
            return m.payload.get("reason")
    return None


@pytest.mark.asyncio
async def test_repair_loop_fixes_a_finding_and_approves():
    def script(run):
        return [] if run >= 1 else [_issue("reward_fn_code")]

    agents, by_id = _build_pipeline(script)
    plan, bus, executor, ctx = await _run_initial(agents)

    await RepairLoop().run(plan, agents, ctx, bus, executor)  # must not raise

    assert bus.get("review_report").approved is True
    assert by_id["reward"].runs == 2  # initial + one repair


@pytest.mark.asyncio
async def test_repair_loop_only_reruns_affected_downstream():
    def script(run):
        return [] if run >= 1 else [_issue("main.py")]  # backend finding

    agents, by_id = _build_pipeline(script)
    plan, bus, executor, ctx = await _run_initial(agents)

    await RepairLoop().run(plan, agents, ctx, bus, executor)

    # Backend and everything downstream re-ran once.
    for rerun in ("backend_builder", "app_assembler", "telemetry",
                  "state_bridge", "correctness_reviewer", "reviewer"):
        assert by_id[rerun].runs == 2, rerun
    # Sibling branches were left untouched.
    for untouched in ("ui_builder", "policy", "reward"):
        assert by_id[untouched].runs == 1, untouched


@pytest.mark.asyncio
async def test_repair_loop_targeted_builder_receives_the_finding():
    def script(run):
        return [] if run >= 1 else [_issue("reward_fn_code")]

    agents, by_id = _build_pipeline(script)
    plan, bus, executor, ctx = await _run_initial(agents)

    await RepairLoop().run(plan, agents, ctx, bus, executor)

    corrections = by_id["reward"].corrections
    assert corrections[0] is None            # initial run: no correction
    assert corrections[1] is not None        # repair run: correction present
    assert "something is wrong" in corrections[1]


@pytest.mark.asyncio
async def test_repair_loop_stops_at_retry_bound():
    # Findings differ every round (so the no-progress breaker never fires) but
    # never resolve — the retry bound must stop the loop.
    def script(run):
        return [_issue("reward_fn_code", category=f"cat{run}")]

    agents, by_id = _build_pipeline(script)
    plan, bus, executor, ctx = await _run_initial(agents)

    with pytest.raises(GenerationReviewError):
        await RepairLoop(max_repair_rounds=2).run(plan, agents, ctx, bus, executor)

    assert by_id["reward"].runs == 3  # initial + 2 repair rounds
    assert _reason(bus) == "retry_bound"


@pytest.mark.asyncio
async def test_repair_loop_circuit_breaks_on_no_progress():
    # Identical finding every round — the breaker must open before the retry bound.
    def script(_run):
        return [_issue("reward_fn_code")]

    agents, by_id = _build_pipeline(script)
    plan, bus, executor, ctx = await _run_initial(agents)

    with pytest.raises(GenerationReviewError):
        await RepairLoop(max_repair_rounds=5).run(plan, agents, ctx, bus, executor)

    assert by_id["reward"].runs == 2  # initial + exactly one repair, then breaker
    assert _reason(bus) == "no_progress"


@pytest.mark.asyncio
async def test_repair_loop_fails_fast_on_unrepairable_finding():
    def script(_run):
        return [_issue(None, category="semantic_review")]

    agents, by_id = _build_pipeline(script)
    plan, bus, executor, ctx = await _run_initial(agents)

    with pytest.raises(GenerationReviewError):
        await RepairLoop().run(plan, agents, ctx, bus, executor)

    assert by_id["reward"].runs == 1  # no repair attempted
    assert _reason(bus) == "unrepairable"


@pytest.mark.asyncio
async def test_repair_loop_records_history_in_the_protocol():
    def script(run):
        return [] if run >= 1 else [_issue("reward_fn_code")]

    agents, _ = _build_pipeline(script)
    plan, bus, executor, ctx = await _run_initial(agents)

    await RepairLoop().run(plan, agents, ctx, bus, executor)

    kinds = _kinds(bus)
    assert MessageKind.REVIEW_REJECTED in kinds
    assert MessageKind.CORRECTION_ASSIGNED in kinds
    assert MessageKind.CORRECTION_COMPLETED in kinds
    assert MessageKind.REPAIR_EXHAUSTED not in kinds  # succeeded


@pytest.mark.asyncio
async def test_repair_loop_raises_when_review_report_missing():
    agents, _ = _build_pipeline(_always_approve)
    ctx = _ctx()
    plan = PromptPlannerAgent().create_plan(ctx, agents)
    bus = ArtifactBus()  # nothing ran; no review_report
    with pytest.raises(RuntimeError, match="review report"):
        await RepairLoop().run(plan, agents, ctx, bus, TaskExecutor())
