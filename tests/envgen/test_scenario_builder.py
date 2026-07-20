"""Realistic initial-state scenario specialist (TASKS.md #4).

The specialist first grounds itself in how the *real* product works — the
research brief from the user-researcher, or the user's own description — then
emits diverse, seed-selectable scenarios carrying distractors, ordering-sensitive
sequences, and conflicting information. ``assess_scenario_suite`` is the
deterministic quality gate that rejects degenerate suites.
"""

from __future__ import annotations

import pytest

from forge.envgen.agents.scenario_builder import (
    Scenario,
    ScenarioBuilderAgent,
    ScenarioBuilderPrompts,
    ScenarioSuite,
    assess_scenario_suite,
    normalize_seeds,
    scenario_for_seed,
)
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.research import SpecialistResearchContext
from forge.extraction.schemas import ActionDef, CompilerInput


# ── Fixtures ──────────────────────────────────────────────────────────────


def _ctx(description: str = "A CRM for tracking sales leads") -> EnvGenContext:
    return EnvGenContext(
        env_name="crm_env",
        description=description,
        compiler_input=CompilerInput(
            project_name="crm_env",
            domain="crm",
            entities=[],
            actions=[
                ActionDef(name="convert_lead", params=[]),
                ActionDef(name="archive_lead", params=[]),
            ],
            tasks=[],
        ),
    )


def _healthy_suite() -> ScenarioSuite:
    return ScenarioSuite(
        env_name="crm_env",
        scenarios=[
            Scenario(
                scenario_id="hot_lead_amid_noise",
                title="One hot lead among distractors",
                seed=1,
                objective="Convert the qualified lead and ignore the rest",
                records={
                    "lead": [
                        {"id": "l1", "status": "qualified", "score": 92},
                        {"id": "l2", "status": "cold", "score": 5},
                        {"id": "l3", "status": "spam", "score": 0},
                    ]
                },
                distractors=["l2", "l3"],
                conflicts=["l1 marked qualified but its last note says 'not interested'"],
                required_actions=["convert_lead"],
                forbidden_actions=["archive_lead"],
                expected_answer="Converted l1",
            ),
            Scenario(
                scenario_id="order_matters",
                title="Archive only after conversion is confirmed",
                seed=2,
                objective="Convert then archive in the correct order",
                records={
                    "lead": [
                        {"id": "l4", "status": "qualified", "score": 70},
                        {"id": "l5", "status": "duplicate", "score": 70},
                    ]
                },
                distractors=["l5"],
                ordering_sensitive=True,
                ordering_note="Archiving before converting loses the sale credit",
                required_actions=["convert_lead", "archive_lead"],
                expected_answer="Converted then archived l4",
            ),
        ],
    )


# ── assess_scenario_suite: happy path ─────────────────────────────────────


def test_healthy_suite_has_no_issues():
    assert assess_scenario_suite(_healthy_suite()) == []


# ── assess_scenario_suite: negative / false-positive guards ───────────────


def test_degenerate_single_record_no_distractor_scenario_is_flagged():
    suite = ScenarioSuite(
        env_name="crm_env",
        scenarios=[
            Scenario(
                scenario_id="trivial",
                title="One lead, nothing to reason about",
                seed=1,
                objective="Convert the lead",
                records={"lead": [{"id": "l1", "status": "qualified"}]},
            )
        ],
    )
    categories = {i.category for i in assess_scenario_suite(suite)}
    assert "degenerate_scenario" in categories
    assert "insufficient_diversity" in categories  # a single scenario is not a suite


def test_duplicate_seeds_are_flagged():
    suite = _healthy_suite()
    suite.scenarios[1].seed = suite.scenarios[0].seed
    assert "duplicate_seed" in {i.category for i in assess_scenario_suite(suite)}


def test_suite_without_any_distractors_is_flagged():
    suite = _healthy_suite()
    for s in suite.scenarios:
        s.distractors = []
    assert "missing_distractors" in {i.category for i in assess_scenario_suite(suite)}


def test_suite_without_any_conflicting_info_is_flagged():
    suite = _healthy_suite()
    for s in suite.scenarios:
        s.conflicts = []
    assert "missing_conflicts" in {i.category for i in assess_scenario_suite(suite)}


def test_suite_without_any_ordering_sensitive_scenario_is_flagged():
    suite = _healthy_suite()
    for s in suite.scenarios:
        s.ordering_sensitive = False
    assert "missing_ordering" in {i.category for i in assess_scenario_suite(suite)}


def test_unknown_action_reference_is_flagged_but_known_actions_are_not():
    suite = _healthy_suite()
    suite.scenarios[0].required_actions = ["convert_lead", "teleport_lead"]
    issues = assess_scenario_suite(suite, known_actions={"convert_lead", "archive_lead"})
    categories = {i.category for i in issues}
    assert "unknown_action" in categories
    # A false-positive guard: the legitimately-declared actions must NOT be flagged.
    messages = " ".join(i.message for i in issues if i.category == "unknown_action")
    assert "teleport_lead" in messages
    assert "convert_lead" not in messages


def test_known_actions_only_produce_no_unknown_action_issue():
    issues = assess_scenario_suite(
        _healthy_suite(), known_actions={"convert_lead", "archive_lead"}
    )
    assert "unknown_action" not in {i.category for i in issues}


# ── Seed reproducibility ──────────────────────────────────────────────────


def test_normalize_seeds_makes_seeds_distinct_and_is_reproducible():
    suite = _healthy_suite()
    suite.scenarios[0].seed = 5
    suite.scenarios[1].seed = 5  # collision from the LLM
    first = normalize_seeds(suite)
    second = normalize_seeds(suite)
    seeds = [s.seed for s in first.scenarios]
    assert len(set(seeds)) == len(seeds)  # distinct
    assert [s.seed for s in second.scenarios] == seeds  # deterministic


def test_normalize_seeds_assigns_missing_seeds():
    suite = _healthy_suite()
    suite.scenarios[0].seed = -1  # unassigned sentinel
    suite.scenarios[1].seed = -1
    seeds = [s.seed for s in normalize_seeds(suite).scenarios]
    assert all(seed >= 0 for seed in seeds)
    assert len(set(seeds)) == len(seeds)


def test_scenario_for_seed_selects_the_matching_scenario():
    suite = _healthy_suite()
    assert scenario_for_seed(suite, 2).scenario_id == "order_matters"
    assert scenario_for_seed(suite, 999) is None


# ── Agent: grounds in the real product first ──────────────────────────────


class _RecordingClient:
    def __init__(self, suite: ScenarioSuite) -> None:
        self.suite = suite
        self.last_user: str | None = None
        self.last_system: str | None = None

    def extract(self, system: str, user: str, schema):
        self.last_system = system
        self.last_user = user
        return self.suite


@pytest.mark.asyncio
async def test_agent_grounds_scenarios_in_research_brief_when_present():
    client = _RecordingClient(_healthy_suite())
    agent = ScenarioBuilderAgent(client=client)
    bus = ArtifactBus()
    await bus.publish(
        "rl_research",
        SpecialistResearchContext(
            role="rl",
            product_summary="Salesloft-style CRM where reps convert qualified leads",
        ),
    )
    await agent.run(_ctx(), bus)
    assert "Salesloft-style CRM" in client.last_user


@pytest.mark.asyncio
async def test_agent_falls_back_to_user_description_without_research():
    client = _RecordingClient(_healthy_suite())
    agent = ScenarioBuilderAgent(client=client)
    bus = ArtifactBus()
    await agent.run(_ctx(description="A helpdesk for triaging support tickets"), bus)
    assert "helpdesk for triaging support tickets" in client.last_user


@pytest.mark.asyncio
async def test_agent_publishes_suite_with_normalized_seeds_and_env_name():
    suite = _healthy_suite()
    suite.scenarios[0].seed = 7
    suite.scenarios[1].seed = 7  # collision the agent must repair
    agent = ScenarioBuilderAgent(client=_RecordingClient(suite))
    bus = ArtifactBus()
    await agent.run(_ctx(), bus)
    published = bus.get("scenario_suite")
    assert isinstance(published, ScenarioSuite)
    assert published.env_name == "crm_env"
    seeds = [s.seed for s in published.scenarios]
    assert len(set(seeds)) == len(seeds)


@pytest.mark.asyncio
async def test_agent_declares_optional_research_dependency_and_output():
    agent = ScenarioBuilderAgent(client=_RecordingClient(_healthy_suite()))
    assert "rl_research" in agent.optional_depends_on
    assert agent.produces == ["scenario_suite"]


# ── Prompt mandates grounding + the required scenario ingredients ─────────


def test_prompt_mandates_grounding_and_scenario_ingredients():
    prompt = ScenarioBuilderPrompts.SYSTEM.lower()
    # Understand the real product first.
    assert "product" in prompt
    assert "distractor" in prompt
    assert "conflict" in prompt
    assert "order" in prompt
    # Verifier-facing fields.
    assert "required_actions" in prompt
    assert "forbidden_actions" in prompt
    assert "expected_answer" in prompt
    assert "seed" in prompt
