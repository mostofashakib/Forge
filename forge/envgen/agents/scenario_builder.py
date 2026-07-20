"""Realistic initial-state scenario specialist (TASKS.md #4).

A generation-time specialist that first internalizes how the *real* product
works — from the user-researcher brief or the user's own description — and then
emits diverse, seed-selectable initial-state scenarios. Each scenario carries
deliberate distractors, ordering-sensitive action sequences, and conflicting
information, plus the per-scenario ground truth (expected answer and the
required/forbidden actions) that task generation and the verifier consume.

``assess_scenario_suite`` is the deterministic quality gate: it rejects
degenerate suites (a single trivial record, no distractors, no conflicts) so a
scenario-free "dumb template" can never masquerade as a rich environment.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, Field

from forge.envgen.agents.base import EnvGenAgent, with_correction
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.config import envgen_config
from forge.envgen.context import EnvGenContext
from forge.extraction.llm_client import LLMClient, get_client

# A seed of -1 marks "the model did not assign one"; normalize_seeds fills it in.
_UNASSIGNED_SEED = -1


class Scenario(BaseModel):
    """One reproducible initial-state universe plus its ground truth."""

    scenario_id: str
    title: str = ""
    # Deterministic selector; wired to seed_state(seed) once container seeding
    # lands (task #7). Distinct across a suite so each scenario is addressable.
    seed: int = _UNASSIGNED_SEED
    objective: str = ""
    # entity name -> list of seeded rows for that entity.
    records: dict[str, list[dict]] = Field(default_factory=dict)
    # Ids/descriptions of plausible-but-irrelevant records the agent must ignore.
    distractors: list[str] = Field(default_factory=list)
    # Contradictory or stale-vs-fresh information the agent must reconcile.
    conflicts: list[str] = Field(default_factory=list)
    ordering_sensitive: bool = False
    ordering_note: str = ""
    # Ground truth for the verifier (#5): the exact actions that must run (in
    # order), the ones that must not, and the expected outcome.
    required_actions: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)
    expected_answer: str = ""

    def record_count(self) -> int:
        return sum(len(rows) for rows in self.records.values())


class ScenarioSuite(BaseModel):
    env_name: str = ""
    scenarios: list[Scenario] = Field(default_factory=list)


@dataclass
class ScenarioIssue:
    category: str
    message: str
    scenario_id: str | None = None


# ---------------------------------------------------------------------------
# Deterministic quality gate
# ---------------------------------------------------------------------------


def assess_scenario_suite(
    suite: ScenarioSuite, known_actions: set[str] | None = None
) -> list[ScenarioIssue]:
    """Return issues that make a scenario suite insufficient for RL training.

    A suite is sufficient when it holds at least two scenarios with distinct
    seeds, each scenario is non-trivial, and — across the suite — distractors,
    conflicting information, and an ordering-sensitive case all appear. When
    ``known_actions`` is supplied, required/forbidden actions must reference
    real actions.
    """
    issues: list[ScenarioIssue] = []
    scenarios = suite.scenarios

    if len(scenarios) < 2:
        issues.append(ScenarioIssue(
            "insufficient_diversity",
            "A scenario suite needs at least two distinct scenarios, not one fixture",
        ))

    seen_seeds: set[int] = set()
    for scenario in scenarios:
        if scenario.seed in seen_seeds:
            issues.append(ScenarioIssue(
                "duplicate_seed",
                f"Seed {scenario.seed} is reused; scenarios must be seed-addressable",
                scenario.scenario_id,
            ))
        seen_seeds.add(scenario.seed)

        if scenario.record_count() < 2:
            issues.append(ScenarioIssue(
                "degenerate_scenario",
                "Scenario has fewer than two records; nothing to reason about "
                "or distinguish from a distractor",
                scenario.scenario_id,
            ))

        if known_actions is not None:
            for name in scenario.required_actions + scenario.forbidden_actions:
                if name not in known_actions:
                    issues.append(ScenarioIssue(
                        "unknown_action",
                        f"Scenario references unknown action {name!r}",
                        scenario.scenario_id,
                    ))

    if scenarios and not any(s.distractors for s in scenarios):
        issues.append(ScenarioIssue(
            "missing_distractors",
            "No scenario includes distractors; add plausible-but-irrelevant records",
        ))
    if scenarios and not any(s.conflicts for s in scenarios):
        issues.append(ScenarioIssue(
            "missing_conflicts",
            "No scenario includes conflicting information to reconcile",
        ))
    if scenarios and not any(s.ordering_sensitive for s in scenarios):
        issues.append(ScenarioIssue(
            "missing_ordering",
            "No scenario is ordering-sensitive; add one where call order matters",
        ))
    return issues


def normalize_seeds(suite: ScenarioSuite) -> ScenarioSuite:
    """Return a copy whose scenarios carry distinct, reproducible seeds.

    Existing distinct, non-negative seeds are preserved; collisions and
    unassigned (``-1``) seeds are deterministically filled from the scenario
    index so the same input always yields the same seeds.
    """
    seen: set[int] = set()
    fixed: list[Scenario] = []
    for index, scenario in enumerate(suite.scenarios):
        seed = scenario.seed
        if seed < 0 or seed in seen:
            seed = index
            while seed in seen:
                seed += 1
        seen.add(seed)
        fixed.append(scenario.model_copy(update={"seed": seed}))
    return suite.model_copy(update={"scenarios": fixed})


def scenario_for_seed(suite: ScenarioSuite, seed: int) -> Scenario | None:
    """Return the scenario addressed by ``seed``, or ``None`` if absent."""
    for scenario in suite.scenarios:
        if scenario.seed == seed:
            return scenario
    return None


# ---------------------------------------------------------------------------
# Specialist agent
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a scenario-design specialist for a reinforcement-learning environment.\n"
    "\n"
    "STEP 1 — UNDERSTAND THE REAL PRODUCT FIRST. Before inventing any data, study the\n"
    "PRODUCT UNDERSTANDING block below (either a research brief about how the real\n"
    "application works, or the user's own description of it). Every record you write\n"
    "must be data that this real product would plausibly contain — realistic names,\n"
    "statuses, relationships, and volumes for this domain. Never emit generic\n"
    "'foo/bar/test' fixtures.\n"
    "\n"
    "STEP 2 — DESIGN A DIVERSE SUITE. Produce SEVERAL distinct scenarios (at least\n"
    "three), each a self-contained initial-state universe with a unique integer\n"
    "`seed`. Across the suite you MUST include:\n"
    "  - DISTRACTORS: plausible-but-irrelevant records the agent must recognize and\n"
    "    NOT act on. List their ids in `distractors`.\n"
    "  - ORDERING-SENSITIVE cases: set `ordering_sensitive` true and describe in\n"
    "    `ordering_note` why the correct sequence of actions matters (a wrong order\n"
    "    fails or causes a harmful side effect).\n"
    "  - CONFLICTING INFORMATION: contradictory or stale-vs-fresh records the agent\n"
    "    must reconcile. Describe each in `conflicts`.\n"
    "\n"
    "STEP 3 — DEFINE THE GROUND TRUTH per scenario so it can be graded:\n"
    "  - `objective`: what the agent must accomplish.\n"
    "  - `required_actions`: the exact actions that must be called, in order.\n"
    "  - `forbidden_actions`: actions that must NOT be called (e.g. acting on a\n"
    "    distractor).\n"
    "  - `expected_answer`: the correct final outcome.\n"
    "Only reference actions that actually exist in the provided action list.\n"
    "\n"
    "Records are keyed by entity name; each value is a list of row dicts. Make each\n"
    "scenario non-trivial (multiple records) so distractors are meaningful.\n"
    "Call the extract tool with the ScenarioSuite."
)


class ScenarioBuilderPrompts:
    SYSTEM = _SYSTEM


class ScenarioBuilderAgent(EnvGenAgent):
    """Generates diverse, product-grounded initial-state scenarios."""

    agent_id = "scenario_builder"
    optional_depends_on: list[str] = ["rl_research"]
    produces: list[str] = ["scenario_suite"]

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or get_client(
            max_tokens=envgen_config().capable_llm_tokens, capable=True
        )

    def _grounding(self, ctx: EnvGenContext, bus: ArtifactBus) -> str:
        research = bus.get("rl_research")
        if research is not None:
            return f"PRODUCT UNDERSTANDING (research brief):\n{research.as_prompt()}"
        return (
            "PRODUCT UNDERSTANDING (from the user's description of the real product):\n"
            f"{ctx.description}"
        )

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        entity_summary = "\n".join(
            f"  - {e.name}: fields={[f.name for f in e.fields]}"
            for e in ctx.compiler_input.entities
        ) or "  (no entities declared)"
        action_names = [a.name for a in ctx.compiler_input.actions]

        user = (
            f"{self._grounding(ctx, bus)}\n\n"
            f"Domain: {ctx.compiler_input.domain}\n\n"
            f"Entities:\n{entity_summary}\n\n"
            f"Available actions (use ONLY these names): {action_names}"
        )
        user = with_correction(bus, self.agent_id, user)

        await bus.log(
            f"[scenario-builder] Designing scenarios for {len(action_names)} actions…"
        )
        loop = asyncio.get_event_loop()
        result: ScenarioSuite = await loop.run_in_executor(
            None,
            lambda: self._client.extract(
                system=ScenarioBuilderPrompts.SYSTEM, user=user, schema=ScenarioSuite
            ),
        )

        suite = normalize_seeds(result.model_copy(update={"env_name": ctx.env_name}))
        issues = assess_scenario_suite(suite, known_actions=set(action_names))
        if issues:
            summary = "; ".join(f"{i.category}: {i.message}" for i in issues)
            await bus.log(f"[scenario-builder] WARNING — suite is thin: {summary}")
        else:
            await bus.log(
                f"[scenario-builder] Published {len(suite.scenarios)} scenarios "
                f"(seeds {[s.seed for s in suite.scenarios]})"
            )
        await bus.publish("scenario_suite", suite)
