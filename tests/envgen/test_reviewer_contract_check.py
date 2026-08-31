from __future__ import annotations

import pytest

from forge.envgen.agents.reviewer import ReviewerAgent, ReviewSeverity
from forge.envgen.artifact_bus import ArtifactBus
from forge.extraction.schemas import ActionDef, CompilerInput

_ROUTES = " ".join((
    "/forge/health", "/forge/state", "/forge/reset", "/forge/snapshot",
    "/forge/restore", "/forge/restore-state", "close_ticket",
))


def _ctx():
    from forge.envgen.context import EnvGenContext

    return EnvGenContext(
        env_name="ticket_env",
        description="A ticket queue",
        compiler_input=CompilerInput(
            project_name="ticket_env", domain="support", entities=[],
            actions=[ActionDef(name="close_ticket", params=[])], tasks=[],
        ),
    )


async def _bus(state_bridge: str, reward: str) -> ArtifactBus:
    bus = ArtifactBus()
    main = f"ROUTES = {_ROUTES!r}\n"
    await bus.publish("app_code", {
        "main.py": main,
        "requirements.txt": "fastapi\n",
        "Dockerfile": "FROM python:3.12-slim\n",
    })
    await bus.publish("instrumented_code", {"main.py": main})
    await bus.publish("state_bridge_code", state_bridge)
    await bus.publish("state_schema_manifest", {"fields": {}})
    await bus.publish("policy_dsl", "policies: []\n")
    await bus.publish("reward_fn_code", reward)
    return bus


_CONFORMING_BRIDGE = (
    "from forge.contracts import Environment\n"
    "class ContainerForgeEnv(Environment):\n    pass\n"
)
_CONFORMING_REWARD = (
    "from forge.contracts import Rubric\n"
    "class TicketRubric(Rubric):\n"
    "    def score(self, state, trajectory, verifier_results, task):\n"
    "        return None\n"
)


async def _review(state_bridge: str, reward: str):
    bus = await _bus(state_bridge, reward)
    agent = ReviewerAgent(semantic_review=False)
    await agent.run(_ctx(), bus)
    return bus.get("review_report")


@pytest.mark.asyncio
async def test_conforming_artifacts_produce_no_contract_issue():
    review = await _review(_CONFORMING_BRIDGE, _CONFORMING_REWARD)
    contract_issues = [i for i in review.issues if i.category == "contract"]
    assert contract_issues == []


@pytest.mark.asyncio
async def test_bridge_not_subclassing_environment_is_a_contract_issue():
    bridge = "class ContainerForgeEnv:\n    pass\n"
    review = await _review(bridge, _CONFORMING_REWARD)
    contract_issues = [i for i in review.issues if i.category == "contract"]
    assert contract_issues
    assert all(i.severity == ReviewSeverity.ERROR for i in contract_issues)
    assert contract_issues[0].artifact == "state_bridge_code"


@pytest.mark.asyncio
async def test_reward_without_rubric_subclass_is_a_contract_issue():
    reward = "class TicketRubric:\n    def score(self, state, trajectory, verifier_results, task):\n        return None\n"
    review = await _review(_CONFORMING_BRIDGE, reward)
    contract_issues = [i for i in review.issues if i.category == "contract"]
    assert contract_issues
    assert all(i.severity == ReviewSeverity.ERROR for i in contract_issues)
    assert contract_issues[0].artifact == "reward_fn_code"


@pytest.mark.asyncio
async def test_rubric_without_score_is_a_contract_issue():
    reward = (
        "from forge.contracts import Rubric\n"
        "class TicketRubric(Rubric):\n"
        "    pass\n"
    )
    review = await _review(_CONFORMING_BRIDGE, reward)
    contract_issues = [i for i in review.issues if i.category == "contract"]
    assert contract_issues
    assert all(i.severity == ReviewSeverity.ERROR for i in contract_issues)
    assert contract_issues[0].artifact == "reward_fn_code"
    assert any("score" in i.message for i in contract_issues)


@pytest.mark.asyncio
async def test_async_score_is_a_contract_issue_with_a_distinct_message():
    # `score` exists here (unlike the missing-score case above) but as
    # `async def`, which fails the `ast.FunctionDef` check since
    # `ast.AsyncFunctionDef` is a different node type. The message must say
    # so plainly rather than reusing "must define score()" -- an automated
    # repair specialist reading that while `score` is visibly present could
    # plausibly "fix" it by adding a second, sync `score` instead of
    # removing `async` from the existing one.
    reward = (
        "from forge.contracts import Rubric\n"
        "class TicketRubric(Rubric):\n"
        "    async def score(self, state, trajectory, verifier_results, task):\n"
        "        return None\n"
    )
    review = await _review(_CONFORMING_BRIDGE, reward)
    contract_issues = [i for i in review.issues if i.category == "contract"]
    assert contract_issues
    assert all(i.severity == ReviewSeverity.ERROR for i in contract_issues)
    assert contract_issues[0].artifact == "reward_fn_code"
    assert any("async" in i.message for i in contract_issues)
    # The two failure modes must not share a message: reading it should
    # make clear this is not the "no score() at all" case.
    assert not any(
        "must define score()" in i.message for i in contract_issues
    )


@pytest.mark.asyncio
async def test_missing_score_and_async_score_produce_different_messages():
    missing_reward = (
        "from forge.contracts import Rubric\n"
        "class TicketRubric(Rubric):\n"
        "    pass\n"
    )
    async_reward = (
        "from forge.contracts import Rubric\n"
        "class TicketRubric(Rubric):\n"
        "    async def score(self, state, trajectory, verifier_results, task):\n"
        "        return None\n"
    )
    missing_review = await _review(_CONFORMING_BRIDGE, missing_reward)
    async_review = await _review(_CONFORMING_BRIDGE, async_reward)

    missing_message = next(
        i.message for i in missing_review.issues if i.category == "contract"
    )
    async_message = next(
        i.message for i in async_review.issues if i.category == "contract"
    )
    assert missing_message != async_message


@pytest.mark.asyncio
async def test_unparseable_source_produces_no_contract_issue():
    # Missing colon -- a syntax error, not a contract violation. The syntax
    # loop already reports this; the contract gate must not double-report it
    # as a spurious "no Environment subclass found" contract error.
    bridge = "class ContainerForgeEnv(Environment)\n    pass\n"
    review = await _review(bridge, _CONFORMING_REWARD)
    syntax_issues = [i for i in review.issues if i.category == "syntax"]
    contract_issues = [i for i in review.issues if i.category == "contract"]
    assert syntax_issues
    assert contract_issues == []


@pytest.mark.asyncio
async def test_dotted_base_is_recognized_as_environment_subclass():
    bridge = (
        "import forge.contracts\n"
        "class ContainerForgeEnv(forge.contracts.Environment):\n    pass\n"
    )
    review = await _review(bridge, _CONFORMING_REWARD)
    contract_issues = [
        i for i in review.issues
        if i.category == "contract" and i.artifact == "state_bridge_code"
    ]
    assert contract_issues == []


@pytest.mark.asyncio
async def test_contract_findings_route_to_the_right_specialist_end_to_end():
    # It is not enough for _contract_issues to construct an issue with the
    # right artifact field -- the repair loop only self-corrects if
    # FindingRouter actually sends that issue to the specialist who produced
    # the offending artifact. Exercise the real path: a non-conforming
    # bridge and reward through ReviewerAgent, then through FindingRouter
    # built from the real specialist classes.
    from forge.envgen.agents.reward import RewardAgent
    from forge.envgen.agents.state_bridge import StateBridgeAgent
    from forge.envgen.repair import FindingRouter

    non_conforming_bridge = "class ContainerForgeEnv:\n    pass\n"
    non_conforming_reward = "class TicketRubric:\n    pass\n"
    review = await _review(non_conforming_bridge, non_conforming_reward)
    contract_issues = [i for i in review.issues if i.category == "contract"]
    assert len(contract_issues) == 2

    router = FindingRouter([StateBridgeAgent, RewardAgent])
    resolved = {router.route(issue) for issue in contract_issues}
    assert resolved == {"state_bridge", "reward"}

    by_artifact = {issue.artifact: router.route(issue) for issue in contract_issues}
    assert by_artifact["state_bridge_code"] == "state_bridge"
    assert by_artifact["reward_fn_code"] == "reward"
