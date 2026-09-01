"""The boundary a driver's proposal has to survive."""
from __future__ import annotations

from forge.contracts.types import Action, ToolParam, ToolSpec
from forge.personas.guardrails import ActionGuard

from tests.personas.conftest import POST_MESSAGE_SPEC, persona


def test_action_inside_the_declared_space_is_allowed():
    guard = ActionGuard(["post_message"], [POST_MESSAGE_SPEC])
    decision = guard.check(
        persona("nurse", allowed_actions=["post_message"]),
        Action(type="post_message", params={"body": "on my way"}),
    )
    assert decision.allowed


def test_action_outside_the_declared_space_is_blocked_with_the_space_named():
    guard = ActionGuard(["post_message", "delete_chart"])
    decision = guard.check(
        persona("nurse", allowed_actions=["post_message"]),
        Action(type="delete_chart"),
    )
    assert not decision.allowed
    assert "delete_chart" in decision.reason
    assert "post_message" in decision.reason


def test_persona_with_an_empty_action_space_may_do_nothing():
    guard = ActionGuard(["post_message"])
    decision = guard.check(
        persona("nurse", allowed_actions=[]), Action(type="post_message")
    )
    assert not decision.allowed
    assert "no allowed_actions" in decision.reason


def test_action_the_environment_does_not_implement_is_blocked():
    """Catches the config drift of renaming an action and forgetting the cast."""
    guard = ActionGuard(["post_message"])
    decision = guard.check(
        persona("nurse", allowed_actions=["send_page"]), Action(type="send_page")
    )
    assert not decision.allowed
    assert "does not implement" in decision.reason


def test_missing_required_parameter_is_blocked():
    guard = ActionGuard(["post_message"], [POST_MESSAGE_SPEC])
    decision = guard.check(
        persona("nurse", allowed_actions=["post_message"]),
        Action(type="post_message", params={}),
    )
    assert not decision.allowed
    assert "body" in decision.reason


def test_optional_parameter_may_be_omitted():
    """False-positive guard: only *required* params are enforced."""
    spec = ToolSpec(
        name="post_message",
        params=[ToolParam(name="body", required=False)],
    )
    guard = ActionGuard(["post_message"], [spec])
    assert guard.check(
        persona("nurse", allowed_actions=["post_message"]),
        Action(type="post_message", params={}),
    ).allowed


def test_environment_check_is_skipped_when_the_surface_is_unknown():
    """A container env cannot enumerate its actions; the other checks still run."""
    guard = ActionGuard(environment_actions=None)
    spec = persona("nurse", allowed_actions=["anything"])
    assert guard.check(spec, Action(type="anything")).allowed
    assert not guard.check(spec, Action(type="something_else")).allowed


def test_action_space_renders_schemas_in_a_stable_order():
    guard = ActionGuard(
        ["post_message", "review_chart"],
        [POST_MESSAGE_SPEC, ToolSpec(name="review_chart")],
    )
    space = guard.action_space(
        persona("nurse", allowed_actions=["review_chart", "post_message"])
    )
    assert [s.name for s in space] == ["post_message", "review_chart"]
    assert space[0].params[0].name == "body"


def test_action_space_hides_actions_the_environment_lacks():
    guard = ActionGuard(["post_message"])
    space = guard.action_space(
        persona("nurse", allowed_actions=["post_message", "stale_action"])
    )
    assert [s.name for s in space] == ["post_message"]
