import pytest
from forge.runtime.policy_engine import PolicyEngine, PolicyViolationResult
from forge.extraction.schemas import PolicyRule


@pytest.fixture
def refund_rule():
    return PolicyRule(
        id="no_refund_without_order",
        condition="state.get('order_id') is None",
        forbidden_actions=["offer_refund"],
        description="high: Refund without order ID is forbidden",
    )


@pytest.fixture
def discount_rule():
    return PolicyRule(
        id="no_discount_vip",
        condition="state.get('customer_tier') == 'vip'",
        forbidden_actions=["apply_discount"],
        description="medium: VIP customers cannot receive discounts",
    )


def test_violation_when_condition_true_and_action_forbidden(refund_rule):
    engine = PolicyEngine([refund_rule])
    violations = engine.check(state={"order_id": None}, action={"type": "offer_refund"})
    assert len(violations) == 1
    assert violations[0].rule_id == "no_refund_without_order"
    assert violations[0].forbidden_action == "offer_refund"
    assert violations[0].severity == "high"


def test_no_violation_when_condition_false(refund_rule):
    engine = PolicyEngine([refund_rule])
    violations = engine.check(state={"order_id": "ord_123"}, action={"type": "offer_refund"})
    assert violations == []


def test_no_violation_when_action_not_forbidden(refund_rule):
    engine = PolicyEngine([refund_rule])
    violations = engine.check(state={"order_id": None}, action={"type": "close_ticket"})
    assert violations == []


def test_multiple_rules_only_matching_action_triggers(refund_rule, discount_rule):
    engine = PolicyEngine([refund_rule, discount_rule])
    violations = engine.check(
        state={"order_id": None, "customer_tier": "vip"},
        action={"type": "offer_refund"},
    )
    assert len(violations) == 1
    assert violations[0].rule_id == "no_refund_without_order"


def test_bad_condition_expression_is_skipped():
    bad_rule = PolicyRule(
        id="bad_rule",
        condition="this is not valid python!!!",
        forbidden_actions=["offer_refund"],
        description="low: bad rule",
    )
    engine = PolicyEngine([bad_rule])
    violations = engine.check(state={}, action={"type": "offer_refund"})
    assert violations == []


def test_empty_rules_no_violations():
    engine = PolicyEngine([])
    violations = engine.check(state={"anything": True}, action={"type": "offer_refund"})
    assert violations == []


def test_private_attribute_traversal_is_rejected():
    rule = PolicyRule(
        id="unsafe",
        condition="state.__class__.__base__.__subclasses__()",
        forbidden_actions=["x"],
    )
    assert PolicyEngine([rule]).check({}, {"type": "x"}) == []


def test_severity_derived_from_description():
    high_rule = PolicyRule(id="r1", condition="True", forbidden_actions=["x"], description="high: bad")
    med_rule = PolicyRule(id="r2", condition="True", forbidden_actions=["x"], description="medium: ok")
    low_rule = PolicyRule(id="r3", condition="True", forbidden_actions=["x"], description="no severity keyword")
    engine = PolicyEngine([high_rule, med_rule, low_rule])
    v = engine.check(state={}, action={"type": "x"})
    assert v[0].severity == "high"
    assert v[1].severity == "medium"
    assert v[2].severity == "low"
