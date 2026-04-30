import pytest
from forge.runtime.observation_filter import ObservationFilter, RBACConfig, RolePermissions


def test_identity_when_no_config():
    f = ObservationFilter()
    obs = {"tickets": [1, 2], "billing": {"amount": 100}}
    assert f.filter(obs) == obs


def test_identity_when_role_not_in_config():
    config = RBACConfig(roles={"admin": RolePermissions(can_see=["tickets"])})
    f = ObservationFilter(rbac_config=config, role="unknown_role")
    obs = {"tickets": [1], "billing": {"amount": 100}}
    assert f.filter(obs) == obs


def test_can_see_keeps_only_allowed():
    config = RBACConfig(
        roles={"support": RolePermissions(can_see=["tickets", "customers"])}
    )
    f = ObservationFilter(rbac_config=config, role="support")
    obs = {"tickets": [1], "customers": [2], "billing": {"amount": 100}}
    result = f.filter(obs)
    assert "tickets" in result
    assert "customers" in result
    assert "billing" not in result


def test_cannot_see_removes_keys():
    config = RBACConfig(
        roles={"support": RolePermissions(cannot_see=["billing", "audit_logs"])}
    )
    f = ObservationFilter(rbac_config=config, role="support")
    obs = {"tickets": [1], "billing": {"amount": 100}, "audit_logs": []}
    result = f.filter(obs)
    assert "tickets" in result
    assert "billing" not in result
    assert "audit_logs" not in result


def test_cannot_see_takes_precedence_over_can_see():
    config = RBACConfig(
        roles={
            "support": RolePermissions(
                can_see=["tickets", "billing"],
                cannot_see=["billing"],
            )
        }
    )
    f = ObservationFilter(rbac_config=config, role="support")
    obs = {"tickets": [1], "billing": {"amount": 100}}
    result = f.filter(obs)
    assert "tickets" in result
    assert "billing" not in result


def test_empty_can_see_and_cannot_see_is_identity():
    config = RBACConfig(roles={"support": RolePermissions()})
    f = ObservationFilter(rbac_config=config, role="support")
    obs = {"tickets": [1], "billing": {"amount": 100}}
    assert f.filter(obs) == obs
