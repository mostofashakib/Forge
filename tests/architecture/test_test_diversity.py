# tests/architecture/test_test_diversity.py
"""The test-scenario-diversity standard, enforced as a review gate.

A test module that defines behavior tests must exercise more than the happy
path: it must carry at least one negative or false-positive scenario (an
`Error`/`raises` assertion, or a test named for a rejection / boundary / guard).

These tests exercise the analyzer that powers the gate. Per the very standard
it enforces, each behavior is paired with a negative case (a happy-only module
must be flagged) and a false-positive case (a module with no behavior tests, or
whose negative signal comes from a non-test helper, must NOT be flagged).
"""
from __future__ import annotations

from pathlib import Path

from tests.architecture.diversity_audit import (
    analyze_source,
    audit_test_tree,
)

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Happy path: modules that DO carry negative / false-positive signal
# ---------------------------------------------------------------------------

def test_module_using_pytest_raises_carries_signal():
    report = analyze_source(
        """
import pytest
def test_returns_sum():
    assert add(1, 2) == 3
def test_rejects_bad_input():
    with pytest.raises(ValueError):
        add(1, "x")
"""
    )
    assert report.has_behavior_tests
    assert report.uses_raises
    assert report.carries_diversity_signal
    assert not report.is_happy_path_only


def test_module_with_a_negatively_named_test_carries_signal():
    report = analyze_source(
        """
def test_parses_valid_row():
    assert parse("ok")
def test_missing_column_is_rejected():
    assert parse("") is None
"""
    )
    assert report.negative_tests == ["test_missing_column_is_rejected"]
    assert report.carries_diversity_signal
    assert not report.is_happy_path_only


def test_false_positive_named_test_counts_as_signal():
    # A test guarding against an over-eager pass is exactly the false-positive
    # coverage the standard asks for.
    report = analyze_source(
        """
def test_tied_pair_produces_no_example():
    assert build([("a", 1.0, 1.0)]) == []
"""
    )
    assert report.carries_diversity_signal


def test_differential_body_assertion_counts_as_signal():
    # A "must differ" check (assert a != b) is negative/false-positive coverage
    # even when the test name reads as a happy path.
    report = analyze_source(
        """
def test_different_seeds_produce_different_sequences():
    assert rng(1) != rng(2)
"""
    )
    assert report.has_negative_assertion
    assert report.carries_diversity_signal
    assert not report.is_happy_path_only


def test_exclusion_body_assertion_counts_as_signal():
    # "must NOT be present" (assert x not in y) is a false-positive guard.
    report = analyze_source(
        """
def test_secret_is_redacted():
    assert "alice@example.com" not in redact(text)
"""
    )
    assert report.has_negative_assertion
    assert not report.is_happy_path_only


def test_assert_not_counts_as_signal():
    report = analyze_source(
        """
def test_accepts_input():
    assert validate("ok")
    assert not validate("")
"""
    )
    assert report.has_negative_assertion


def test_only_positive_equality_assertions_do_not_count():
    # False-positive guard for the analyzer itself: a body full of `==` / `in`
    # happy assertions carries no negative signal and must be flagged.
    report = analyze_source(
        """
def test_returns_sum():
    assert add(1, 2) == 3
    assert 3 in results
"""
    )
    assert not report.has_negative_assertion
    assert report.is_happy_path_only


def test_negative_signal_in_a_test_class_method_is_detected():
    report = analyze_source(
        """
class TestParser:
    def test_parses(self):
        assert parse("ok")
    def test_raises_on_garbage(self):
        with pytest.raises(ValueError):
            parse("!!")
"""
    )
    assert "test_parses" in report.test_functions
    assert report.uses_raises
    assert not report.is_happy_path_only


# ---------------------------------------------------------------------------
# Negative: a happy-path-only module MUST be flagged
# ---------------------------------------------------------------------------

def test_happy_path_only_module_is_flagged():
    report = analyze_source(
        """
def test_returns_sum():
    assert add(1, 2) == 3
def test_round_trips_json():
    assert loads(dumps(x)) == x
"""
    )
    assert report.has_behavior_tests
    assert not report.carries_diversity_signal
    assert report.is_happy_path_only


# ---------------------------------------------------------------------------
# False-positive guards: modules that must NOT be flagged
# ---------------------------------------------------------------------------

def test_module_with_no_behavior_tests_is_not_flagged():
    # Pure helpers / fixtures with no test_* functions assert nothing, so there
    # is no happy path to be "only" — flagging it would be an over-eager pass.
    report = analyze_source(
        """
import pytest
@pytest.fixture
def widget():
    return Widget()
def _build_invalid_payload():
    return {"bad": True}
"""
    )
    assert not report.has_behavior_tests
    assert not report.is_happy_path_only


def test_negative_keyword_in_a_non_test_helper_does_not_count():
    # A helper named for rejection is not a test; a module whose only *tests*
    # are happy-path must still be flagged despite the helper's name.
    report = analyze_source(
        """
def _assert_invalid(x):
    ...
def test_returns_sum():
    assert add(1, 2) == 3
"""
    )
    assert report.negative_tests == []
    assert report.is_happy_path_only


def test_syntactically_broken_module_is_skipped_not_flagged():
    # An unparseable file is not evidence of a happy-path-only test module.
    report = analyze_source("def test_(:\n")
    assert report.unparsed
    assert not report.is_happy_path_only


# ---------------------------------------------------------------------------
# Tree audit + exemptions
# ---------------------------------------------------------------------------

def test_audit_tree_respects_exemptions(tmp_path):
    (tmp_path / "test_happy.py").write_text(
        "def test_a():\n    assert True\n"
    )
    (tmp_path / "test_good.py").write_text(
        "import pytest\n"
        "def test_a():\n    assert True\n"
        "def test_rejects():\n    \n    assert True\n"
    )
    # Without exemption the happy-only module is reported.
    flagged = {r.path for r in audit_test_tree(tmp_path)}
    assert any(p.endswith("test_happy.py") for p in flagged)
    assert not any(p.endswith("test_good.py") for p in flagged)
    # Exempting it by relative path clears the report.
    flagged_exempt = audit_test_tree(tmp_path, exempt={"test_happy.py"})
    assert flagged_exempt == []


# ---------------------------------------------------------------------------
# The real gate: the Forge test suite must satisfy the standard
# ---------------------------------------------------------------------------

# Modules that legitimately assert a single happy path (pure smoke / import /
# round-trip checks with no rejectable input). Each needs a reason.
_EXEMPT: dict[str, str] = {}


def test_forge_test_suite_covers_negative_and_false_positive_scenarios():
    offenders = audit_test_tree(ROOT / "tests", exempt=set(_EXEMPT))
    listing = "\n".join(f"  - {r.path} (tests: {', '.join(r.test_functions)})" for r in offenders)
    assert not offenders, (
        "These test modules assert only the happy path. Add at least one "
        "negative case (invalid input / error path) and one false-positive "
        "case (a look-valid input that must be rejected), or exempt the module "
        "in _EXEMPT with a reason:\n" + listing
    )
