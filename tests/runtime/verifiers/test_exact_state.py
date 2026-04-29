from dataclasses import dataclass, field
from forge.runtime.verifiers.exact_state import ExactStateVerifier


@dataclass
class _FakeTraj:
    steps: list = field(default_factory=list)
    events: list = field(default_factory=list)


def test_passes_when_expression_is_true():
    v = ExactStateVerifier("x > 5")
    result = v.check({"x": 10}, _FakeTraj(), {})
    assert result.passed
    assert result.score == 1.0
    assert result.evidence is None


def test_fails_when_expression_is_false():
    v = ExactStateVerifier("x > 5")
    result = v.check({"x": 3}, _FakeTraj(), {})
    assert not result.passed
    assert result.score == 0.0
    assert "3" in result.evidence


def test_fails_on_eval_error():
    v = ExactStateVerifier("undefined_var > 5")
    result = v.check({}, _FakeTraj(), {})
    assert not result.passed
    assert "Eval error" in result.evidence


def test_name_is_expression():
    v = ExactStateVerifier("x == 1")
    result = v.check({"x": 1}, _FakeTraj(), {})
    assert result.name == "x == 1"


def test_no_builtins_in_eval():
    v = ExactStateVerifier("__import__('os')")
    result = v.check({}, _FakeTraj(), {})
    assert not result.passed
