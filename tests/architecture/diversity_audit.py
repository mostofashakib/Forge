# tests/architecture/diversity_audit.py
"""Static detection of test-scenario diversity.

Powers the review gate in ``test_test_diversity.py``: a test module that
defines behavior tests must carry at least one *negative* or *false-positive*
scenario, not just the happy path. We approximate that from the AST — no test
is executed — via two signals:

  * ``uses_raises`` — the module asserts a failure path with ``pytest.raises``
    or ``pytest.warns``.
  * ``negative_tests`` — a ``test_*`` function whose name is written for a
    rejection, boundary, guard, or must-not-happen scenario.
  * ``has_negative_assertion`` — a differential / exclusion assertion in the
    body (``assert a != b``, ``assert x not in y``, ``assert not x``,
    ``assert x is not y``), the way this codebase often expresses a
    "must differ" or "must be absent" guard.

A module with neither, yet with behavior tests, is *happy-path only* and is
flagged. A module with no behavior tests (pure helpers/fixtures) is never
flagged — there is no happy path for it to be "only".
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

# Substrings that mark a test function as covering a non-happy-path scenario:
# invalid/error/failure paths, boundary & limit violations, and false-positive
# guards (a look-valid input that must be rejected, a behavior that must NOT
# trigger). Chosen to be distinctive enough not to match ordinary happy names.
NEGATIVE_MARKERS: frozenset[str] = frozenset(
    {
        "reject",
        "invalid",
        "empty",
        "missing",
        "malformed",
        "fail",
        "error",
        "raise",
        "false",
        "negative",
        "boundary",
        "limit",
        "duplicate",
        "conflict",
        "guard",
        "_tie",
        "tied",
        "skip",
        "refuse",
        "forbid",
        "denied",
        "deny",
        "unauthorized",
        "overflow",
        "nonexistent",
        "unknown",
        "corrupt",
        "mismatch",
        "abort",
        "timeout",
        "disallow",
        "does_not",
        "must_not",
        "cannot",
        "never",
        "without",
        "out_of",
        "no_signal",
        "no_op",
        "noop",
        "not_",
        "_not",
        "_no_",
        "bad_",
        "_bad",
        "rejects",
        "ignores",
        "ignored",
        "drops",
        "dropped",
        "clamp",
        "truncat",
        "partial",
        "stale",
        "expired",
        "aborts",
        "warns",
        "differ",
        "diverg",
        "unchanged",
        "non_",
        "undocumented",
        "unseeded",
        "untouched",
        "preserv",
    }
)


def _is_negative_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in NEGATIVE_MARKERS)


@dataclass
class ModuleReport:
    path: str
    test_functions: list[str] = field(default_factory=list)
    negative_tests: list[str] = field(default_factory=list)
    uses_raises: bool = False
    has_negative_assertion: bool = False
    unparsed: bool = False

    @property
    def has_behavior_tests(self) -> bool:
        return bool(self.test_functions)

    @property
    def carries_diversity_signal(self) -> bool:
        return self.uses_raises or self.has_negative_assertion or bool(self.negative_tests)

    @property
    def is_happy_path_only(self) -> bool:
        # Only a module that actually tests behavior can be *only* happy-path.
        if self.unparsed or not self.has_behavior_tests:
            return False
        return not self.carries_diversity_signal


def _iter_test_functions(tree: ast.AST):
    """Yield every ``test_*`` def, whether top-level or a Test-class method."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            yield node


def _uses_raises_or_warns(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"raises", "warns"}:
                return True
    return False


def _is_negative_compare(node: ast.Compare) -> bool:
    # `!=`, `not in`, and `is not` express a "must differ / must be absent /
    # must not be" guard.
    return any(isinstance(op, (ast.NotEq, ast.NotIn, ast.IsNot)) for op in node.ops)


def _has_negative_assertion(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        for sub in ast.walk(node.test):
            if isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.Not):
                return True
            if isinstance(sub, ast.Compare) and _is_negative_compare(sub):
                return True
    return False


def analyze_source(source: str, path: str = "<memory>") -> ModuleReport:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ModuleReport(path=path, unparsed=True)

    test_functions = [fn.name for fn in _iter_test_functions(tree)]
    negative_tests = [name for name in test_functions if _is_negative_name(name)]
    return ModuleReport(
        path=path,
        test_functions=test_functions,
        negative_tests=negative_tests,
        uses_raises=_uses_raises_or_warns(tree),
        has_negative_assertion=_has_negative_assertion(tree),
    )


def analyze_file(path: Path, *, rel_to: Path | None = None) -> ModuleReport:
    display = str(path.relative_to(rel_to)) if rel_to else str(path)
    return analyze_source(path.read_text(), path=display)


def audit_test_tree(root: Path, exempt: set[str] | frozenset[str] = frozenset()) -> list[ModuleReport]:
    """Return the happy-path-only test modules under ``root``, minus exemptions.

    ``exempt`` holds module paths relative to ``root`` (POSIX style) that are
    permitted to assert only a happy path.
    """
    root = Path(root)
    offenders: list[ModuleReport] = []
    for path in sorted(root.rglob("test_*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in exempt:
            continue
        report = analyze_file(path, rel_to=root)
        if report.is_happy_path_only:
            offenders.append(report)
    return offenders
