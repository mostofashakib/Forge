from __future__ import annotations

import ast

from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.agents.reviewer import GenerationReview, ReviewIssue, ReviewSeverity
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext

# Calls whose dotted path ends with one of these are wall-clock reads.
_WALL_CLOCK_SUFFIXES = (
    ".now", ".utcnow", ".today", ".time", ".monotonic", ".perf_counter",
)
# Roots/paths that produce nondeterministic identifiers.
_NONDET_ID_SUFFIXES = (".uuid1", ".uuid3", ".uuid4", ".uuid5", ".urandom")
_NONDET_ID_ROOTS = ("secrets",)
# Functions whose bodies are telemetry envelopes — exempt from the audit.
_TELEMETRY_EMITTERS = {"emit_event", "emit", "_emit_event", "record_event"}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _root(path: str) -> str:
    return path.split(".", 1)[0] if path else ""


class _DeterminismVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.wall_clock: list[str] = []
        self.nondet_id: list[str] = []
        self.random_calls: list[str] = []
        self.random_seeded = False
        self._emitter_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        in_emitter = node.name in _TELEMETRY_EMITTERS
        if in_emitter:
            self._emitter_depth += 1
        self.generic_visit(node)
        if in_emitter:
            self._emitter_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        path = _dotted(node.func)
        if path:
            if path.endswith(".seed") and _root(path) == "random":
                self.random_seeded = True
            elif _root(path) == "random" and not path.endswith(".Random"):
                if self._emitter_depth == 0:
                    self.random_calls.append(path)
            elif self._emitter_depth == 0:
                if path.endswith(_WALL_CLOCK_SUFFIXES):
                    self.wall_clock.append(path)
                elif path.endswith(_NONDET_ID_SUFFIXES) or _root(path) in _NONDET_ID_ROOTS:
                    self.nondet_id.append(path)
        self.generic_visit(node)


def _contract(files: dict[str, str]) -> tuple[bool, bool]:
    """Return (has_clock_and_id_helpers, reset_reinitializes)."""
    has_forge_now = has_next_id = reset_reinit = False
    for source in files.values():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "forge_now":
                    has_forge_now = True
                if node.name == "_next_id":
                    has_next_id = True
                if "reset" in node.name.lower():
                    body = ast.unparse(node)
                    if "_FORGE_CLOCK" in body and (
                        "_ID_COUNTERS" in body or "_next_id" in body
                    ):
                        reset_reinit = True
    return (has_forge_now and has_next_id), reset_reinit


def audit_determinism(files: dict[str, str]) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    for label, source in files.items():
        if not source or not source.strip():
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue  # syntax is the reviewer's job, not ours
        visitor = _DeterminismVisitor()
        visitor.visit(tree)
        for path in visitor.wall_clock:
            issues.append(ReviewIssue(
                severity=ReviewSeverity.ERROR, category="wall_clock",
                message=f"Wall-clock access {path!r}; use forge_now() instead", artifact=label,
            ))
        for path in visitor.nondet_id:
            issues.append(ReviewIssue(
                severity=ReviewSeverity.ERROR, category="nondeterministic_id",
                message=f"Nondeterministic identifier {path!r}; use _next_id() instead",
                artifact=label,
            ))
        if visitor.random_calls and not visitor.random_seeded:
            issues.append(ReviewIssue(
                severity=ReviewSeverity.ERROR, category="unseeded_randomness",
                message="Unseeded random.* usage; seed the RNG or use deterministic values",
                artifact=label,
            ))

    has_helpers, reset_reinit = _contract(files)
    if not has_helpers:
        issues.append(ReviewIssue(
            severity=ReviewSeverity.ERROR, category="contract_missing",
            message="Determinism contract absent: define forge_now() and _next_id()",
            artifact="main.py",
        ))
    elif not reset_reinit:
        issues.append(ReviewIssue(
            severity=ReviewSeverity.ERROR, category="reset_not_reinitialized",
            message="/forge/reset must reset _FORGE_CLOCK and _ID_COUNTERS before re-seeding",
            artifact="main.py",
        ))
    return issues


class EnvironmentCorrectnessAgent(EnvGenAgent):
    """Static determinism/reproducibility gate for generated code."""

    agent_id = "correctness_reviewer"
    depends_on = ["app_code", "instrumented_code", "state_bridge_code", "reward_fn_code"]
    produces = ["correctness_report"]

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        app_code: dict[str, str] = await bus.wait_for("app_code") or {}
        instrumented: dict[str, str] = await bus.wait_for("instrumented_code") or {}
        state_bridge: str = await bus.wait_for("state_bridge_code") or ""
        reward_fn: str = await bus.wait_for("reward_fn_code") or ""

        files: dict[str, str] = {
            path: content for path, content in app_code.items() if path.endswith(".py")
        }
        for path, content in instrumented.items():
            if path.endswith(".py"):
                files[f"instrumented:{path}"] = content
        files["state_bridge_code"] = state_bridge
        files["reward_fn_code"] = reward_fn

        issues = audit_determinism(files)
        report = GenerationReview(
            approved=not any(i.severity == ReviewSeverity.ERROR for i in issues),
            requirements_checked=[
                "No wall-clock access in persisted state",
                "No nondeterministic identifiers",
                "No unseeded randomness",
                "Determinism contract present and reset re-initializes it",
            ],
            issues=issues,
        )
        await bus.publish("correctness_report", report)
