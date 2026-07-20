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

# Authoring contract: the state-management class must expose both of these.
_STATE_METHODS = ("reset_state", "seed_state")
# HTTP verbs whose decorators mark a function as a route handler.
_ROUTE_VERBS = {"get", "post", "put", "patch", "delete"}


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


def _is_route_handler(node: ast.AST) -> bool:
    """A function decorated with an HTTP-verb route decorator (``@app.post(...)``)."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for deco in node.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        if isinstance(target, ast.Attribute) and target.attr in _ROUTE_VERBS:
            return True
    return False


def _handler_returns(fn: ast.AST) -> list[ast.Return]:
    """Return statements owned by ``fn`` itself — not by any nested def/lambda."""
    found: list[ast.Return] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue  # a nested scope's returns are not this handler's
            if isinstance(child, ast.Return):
                found.append(child)
                continue  # the returned expression itself is not walked
            visit(child)

    visit(fn)
    return found


def _returns_bare_string(ret: ast.Return) -> bool:
    value = ret.value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return True
    return isinstance(value, ast.JoinedStr)  # f-string


def _state_class_status(files: dict[str, str]) -> tuple[bool, bool]:
    """Return (a class exposes both contract methods, seed_state takes a seed)."""
    has_class = seed_ok = False
    for source in files.values():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {
                m.name: m
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if not all(name in methods for name in _STATE_METHODS):
                continue
            has_class = True
            seed_fn = methods["seed_state"]
            params = [a.arg for a in seed_fn.args.posonlyargs + seed_fn.args.args]
            # A seed-driven builder needs a parameter beyond ``self``.
            if len(params) > 1 or seed_fn.args.kwonlyargs or seed_fn.args.kwarg:
                seed_ok = True
    return has_class, seed_ok


def audit_authoring_contract(files: dict[str, str]) -> list[ReviewIssue]:
    """Enforce the generated-environment authoring contract (TASKS.md #9).

    1. State is centralized in a single class exposing ``reset_state()`` and a
       seed-driven ``seed_state(seed)``.
    2. Every route handler returns a typed dict — never a bare string/f-string,
       on the success path or the error path.
    """
    issues: list[ReviewIssue] = []

    has_class, seed_ok = _state_class_status(files)
    if not has_class:
        issues.append(ReviewIssue(
            severity=ReviewSeverity.ERROR, category="state_class_missing",
            message=(
                "No state-management class exposing reset_state() and seed_state(seed); "
                "centralize state in one class with both methods"
            ),
            artifact="main.py",
        ))
    elif not seed_ok:
        issues.append(ReviewIssue(
            severity=ReviewSeverity.ERROR, category="seed_state_signature",
            message="seed_state must accept a seed argument so the universe is reproducible",
            artifact="main.py",
        ))

    for label, source in files.items():
        if not source or not source.strip():
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not _is_route_handler(node):
                continue
            if any(_returns_bare_string(r) for r in _handler_returns(node)):
                issues.append(ReviewIssue(
                    severity=ReviewSeverity.ERROR, category="untyped_return",
                    message=(
                        f"Route handler {node.name!r} returns a bare string; return a "
                        "typed dict (errors too, e.g. {'ok': False, 'error': ...})"
                    ),
                    artifact=label,
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

        issues = audit_determinism(files) + audit_authoring_contract(files)
        report = GenerationReview(
            approved=not any(i.severity == ReviewSeverity.ERROR for i in issues),
            requirements_checked=[
                "No wall-clock access in persisted state",
                "No nondeterministic identifiers",
                "No unseeded randomness",
                "Determinism contract present and reset re-initializes it",
                "State centralized in a class with reset_state()/seed_state(seed)",
                "Route handlers return typed dicts, never bare strings",
            ],
            issues=issues,
        )
        await bus.publish("correctness_report", report)
