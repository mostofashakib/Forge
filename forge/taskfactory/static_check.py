"""Reject drafts that cannot run, in code, before any execution or LLM review."""
from __future__ import annotations

from forge.runtime.tools import RESERVED_PATHS, RESERVED_PREFIX
from forge.taskfactory.profile import EnvironmentProfile
from forge.taskfactory.schemas import (
    DIFFICULTY_STEP_BOUNDS,
    FAMILY_CHECK_KINDS,
    LONG_HORIZON_DIFFICULTY,
    MAX_GOLDEN_STEPS,
    MIN_REFLECTION_POINTS,
    OUTCOME_KINDS,
    Check,
    TaskDraft,
)
from forge.taskfactory.slots import Slot
from forge.taskfactory.state_checks import COMPARE_OPS, collection_rows

BROWSER_TOOLS: dict[str, tuple[str, ...]] = {
    "goto": ("path",),
    "click": ("selector",),
    "fill": ("selector", "value"),
    "select": ("selector", "value"),
    "press": ("key",),
    "check": ("selector",),
}
_DOM_OPS = {"equals", "contains", "exists", "absent"}
_DOM_PROPS = {"text", "value", "attr"}
_URL_OPS = {"equals", "contains"}


def static_problems(draft: TaskDraft, slot: Slot, profile: EnvironmentProfile) -> list[str]:
    """Every reason `draft` cannot fill `slot` in this environment. Empty means it may run."""
    problems = _length_problems(draft, slot)
    problems += _reflection_problems(draft, slot)
    problems += _check_problems(draft, profile)
    family_rules = {"state": _state_problems, "cli": _cli_problems, "browser": _browser_problems}
    problems += family_rules[profile.family](draft, profile)
    # Several checks often share one problem. Each reason is listed once.
    return list(dict.fromkeys(problems))


def _length_problems(draft: TaskDraft, slot: Slot) -> list[str]:
    steps = len(draft.golden)
    low, high = DIFFICULTY_STEP_BOUNDS[slot.difficulty]
    if steps > MAX_GOLDEN_STEPS:
        return [f"golden solution has {steps} steps, more than the {MAX_GOLDEN_STEPS} allowed"]
    if steps < low or (high is not None and steps > high):
        span = f"{low} or more" if high is None else f"{low} to {high}"
        return [f"difficulty {slot.difficulty} needs a golden solution of {span} steps, got {steps}"]
    return []


def _reflection_problems(draft: TaskDraft, slot: Slot) -> list[str]:
    problems = [
        f"reflection point at step {point.step} is outside the golden solution"
        for point in draft.reflection_points
        if not 0 <= point.step < len(draft.golden)
    ]
    if slot.difficulty >= LONG_HORIZON_DIFFICULTY and len(draft.reflection_points) < MIN_REFLECTION_POINTS:
        problems.append(
            f"difficulty {slot.difficulty} needs at least {MIN_REFLECTION_POINTS} reflection points, "
            f"got {len(draft.reflection_points)}"
        )
    return problems


def _check_problems(draft: TaskDraft, profile: EnvironmentProfile) -> list[str]:
    if not draft.checks:
        return ["the task has no checks"]
    allowed = FAMILY_CHECK_KINDS[profile.family]
    problems = [
        f"check kind {check.kind!r} does not apply to {profile.family} environments"
        for check in draft.checks
        if check.kind not in allowed
    ]
    for index, check in enumerate(draft.checks):
        problems += [f"check {index + 1} ({check.kind}): {p}" for p in _missing_fields(check)]
    if not any(check.kind in OUTCOME_KINDS for check in draft.checks):
        problems.append("the task needs at least one outcome check on the final state")
    return problems


def _missing_fields(check: Check) -> list[str]:
    kind = check.kind
    problems: list[str] = []
    if kind in ("record_exists", "record_absent", "record_count"):
        if not check.collection:
            problems.append("needs a collection")
        if kind != "record_count" and not check.match:
            problems.append("needs a match")
        if kind == "record_count" and (check.op not in COMPARE_OPS - {"contains"} or not isinstance(check.value, int)):
            problems.append("needs a numeric op and an integer value")
    elif kind == "value":
        if not check.path:
            problems.append("needs a path")
        if check.op not in COMPARE_OPS:
            problems.append(f"op must be one of {sorted(COMPARE_OPS)}")
    elif kind in ("actions_called", "actions_not_called"):
        if not check.tools:
            problems.append("needs tools")
    elif kind == "shell":
        if not (check.command or "").strip():
            problems.append("needs a command")
    elif kind == "dom":
        problems += _dom_missing(check)
    elif kind == "url":
        if check.op not in _URL_OPS or not isinstance(check.value, str):
            problems.append("needs op equals or contains and a string value")
    return problems


def _dom_missing(check: Check) -> list[str]:
    problems = []
    if not check.selector:
        problems.append("needs a selector")
    if check.op not in _DOM_OPS:
        problems.append(f"op must be one of {sorted(_DOM_OPS)}")
    if (check.prop or "text") not in _DOM_PROPS:
        problems.append(f"prop must be one of {sorted(_DOM_PROPS)}")
    if check.prop == "attr" and not check.attr:
        problems.append("needs an attr name")
    if check.op in ("equals", "contains") and not isinstance(check.value, str):
        problems.append("needs a string value")
    return problems


def is_control_plane(tool: str) -> bool:
    return tool.startswith(RESERVED_PREFIX) or tool.rstrip("/") == "/forge" or tool in RESERVED_PATHS


def _state_problems(draft: TaskDraft, profile: EnvironmentProfile) -> list[str]:
    problems = []
    if draft.seed.setup or draft.seed.pages:
        problems.append("state environments take seed records, not setup commands or pages")
    for step in draft.golden:
        if is_control_plane(step.tool):
            problems.append(f"golden step {step.tool!r} targets the control plane")
        elif not profile.has_tool(step.tool):
            problems.append(f"golden step uses {step.tool!r}, which the environment does not have")
    for check in draft.checks:
        for tool in check.tools or []:
            if not profile.has_tool(tool):
                problems.append(f"check names {tool!r}, which the environment does not have")
    if profile.state_sample is not None:
        problems += [
            f"seed collection {path!r} does not exist in the environment"
            for path in draft.seed.records
            if not _is_collection(profile.state_sample, path)
        ]
        problems += _observed_path_problems(draft, profile.state_sample)
    return problems


def _is_collection(state: dict, path: str) -> bool:
    try:
        collection_rows(state, path)
    except (KeyError, TypeError):
        return False
    return True


def _observed_path_problems(draft: TaskDraft, state: dict) -> list[str]:
    problems = []
    for check in draft.checks:
        if check.collection and not _is_collection(state, check.collection):
            problems.append(f"check reads collection {check.collection!r}, which the state does not have")
        if check.kind == "value" and check.path and check.path.split(".")[0] not in state:
            problems.append(f"check reads {check.path!r}, which the state does not have")
    return problems


def _cli_problems(draft: TaskDraft, profile: EnvironmentProfile) -> list[str]:
    problems = []
    if draft.seed.records or draft.seed.pages:
        problems.append("CLI environments take setup commands, not records or pages")
    for index, step in enumerate(draft.golden):
        if step.tool != "shell" or not str(step.args.get("command", "")).strip():
            problems.append(f"golden step {index + 1} must be a shell step with a command")
    return problems


def _browser_problems(draft: TaskDraft, profile: EnvironmentProfile) -> list[str]:
    seed = draft.seed
    problems = []
    if seed.records or seed.setup:
        problems.append("browser environments take pages, not records or setup commands")
    paths = [page.path for page in seed.pages]
    if not paths:
        problems.append("the task needs at least one page")
    if len(set(paths)) != len(paths):
        problems.append("page paths must be unique")
    problems += [f"page path {p!r} must start with /" for p in paths if not p.startswith("/")]
    if seed.start_path not in paths:
        problems.append(f"start page {seed.start_path!r} is not one of the task's pages")
    for index, step in enumerate(draft.golden):
        needed = BROWSER_TOOLS.get(step.tool)
        if needed is None:
            problems.append(f"golden step {index + 1} uses {step.tool!r}; use one of {sorted(BROWSER_TOOLS)}")
            continue
        problems += [
            f"golden step {index + 1} ({step.tool}) needs {arg!r}"
            for arg in needed
            if not str(step.args.get(arg, "")).strip()
        ]
        if step.tool == "goto" and step.args.get("path") and step.args["path"] not in paths:
            problems.append(f"golden step {index + 1} goes to {step.args['path']!r}, which is not one of the task's pages")
    return problems
