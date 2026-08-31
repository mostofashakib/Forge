# Forge Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `forge/contracts/` — eleven ABCs plus an `Environment` facade covering the concerns every Forge environment family solves independently today — and rebase the runtime, the four environment families, the code generators, and the envgen specialists onto them.

**Architecture:** Eleven independent ABCs, one per concern, each in its own module, re-exported from `forge/contracts/__init__.py`. Shared data types live in `forge/contracts/types.py`. The `Environment` facade composes ten of them as properties (seven required, three optional); `EpisodeController` is deliberately excluded because it drives an environment from outside. `forge/contracts/` has no runtime import edge to `forge/runtime/`, `forge/envgen/`, `forge/extraction/`, or `backend/` — the three runtime types it needs in signatures (`RuntimeContext`, `Trajectory`, `TransitionResult`) are imported under `TYPE_CHECKING` with string annotations.

**Tech Stack:** Python 3.12+, `abc.ABC`, pydantic v2, gymnasium, pytest + pytest-asyncio. Run tests with `.venv/bin/python -m pytest`.

**Spec:** `specs/2026-08-30-forge-contracts-design.md`

## Global Constraints

- **Import direction is absolute.** `forge/contracts/` may import from `forge/schema/`, the standard library, and pydantic at runtime. Any runtime import from `forge/runtime/`, `forge/envgen/`, `forge/extraction/`, or `backend/` is a defect. `TYPE_CHECKING` imports from those packages are permitted.
- **No behavior change.** Every existing test must pass unchanged. The only deliberate signature tightenings are `InitialStateProvider.reset` taking an explicit `seed` keyword, and `CliEpisodeRunner`/`BrowserEpisodeRunner` `run_episode` gaining an accepted-and-ignored `seed` keyword.
- **Moved types re-export.** Every type moved out of its current module is re-exported from the old location so existing imports keep working.
- **Test diversity is required.** Per the project standard, every task's tests must include a negative case and a false-positive guard, not only the happy path.
- **Commit message style.** No Claude attribution, no `Co-Authored-By` trailer.
- **Test command.** `.venv/bin/python -m pytest` from the repo root. Full suite baseline before starting: 1265 passed.

---

# Phase 1 — The contracts package

Phase 1 adds `forge/contracts/` and changes no existing class. It ends with the package importable, the facade defined, and the conformance harness in place.

### Task 1: Package skeleton, shared types, and the import-direction guard

**Files:**
- Create: `forge/contracts/__init__.py`
- Create: `forge/contracts/types.py`
- Create: `tests/contracts/__init__.py`
- Create: `tests/contracts/test_import_direction.py`
- Create: `tests/contracts/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Task`, `Observation`, `Action`, `ActionResult`, `Termination`, `StepOutcome`, `ToolParam`, `ToolSpec`, `RewardComponent`, `RewardBreakdown`, `CheckResult`, `VerificationResult`, `AgentAdapter` — all importable from `forge.contracts`.

- [ ] **Step 1: Write the failing import-direction test**

```python
# tests/contracts/test_import_direction.py
"""forge/contracts/ must not depend on the packages that depend on it."""
from __future__ import annotations

import ast
import pathlib

import pytest

CONTRACTS = pathlib.Path(__file__).resolve().parents[2] / "forge" / "contracts"
FORBIDDEN = ("forge.runtime", "forge.envgen", "forge.extraction", "backend")


def _runtime_imports(source: str) -> list[str]:
    """Module names imported at runtime, ignoring `if TYPE_CHECKING:` blocks."""
    tree = ast.parse(source)
    type_checking_blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and (
            (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
            or (isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING")
        )
    ]
    guarded = {id(child) for block in type_checking_blocks for child in ast.walk(block)}

    names: list[str] = []
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize(
    "path", sorted(CONTRACTS.rglob("*.py")), ids=lambda p: p.name
)
def test_contracts_module_has_no_forbidden_runtime_import(path):
    for name in _runtime_imports(path.read_text()):
        assert not name.startswith(FORBIDDEN), (
            f"{path.name} imports {name!r} at runtime; contracts/ must not "
            f"depend on {FORBIDDEN}. Move it under `if TYPE_CHECKING:`."
        )


def test_type_checking_imports_are_allowed():
    """False-positive guard: the guard must not reject a TYPE_CHECKING import."""
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from forge.runtime.context import RuntimeContext\n"
    )
    assert _runtime_imports(source) == ["typing"]


def test_plain_forbidden_import_is_detected():
    """Negative: an unguarded forbidden import must be caught."""
    source = "from forge.runtime.context import RuntimeContext\n"
    assert any(n.startswith(FORBIDDEN) for n in _runtime_imports(source))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/contracts/test_import_direction.py -v`
Expected: FAIL — collection error, `forge/contracts` does not exist so `CONTRACTS.rglob` yields nothing and the parametrize list is empty. `test_type_checking_imports_are_allowed` and `test_plain_forbidden_import_is_detected` should already pass; only the parametrized test is empty. Create the package in step 3 to give it something to check.

- [ ] **Step 3: Write `forge/contracts/types.py`**

```python
# forge/contracts/types.py
"""Data types the contract interfaces speak in.

These are shapes, not behavior. They live here rather than in forge/runtime/
so that both the in-process and container environment families can depend on
them without either depending on the other.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """One problem the model should solve.

    Unifies the compiler's TaskTemplate and envgen's Scenario. `objective` is
    the natural-language goal; CLI and browser environments carry only that.
    """

    id: str
    objective: str
    seed: int | None = None
    success_conditions: list[dict] = Field(default_factory=list)
    failure_conditions: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Actions and observations
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """One action the model takes.

    The runtime's public surface still accepts plain dicts; engines convert at
    the boundary via `from_dict` so handler code receives a typed value.
    """

    type: str
    params: dict = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "Action":
        params = {k: v for k, v in raw.items() if k != "type"}
        return cls(type=raw["type"], params=params)

    def to_dict(self) -> dict:
        return {"type": self.type, **self.params}


class ActionResult(BaseModel):
    """What an execution backend returns after running one action."""

    state: dict
    events: list[dict] = Field(default_factory=list)
    error: dict | None = None


class Observation(BaseModel):
    """What the model sees back after an action.

    Carries all three shapes the families produce: a structured payload (gym
    dict, /forge/state), rendered text, and typed blocks for tool output.
    """

    payload: dict = Field(default_factory=dict)
    text: str | None = None
    blocks: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

class StepOutcome(BaseModel):
    """Everything a termination policy is allowed to decide on."""

    step_index: int
    score: float = 0.0
    reward: float = 0.0
    state_hash: str | None = None
    verifier_results: list["VerificationResult"] = Field(default_factory=list)


class Termination(BaseModel):
    """A decision to end the episode."""

    reason: str
    truncated: bool = False


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class ToolParam(BaseModel):
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True


class ToolSpec(BaseModel):
    """Schema describing one tool an agent may call — the env's tool surface."""

    name: str
    description: str = ""
    params: list[ToolParam] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Verification and reward
# ---------------------------------------------------------------------------

class CheckResult(BaseModel):
    name: str
    passed: bool
    score: float
    evidence: str | None = None


class VerificationResult(BaseModel):
    verifier_id: str
    passed: bool
    score: float
    checks: list[CheckResult]
    explanation: str = ""

    @classmethod
    def from_checks(
        cls, verifier_id: str, checks: list[CheckResult]
    ) -> "VerificationResult":
        passed = all(c.passed for c in checks)
        score = sum(c.score for c in checks) / len(checks) if checks else 0.0
        return cls(verifier_id=verifier_id, passed=passed, score=score, checks=checks)


class RewardComponent(BaseModel):
    name: str
    value: float


class RewardBreakdown(BaseModel):
    total_reward: float
    components: list[RewardComponent]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

@runtime_checkable
class AgentAdapter(Protocol):
    """Anything that can pick an action given an observation."""

    def act(self, obs: dict, action_types: frozenset[str]) -> dict: ...


StepOutcome.model_rebuild()
```

- [ ] **Step 4: Write `forge/contracts/__init__.py`**

```python
# forge/contracts/__init__.py
"""Interface contracts every Forge environment implements.

Eleven concerns recur across all four environment families (in-process,
container, CLI, browser). Each is one ABC here; the `Environment` facade
composes the ten that describe an environment's state and behavior.
`EpisodeController` is not part of the facade because it drives an environment
from the outside — the same environment may be run by different controllers.
"""
from forge.contracts.types import (
    Action,
    ActionResult,
    AgentAdapter,
    CheckResult,
    Observation,
    RewardBreakdown,
    RewardComponent,
    StepOutcome,
    Task,
    Termination,
    ToolParam,
    ToolSpec,
    VerificationResult,
)

__all__ = [
    "Action",
    "ActionResult",
    "AgentAdapter",
    "CheckResult",
    "Observation",
    "RewardBreakdown",
    "RewardComponent",
    "StepOutcome",
    "Task",
    "Termination",
    "ToolParam",
    "ToolSpec",
    "VerificationResult",
]
```

- [ ] **Step 5: Write the types test**

```python
# tests/contracts/test_types.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge.contracts import (
    Action,
    CheckResult,
    Observation,
    Task,
    Termination,
    VerificationResult,
)


def test_action_round_trips_through_a_plain_dict():
    action = Action.from_dict({"type": "close_ticket", "ticket_id": "t_1"})
    assert action.type == "close_ticket"
    assert action.params == {"ticket_id": "t_1"}
    assert action.to_dict() == {"type": "close_ticket", "ticket_id": "t_1"}


def test_action_requires_a_type():
    # Negative: an action with no type is not an action.
    with pytest.raises(KeyError):
        Action.from_dict({"ticket_id": "t_1"})


def test_observation_defaults_are_empty_not_none():
    # False-positive guard: an observation with no text still has a usable
    # payload and blocks, so consumers never branch on None.
    obs = Observation()
    assert obs.payload == {}
    assert obs.blocks == []
    assert obs.text is None


def test_verification_result_from_checks_averages_scores():
    result = VerificationResult.from_checks(
        "v1",
        [
            CheckResult(name="a", passed=True, score=1.0),
            CheckResult(name="b", passed=False, score=0.0),
        ],
    )
    assert result.passed is False
    assert result.score == 0.5


def test_verification_result_from_no_checks_does_not_pass_vacuously():
    # Negative: zero checks must not average to a passing score.
    result = VerificationResult.from_checks("v1", [])
    assert result.score == 0.0


def test_task_rejects_a_missing_objective():
    with pytest.raises(ValidationError):
        Task(id="t1")


def test_termination_defaults_to_not_truncated():
    assert Termination(reason="success").truncated is False
```

- [ ] **Step 6: Run both tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/contracts/ -v`
Expected: PASS. The import-direction test now parametrizes over `types.py` and `__init__.py` and finds no forbidden imports.

- [ ] **Step 7: Commit**

```bash
git add forge/contracts tests/contracts
git commit -m "Add forge/contracts package with shared contract data types

Types the eleven interfaces speak in, placed where both the in-process and
container environment families can depend on them without either depending on
the other. An AST-based test enforces the import direction, permitting
TYPE_CHECKING imports and rejecting runtime ones."
```

---

### Task 2: `StateManager` and `TerminationPolicy`

**Files:**
- Create: `forge/contracts/state.py`
- Create: `forge/contracts/termination.py`
- Modify: `forge/contracts/__init__.py`
- Create: `tests/contracts/test_state_contract.py`
- Create: `tests/contracts/test_termination_contract.py`

**Interfaces:**
- Consumes: `StepOutcome`, `Termination` from `forge.contracts.types`.
- Produces: `StateManager` with `get() -> dict`, `apply(state: dict) -> None`, `hash() -> str`, `snapshot(slot: str) -> None`, `restore(slot: str) -> None`. `TerminationPolicy` with `check(outcome: StepOutcome) -> Termination | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/contracts/test_state_contract.py
from __future__ import annotations

import pytest

from forge.contracts import StateManager


class _Conforming(StateManager):
    def __init__(self) -> None:
        self._state: dict = {}

    def get(self) -> dict:
        return dict(self._state)

    def apply(self, state: dict) -> None:
        self._state = dict(state)

    def hash(self) -> str:
        return f"sha256:{len(self._state)}"


def test_a_conforming_state_manager_instantiates():
    manager = _Conforming()
    manager.apply({"a": 1})
    assert manager.get() == {"a": 1}


def test_a_state_manager_missing_a_method_cannot_be_instantiated():
    # Negative: the failure must land at instantiation, not at first call.
    class Incomplete(StateManager):
        def get(self) -> dict:
            return {}

        def apply(self, state: dict) -> None:
            return None

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_snapshot_slots_are_optional_and_fail_loudly():
    # False-positive guard: only the container family supports slots, so
    # snapshot/restore are concrete and must raise rather than silently no-op.
    manager = _Conforming()
    with pytest.raises(NotImplementedError):
        manager.snapshot("slot_a")
    with pytest.raises(NotImplementedError):
        manager.restore("slot_a")
```

```python
# tests/contracts/test_termination_contract.py
from __future__ import annotations

import pytest

from forge.contracts import StepOutcome, Termination, TerminationPolicy


class _StopAtScore(TerminationPolicy):
    def check(self, outcome: StepOutcome) -> Termination | None:
        if outcome.score >= 0.9:
            return Termination(reason="success")
        return None


def test_a_policy_returns_none_to_continue():
    assert _StopAtScore().check(StepOutcome(step_index=0, score=0.1)) is None


def test_a_policy_returns_a_termination_to_stop():
    decision = _StopAtScore().check(StepOutcome(step_index=3, score=0.95))
    assert decision is not None
    assert decision.reason == "success"
    assert decision.truncated is False


def test_a_policy_missing_check_cannot_be_instantiated():
    # Negative: an incomplete policy fails at instantiation.
    class Incomplete(TerminationPolicy):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/contracts/test_state_contract.py tests/contracts/test_termination_contract.py -v`
Expected: FAIL with `ImportError: cannot import name 'StateManager' from 'forge.contracts'`

- [ ] **Step 3: Write the two contracts**

```python
# forge/contracts/state.py
"""How state is tracked across turns."""
from __future__ import annotations

from abc import ABC, abstractmethod


class StateManager(ABC):
    """Owns the environment's state and its content-addressed hash.

    `snapshot`/`restore` are concrete because only the container family
    supports named slots today (POST /forge/snapshot, POST /forge/restore/{slot}).
    An implementation that does not support them inherits a loud failure rather
    than a silent no-op.
    """

    @abstractmethod
    def get(self) -> dict:
        """Current state. Implementations return a copy, never a live reference."""

    @abstractmethod
    def apply(self, state: dict) -> None:
        """Replace the current state."""

    @abstractmethod
    def hash(self) -> str:
        """Stable content hash of the current state, prefixed with its algorithm."""

    def snapshot(self, slot: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support named state slots"
        )

    def restore(self, slot: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support named state slots"
        )
```

```python
# forge/contracts/termination.py
"""How an episode ends."""
from __future__ import annotations

from abc import ABC, abstractmethod

from forge.contracts.types import StepOutcome, Termination


class TerminationPolicy(ABC):
    """Decides, after each step, whether the episode is over.

    Returning None means continue. Policies are consulted in order by the
    controller, so each one answers only about its own stopping condition.
    """

    @abstractmethod
    def check(self, outcome: StepOutcome) -> Termination | None:
        """Return a Termination to stop, or None to continue."""
```

- [ ] **Step 4: Export them from `forge/contracts/__init__.py`**

Add to the imports and to `__all__`:

```python
from forge.contracts.state import StateManager
from forge.contracts.termination import TerminationPolicy
```

Add `"StateManager"` and `"TerminationPolicy"` to `__all__`, keeping it alphabetically sorted.

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/contracts/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add forge/contracts tests/contracts
git commit -m "Add StateManager and TerminationPolicy contracts

Named state slots stay concrete-but-raising rather than abstract, because only
the container family supports them; an in-process manager should not have to
stub a method it has no meaning for."
```

---

### Task 3: `TaskSource` and `InitialStateProvider`

**Files:**
- Create: `forge/contracts/dataset.py`
- Create: `forge/contracts/initial_state.py`
- Modify: `forge/contracts/__init__.py`
- Create: `tests/contracts/test_dataset_contract.py`
- Create: `tests/contracts/test_initial_state_contract.py`

**Interfaces:**
- Consumes: `Task` from `forge.contracts.types`.
- Produces: `TaskSource` with `tasks() -> Sequence[Task]` and `get(task_id: str) -> Task`. `InitialStateProvider` with `reset(ctx, *, seed: int | None, options: Mapping[str, object]) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/contracts/test_dataset_contract.py
from __future__ import annotations

from collections.abc import Sequence

import pytest

from forge.contracts import Task, TaskSource


class _Fixed(TaskSource):
    def __init__(self, tasks: list[Task]) -> None:
        self._tasks = tasks

    def tasks(self) -> Sequence[Task]:
        return list(self._tasks)

    def get(self, task_id: str) -> Task:
        for task in self._tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)


def test_a_task_source_lists_and_looks_up_tasks():
    source = _Fixed([Task(id="t1", objective="close the ticket")])
    assert [t.id for t in source.tasks()] == ["t1"]
    assert source.get("t1").objective == "close the ticket"


def test_an_unknown_task_id_raises():
    # Negative: a miss must raise, not return a default task.
    source = _Fixed([Task(id="t1", objective="close the ticket")])
    with pytest.raises(KeyError):
        source.get("nope")


def test_an_empty_task_source_stays_empty():
    # False-positive guard: no tasks means no tasks, not a synthesized one.
    assert list(_Fixed([]).tasks()) == []


def test_a_task_source_missing_get_cannot_be_instantiated():
    class Incomplete(TaskSource):
        def tasks(self) -> Sequence[Task]:
            return []

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
```

```python
# tests/contracts/test_initial_state_contract.py
from __future__ import annotations

from collections.abc import Mapping

import pytest

from forge.contracts import InitialStateProvider


class _Seeded(InitialStateProvider):
    def reset(self, ctx, *, seed: int | None, options: Mapping[str, object]) -> dict:
        return {"seed": seed, "options": dict(options)}


def test_the_seed_is_an_explicit_keyword_not_smuggled_in_options():
    state = _Seeded().reset(None, seed=7, options={})
    assert state["seed"] == 7
    assert state["options"] == {}


def test_an_unseeded_reset_is_representable():
    # False-positive guard: seed=None is a valid, distinct request for the
    # provider's fixed baseline — it must not be confused with seed=0.
    assert _Seeded().reset(None, seed=None, options={})["seed"] is None


def test_seed_must_be_passed_by_keyword():
    # Negative: positional seed is rejected, so call sites cannot drift.
    with pytest.raises(TypeError):
        _Seeded().reset(None, 7, {})


def test_a_provider_missing_reset_cannot_be_instantiated():
    class Incomplete(InitialStateProvider):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/contracts/test_dataset_contract.py tests/contracts/test_initial_state_contract.py -v`
Expected: FAIL with `ImportError: cannot import name 'TaskSource' from 'forge.contracts'`

- [ ] **Step 3: Write the two contracts**

```python
# forge/contracts/dataset.py
"""What problems the model should solve."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from forge.contracts.types import Task


class TaskSource(ABC):
    """The set of tasks an environment can pose.

    Backed by compiler TaskTemplates, an envgen ScenarioSuite, or a single
    natural-language objective, depending on the family.
    """

    @abstractmethod
    def tasks(self) -> Sequence[Task]:
        """Every task this source can pose, in a stable order."""

    @abstractmethod
    def get(self, task_id: str) -> Task:
        """One task by id. Raises KeyError when the id is unknown."""
```

```python
# forge/contracts/initial_state.py
"""How per-episode state is set up at the start of a rollout."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.runtime.context import RuntimeContext


class InitialStateProvider(ABC):
    """Produces the starting state for one episode.

    `seed` is an explicit keyword rather than an entry in `options` because
    call sites previously disagreed about where it lived — some passed it in
    `options`, others read `ctx.seed`. A seed of None means the provider's
    fixed baseline, which is distinct from seed 0.
    """

    @abstractmethod
    def reset(
        self,
        ctx: "RuntimeContext",
        *,
        seed: int | None,
        options: Mapping[str, object],
    ) -> dict:
        """Return the initial state for an episode."""
```

- [ ] **Step 4: Export from `forge/contracts/__init__.py`**

```python
from forge.contracts.dataset import TaskSource
from forge.contracts.initial_state import InitialStateProvider
```

Add `"InitialStateProvider"` and `"TaskSource"` to `__all__`.

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/contracts/ -v`
Expected: PASS, including the import-direction test — `initial_state.py` imports `RuntimeContext` under `TYPE_CHECKING`, which the guard permits.

- [ ] **Step 6: Commit**

```bash
git add forge/contracts tests/contracts
git commit -m "Add TaskSource and InitialStateProvider contracts

InitialStateProvider.reset takes seed as an explicit keyword. Call sites
previously disagreed about whether the seed lived in options or on the context,
which made an unseeded reset ambiguous with a zero-seeded one."
```

---

### Task 4: `PromptTemplate`, `ToolProvider`, and `ObservationEncoder`

**Files:**
- Create: `forge/contracts/prompting.py`
- Create: `forge/contracts/tools.py`
- Create: `forge/contracts/observation.py`
- Modify: `forge/contracts/__init__.py`
- Create: `tests/contracts/test_prompting_contract.py`
- Create: `tests/contracts/test_tools_contract.py`
- Create: `tests/contracts/test_observation_contract.py`

**Interfaces:**
- Consumes: `Task`, `Observation`, `ToolSpec` from `forge.contracts.types`.
- Produces: `PromptTemplate` with `system(task: Task) -> str`, `user(observation: Observation, task: Task) -> str`, `tool_descriptions(tools: Sequence[ToolSpec]) -> list[dict]`. `ToolProvider` with `tools() -> Sequence[ToolSpec]`. `ObservationEncoder` with `encode(state: dict, ctx) -> Observation`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/contracts/test_prompting_contract.py
from __future__ import annotations

from collections.abc import Sequence

import pytest

from forge.contracts import Observation, PromptTemplate, Task, ToolSpec


class _Minimal(PromptTemplate):
    def system(self, task: Task) -> str:
        return f"Goal: {task.objective}"

    def user(self, observation: Observation, task: Task) -> str:
        return observation.text or str(observation.payload)

    def tool_descriptions(self, tools: Sequence[ToolSpec]) -> list[dict]:
        return [{"name": t.name, "description": t.description} for t in tools]


def test_a_template_renders_system_user_and_tools():
    template = _Minimal()
    task = Task(id="t1", objective="close the ticket")
    assert template.system(task) == "Goal: close the ticket"
    assert template.user(Observation(text="state"), task) == "state"
    assert template.tool_descriptions([ToolSpec(name="close")]) == [
        {"name": "close", "description": ""}
    ]


def test_no_tools_renders_no_descriptions():
    # False-positive guard: an env with no tools must not get a placeholder tool.
    assert _Minimal().tool_descriptions([]) == []


def test_a_template_missing_a_method_cannot_be_instantiated():
    class Incomplete(PromptTemplate):
        def system(self, task: Task) -> str:
            return ""

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
```

```python
# tests/contracts/test_tools_contract.py
from __future__ import annotations

from collections.abc import Sequence

import pytest

from forge.contracts import ToolProvider, ToolSpec


class _Static(ToolProvider):
    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = tools

    def tools(self) -> Sequence[ToolSpec]:
        return list(self._tools)


def test_a_provider_lists_its_tools():
    provider = _Static([ToolSpec(name="close_ticket")])
    assert [t.name for t in provider.tools()] == ["close_ticket"]


def test_a_provider_with_no_tools_stays_empty():
    # False-positive guard: a shell environment genuinely has no tool schema.
    assert list(_Static([]).tools()) == []


def test_a_provider_missing_tools_cannot_be_instantiated():
    class Incomplete(ToolProvider):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
```

```python
# tests/contracts/test_observation_contract.py
from __future__ import annotations

import pytest

from forge.contracts import Observation, ObservationEncoder


class _Passthrough(ObservationEncoder):
    def encode(self, state: dict, ctx) -> Observation:
        return Observation(payload=state)


def test_an_encoder_wraps_state_in_an_observation():
    obs = _Passthrough().encode({"tickets": []}, None)
    assert obs.payload == {"tickets": []}


def test_an_encoder_returns_an_observation_not_a_dict():
    # Negative: consumers rely on the typed shape; a bare dict would break them.
    assert isinstance(_Passthrough().encode({}, None), Observation)


def test_an_encoder_missing_encode_cannot_be_instantiated():
    class Incomplete(ObservationEncoder):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/contracts/test_prompting_contract.py tests/contracts/test_tools_contract.py tests/contracts/test_observation_contract.py -v`
Expected: FAIL with `ImportError: cannot import name 'PromptTemplate' from 'forge.contracts'`

- [ ] **Step 3: Write the three contracts**

```python
# forge/contracts/prompting.py
"""How the task is presented to the model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from forge.contracts.types import Observation, Task, ToolSpec


class PromptTemplate(ABC):
    """Renders the text an LLM agent sees each turn.

    Optional on the Environment facade: an environment driven by a trainer that
    supplies its own prompting has no template of its own.
    """

    @abstractmethod
    def system(self, task: Task) -> str:
        """The system prompt for this task."""

    @abstractmethod
    def user(self, observation: Observation, task: Task) -> str:
        """The per-turn user message carrying the current observation."""

    @abstractmethod
    def tool_descriptions(self, tools: Sequence[ToolSpec]) -> list[dict]:
        """Provider-agnostic tool descriptions for the given tool surface."""
```

```python
# forge/contracts/tools.py
"""What the model can do in the world."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from forge.contracts.types import ToolSpec


class ToolProvider(ABC):
    """The set of tools an environment exposes to the agent.

    Optional on the Environment facade: a shell environment exposes a command
    line rather than a tool schema.
    """

    @abstractmethod
    def tools(self) -> Sequence[ToolSpec]:
        """Every tool the agent may call, in a stable order."""
```

```python
# forge/contracts/observation.py
"""What the model sees back after an action."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from forge.contracts.types import Observation

if TYPE_CHECKING:
    from forge.runtime.context import RuntimeContext


class ObservationEncoder(ABC):
    """Turns raw environment state into what the agent is allowed to see.

    This is the seam where redaction and role-based filtering belong: the
    encoder decides what leaves the environment, so a filter cannot be bypassed
    by reading state directly.
    """

    @abstractmethod
    def encode(self, state: dict, ctx: "RuntimeContext") -> Observation:
        """Render state as an Observation."""
```

- [ ] **Step 4: Export from `forge/contracts/__init__.py`**

```python
from forge.contracts.observation import ObservationEncoder
from forge.contracts.prompting import PromptTemplate
from forge.contracts.tools import ToolProvider
```

Add `"ObservationEncoder"`, `"PromptTemplate"`, `"ToolProvider"` to `__all__`.

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/contracts/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add forge/contracts tests/contracts
git commit -m "Add PromptTemplate, ToolProvider, and ObservationEncoder contracts

ObservationEncoder is the single seam through which state leaves an
environment, so role-based filtering cannot be bypassed by reading state
directly."
```

---

### Task 5: `ExecutionBackend`, `TransitionHandler`, and `Transport`

**Files:**
- Create: `forge/contracts/backend.py`
- Create: `forge/contracts/transport.py`
- Modify: `forge/contracts/__init__.py`
- Create: `tests/contracts/test_backend_contract.py`
- Create: `tests/contracts/test_transport_contract.py`

**Interfaces:**
- Consumes: `Action`, `ActionResult` from `forge.contracts.types`.
- Produces: `TransitionHandler` with `apply(state: dict, action: Action, ctx) -> TransitionResult`. `ExecutionBackend` with `execute(action: Action, state: dict, ctx) -> ActionResult` and `close() -> None`. `Transport` with `call(request: TransportRequest) -> TransportResponse` and `close() -> None`. `TransportRequest` fields `method`, `target`, `payload`, `timeout`. `TransportResponse` fields `status`, `body`, `error`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/contracts/test_backend_contract.py
from __future__ import annotations

import pytest

from forge.contracts import Action, ActionResult, ExecutionBackend, TransitionHandler


class _Echo(ExecutionBackend):
    def execute(self, action: Action, state: dict, ctx) -> ActionResult:
        return ActionResult(
            state={**state, "last": action.type},
            events=[{"type": f"{action.type}_done"}],
        )


def test_a_backend_returns_new_state_and_events():
    result = _Echo().execute(Action(type="close"), {"n": 1}, None)
    assert result.state == {"n": 1, "last": "close"}
    assert result.events == [{"type": "close_done"}]
    assert result.error is None


def test_close_is_concrete_so_stateless_backends_need_not_define_it():
    # False-positive guard: an in-process backend holds no connection to close.
    _Echo().close()


def test_a_backend_missing_execute_cannot_be_instantiated():
    class Incomplete(ExecutionBackend):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_a_transition_handler_missing_apply_cannot_be_instantiated():
    class Incomplete(TransitionHandler):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
```

```python
# tests/contracts/test_transport_contract.py
from __future__ import annotations

import pytest

from forge.contracts import Transport, TransportRequest, TransportResponse


class _Loopback(Transport):
    def call(self, request: TransportRequest) -> TransportResponse:
        return TransportResponse(status=200, body={"target": request.target})


def test_a_transport_round_trips_a_request():
    response = _Loopback().call(TransportRequest(method="POST", target="/close"))
    assert response.status == 200
    assert response.body == {"target": "/close"}
    assert response.error is None


def test_a_transport_response_can_carry_an_error():
    # Negative: transport failures are reported in-band, not by raising, so a
    # runner can record the failed step rather than losing the episode.
    response = TransportResponse(status=0, body={}, error="connection refused")
    assert response.error == "connection refused"


def test_a_transport_missing_call_cannot_be_instantiated():
    class Incomplete(Transport):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/contracts/test_backend_contract.py tests/contracts/test_transport_contract.py -v`
Expected: FAIL with `ImportError: cannot import name 'ExecutionBackend' from 'forge.contracts'`

- [ ] **Step 3: Write the two modules**

```python
# forge/contracts/backend.py
"""Where actions actually run."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from forge.contracts.types import Action, ActionResult

if TYPE_CHECKING:
    from forge.runtime.context import RuntimeContext
    from forge.runtime.transition import TransitionResult


class TransitionHandler(ABC):
    """One action's state transition, in-process.

    Replaces the bare Callable the transition registry accepted. A handler with
    the wrong signature is now rejected when it is registered rather than when
    it is first invoked, mid-episode.
    """

    @abstractmethod
    def apply(
        self, state: dict, action: Action, ctx: "RuntimeContext"
    ) -> "TransitionResult":
        """Return the new state and the events this action emitted."""


class ExecutionBackend(ABC):
    """Executes one action wherever the environment actually lives.

    In-process, a container over HTTP, a shell over docker exec, or a browser
    over CDP. `close` is concrete because a stateless backend holds nothing to
    release.
    """

    @abstractmethod
    def execute(
        self, action: Action, state: dict, ctx: "RuntimeContext"
    ) -> ActionResult:
        """Run the action and return the resulting state and events."""

    def close(self) -> None:
        """Release any held resource. No-op by default."""
        return None
```

```python
# forge/contracts/transport.py
"""How the model talks to the environment."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class TransportRequest(BaseModel):
    method: str = "POST"
    target: str = ""
    payload: dict = Field(default_factory=dict)
    timeout: float | None = None


class TransportResponse(BaseModel):
    """A transport result.

    Failures are reported in-band via `error` rather than raised, so a runner
    can record a failed step and continue instead of losing the whole episode
    to an exception from the wire.
    """

    status: int = 0
    body: dict = Field(default_factory=dict)
    error: str | None = None


class Transport(ABC):
    """The wire between the controller and the environment.

    Optional on the Environment facade: a pure-Python environment is reached by
    direct call and has no wire.
    """

    @abstractmethod
    def call(self, request: TransportRequest) -> TransportResponse:
        """Perform one round trip."""

    def close(self) -> None:
        """Release the connection. No-op by default."""
        return None
```

- [ ] **Step 4: Export from `forge/contracts/__init__.py`**

```python
from forge.contracts.backend import ExecutionBackend, TransitionHandler
from forge.contracts.transport import Transport, TransportRequest, TransportResponse
```

Add `"ExecutionBackend"`, `"TransitionHandler"`, `"Transport"`, `"TransportRequest"`, `"TransportResponse"` to `__all__`.

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/contracts/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add forge/contracts tests/contracts
git commit -m "Add ExecutionBackend, TransitionHandler, and Transport contracts

TransitionHandler is the typed replacement for the bare Callable the transition
registry accepts today. Transport reports failures in-band so a wire error
costs one step rather than the episode."
```

---

### Task 6: `Rubric` and `Verifier`

**Files:**
- Create: `forge/contracts/reward.py`
- Modify: `forge/contracts/__init__.py`
- Create: `tests/contracts/test_reward_contract.py`

**Interfaces:**
- Consumes: `RewardBreakdown`, `RewardComponent`, `VerificationResult`, `Task` from `forge.contracts.types`.
- Produces: `Verifier` with `verify(state: dict, trajectory, task: Task | None) -> VerificationResult`. `Rubric` with `score(state: dict, trajectory, verifier_results: Sequence[VerificationResult], task: Task | None) -> RewardBreakdown`.

- [ ] **Step 1: Write the failing test**

```python
# tests/contracts/test_reward_contract.py
from __future__ import annotations

from collections.abc import Sequence

import pytest

from forge.contracts import (
    CheckResult,
    RewardBreakdown,
    RewardComponent,
    Rubric,
    Task,
    VerificationResult,
    Verifier,
)


class _AlwaysPasses(Verifier):
    def verify(self, state: dict, trajectory, task: Task | None) -> VerificationResult:
        return VerificationResult.from_checks(
            "v1", [CheckResult(name="ok", passed=True, score=1.0)]
        )


class _TaskSuccess(Rubric):
    def score(
        self,
        state: dict,
        trajectory,
        verifier_results: Sequence[VerificationResult],
        task: Task | None,
    ) -> RewardBreakdown:
        value = 1.0 if any(r.passed for r in verifier_results) else 0.0
        return RewardBreakdown(
            total_reward=value,
            components=[RewardComponent(name="task_success", value=value)],
        )


def test_a_verifier_returns_a_verification_result():
    result = _AlwaysPasses().verify({}, None, None)
    assert result.passed is True
    assert result.score == 1.0


def test_a_rubric_scores_from_verifier_results():
    verdict = _AlwaysPasses().verify({}, None, None)
    breakdown = _TaskSuccess().score({}, None, [verdict], None)
    assert breakdown.total_reward == 1.0
    assert breakdown.components[0].name == "task_success"


def test_no_verifier_results_scores_zero_not_one():
    # Negative: an unverified episode must not be rewarded by default.
    assert _TaskSuccess().score({}, None, [], None).total_reward == 0.0


def test_a_rubric_missing_score_cannot_be_instantiated():
    class Incomplete(Rubric):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_a_verifier_missing_verify_cannot_be_instantiated():
    class Incomplete(Verifier):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/contracts/test_reward_contract.py -v`
Expected: FAIL with `ImportError: cannot import name 'Rubric' from 'forge.contracts'`

- [ ] **Step 3: Write the contract**

```python
# forge/contracts/reward.py
"""How the model's behavior is scored."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from forge.contracts.types import RewardBreakdown, Task, VerificationResult

if TYPE_CHECKING:
    from forge.runtime.trajectory import Trajectory


class Verifier(ABC):
    """Decides whether a task was accomplished.

    Separate from Rubric because passing and scoring are different questions:
    a verifier answers "did it happen", a rubric answers "how much is that
    worth". Keeping them apart lets one rubric weigh several verifiers.
    """

    @abstractmethod
    def verify(
        self, state: dict, trajectory: "Trajectory", task: Task | None
    ) -> VerificationResult:
        """Check the task's conditions against the final state and trajectory."""


class Rubric(ABC):
    """Turns verification into a reward.

    Implemented by string matching, unit tests, an LLM judge, a tiered engine,
    or any combination — the contract does not care which, only that a
    breakdown comes back so the components are auditable.
    """

    @abstractmethod
    def score(
        self,
        state: dict,
        trajectory: "Trajectory",
        verifier_results: Sequence[VerificationResult],
        task: Task | None,
    ) -> RewardBreakdown:
        """Return the total reward and the components that produced it."""
```

- [ ] **Step 4: Export from `forge/contracts/__init__.py`**

```python
from forge.contracts.reward import Rubric, Verifier
```

Add `"Rubric"` and `"Verifier"` to `__all__`.

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/contracts/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add forge/contracts tests/contracts
git commit -m "Add Rubric and Verifier contracts

Split because passing and scoring are different questions. A verifier answers
whether the task happened; a rubric answers what that is worth, and one rubric
may weigh several verifiers."
```

---

### Task 7: `EpisodeController` and the episode data types

**Files:**
- Create: `forge/contracts/episode.py`
- Modify: `forge/contracts/__init__.py`
- Create: `tests/contracts/test_episode_contract.py`

**Interfaces:**
- Consumes: `AgentAdapter` from `forge.contracts.types`.
- Produces: `BaseEpisodeConfig`, `BaseEpisodeResult`, `TrajectoryWriter`, `EpisodeController` with `run_episode(agent, *, episode_id=None, seed=None, jsonl_path=None) -> BaseEpisodeResult`.

Note: this task moves `BaseEpisodeConfig`, `BaseEpisodeResult`, and `TrajectoryWriter` verbatim from `forge/envgen/episode_base.py`. `TerminationMonitor` stays in `episode_base.py` for now; Task 10 moves it.

- [ ] **Step 1: Write the failing test**

```python
# tests/contracts/test_episode_contract.py
from __future__ import annotations

from pathlib import Path

import pytest

from forge.contracts import BaseEpisodeConfig, BaseEpisodeResult, EpisodeController


class _OneStep(EpisodeController):
    def run_episode(
        self,
        agent,
        *,
        episode_id: str | None = None,
        seed: int | None = None,
        jsonl_path: Path | None = None,
    ) -> BaseEpisodeResult:
        result = BaseEpisodeResult()
        result.termination_reason = f"seed={seed}"
        return result


def test_a_controller_runs_an_episode_and_returns_a_result():
    result = _OneStep().run_episode(agent=None, seed=7)
    assert result.termination_reason == "seed=7"


def test_seed_is_uniform_across_controllers_even_when_unused():
    # False-positive guard: a family with no seeding path still accepts the
    # keyword, so callers need no per-family special case.
    assert _OneStep().run_episode(agent=None).termination_reason == "seed=None"


def test_a_controller_missing_run_episode_cannot_be_instantiated():
    class Incomplete(EpisodeController):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_episode_config_defaults_match_the_documented_thresholds():
    config = BaseEpisodeConfig(objective="close the ticket")
    assert config.max_steps == 30
    assert config.divergence_threshold == 0.2
    assert config.consecutive_below_threshold == 3
    assert config.dead_end_patience == 5
    assert config.success_threshold == 0.9
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/contracts/test_episode_contract.py -v`
Expected: FAIL with `ImportError: cannot import name 'BaseEpisodeConfig' from 'forge.contracts'`

- [ ] **Step 3: Create `forge/contracts/episode.py`**

Move `BaseEpisodeConfig`, `BaseEpisodeResult`, and `TrajectoryWriter` verbatim from `forge/envgen/episode_base.py` (they are currently at lines 17-105), preserving their docstrings and behavior exactly, then add the controller ABC:

```python
# forge/contracts/episode.py — appended after the moved classes
class EpisodeController(ABC):
    """Drives the multi-turn loop and decides when to stop.

    Deliberately not a member of the Environment facade: a controller drives an
    environment from outside, and the same environment may be run by a trainer,
    a benchmark harness, or a parallel rollout worker.
    """

    @abstractmethod
    def run_episode(
        self,
        agent: AgentAdapter,
        *,
        episode_id: str | None = None,
        seed: int | None = None,
        jsonl_path: Path | None = None,
    ) -> BaseEpisodeResult:
        """Run one episode to termination and return its result.

        `seed` is accepted by every controller for a uniform call signature,
        even where the family has no seeding path and ignores it.
        """
```

Required imports at the top of the new file:

```python
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from forge.contracts.types import AgentAdapter
```

- [ ] **Step 4: Make `forge/envgen/episode_base.py` re-export the moved names**

Replace the moved class definitions with:

```python
# Moved to forge/contracts/episode.py. Re-exported here so existing imports
# keep working; prefer importing from forge.contracts.
from forge.contracts.episode import (  # noqa: F401
    BaseEpisodeConfig,
    BaseEpisodeResult,
    TrajectoryWriter,
)
```

Keep `TerminationMonitor` in place — Task 10 moves it.

- [ ] **Step 5: Export from `forge/contracts/__init__.py`**

```python
from forge.contracts.episode import (
    BaseEpisodeConfig,
    BaseEpisodeResult,
    EpisodeController,
    TrajectoryWriter,
)
```

Add `"BaseEpisodeConfig"`, `"BaseEpisodeResult"`, `"EpisodeController"`, `"TrajectoryWriter"` to `__all__`.

- [ ] **Step 6: Run the contract tests and the existing episode tests**

Run: `.venv/bin/python -m pytest tests/contracts/ tests/envgen/test_episode_base.py tests/envgen/test_trajectory_durability.py -v`
Expected: PASS. The existing tests import from `forge.envgen.episode_base` and must keep working through the re-export.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: 1265 passed plus the new contract tests. Any failure here means the move was not verbatim.

- [ ] **Step 8: Commit**

```bash
git add forge/contracts forge/envgen/episode_base.py tests/contracts
git commit -m "Add EpisodeController contract and move episode data types

BaseEpisodeConfig, BaseEpisodeResult, and TrajectoryWriter move to
forge/contracts/episode.py; episode_base.py re-exports them. EpisodeController
is not part of the Environment facade because a controller drives an
environment from outside, and the same environment may be run by several."
```

---

### Task 8: The `Environment` facade

**Files:**
- Create: `forge/contracts/environment.py`
- Modify: `forge/contracts/__init__.py`
- Create: `tests/contracts/test_environment_facade.py`

**Interfaces:**
- Consumes: all seven required and three optional contracts.
- Produces: `Environment` with required properties `task_source`, `initial_state`, `observations`, `backend`, `state`, `rubric`, `termination`, and optional properties `prompt`, `tools`, `transport` defaulting to `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/contracts/test_environment_facade.py
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from forge.contracts import (
    Action,
    ActionResult,
    Environment,
    ExecutionBackend,
    InitialStateProvider,
    Observation,
    ObservationEncoder,
    RewardBreakdown,
    RewardComponent,
    Rubric,
    StateManager,
    StepOutcome,
    Task,
    TaskSource,
    Termination,
    TerminationPolicy,
    ToolProvider,
    ToolSpec,
    VerificationResult,
)


class _Tasks(TaskSource):
    def tasks(self) -> Sequence[Task]:
        return [Task(id="t1", objective="close the ticket")]

    def get(self, task_id: str) -> Task:
        return self.tasks()[0]


class _Initial(InitialStateProvider):
    def reset(self, ctx, *, seed: int | None, options: Mapping[str, object]) -> dict:
        return {}


class _Obs(ObservationEncoder):
    def encode(self, state: dict, ctx) -> Observation:
        return Observation(payload=state)


class _Backend(ExecutionBackend):
    def execute(self, action: Action, state: dict, ctx) -> ActionResult:
        return ActionResult(state=state)


class _State(StateManager):
    def get(self) -> dict:
        return {}

    def apply(self, state: dict) -> None:
        return None

    def hash(self) -> str:
        return "sha256:0"


class _Rubric(Rubric):
    def score(self, state, trajectory, verifier_results, task) -> RewardBreakdown:
        return RewardBreakdown(
            total_reward=0.0, components=[RewardComponent(name="none", value=0.0)]
        )


class _Termination(TerminationPolicy):
    def check(self, outcome: StepOutcome) -> Termination | None:
        return None


class _Headless(Environment):
    """An environment with no tools, prompt, or transport — a CLI-shaped one."""

    task_source = _Tasks()
    initial_state = _Initial()
    observations = _Obs()
    backend = _Backend()
    state = _State()
    rubric = _Rubric()
    termination = _Termination()


def test_an_environment_supplying_the_seven_required_members_instantiates():
    env = _Headless()
    assert env.task_source.get("t1").objective == "close the ticket"
    assert env.state.hash() == "sha256:0"


def test_optional_members_default_to_none():
    # False-positive guard: a shell environment has no tool schema and no wire.
    # It must not be forced to stub them.
    env = _Headless()
    assert env.prompt is None
    assert env.tools is None
    assert env.transport is None


def test_an_environment_missing_a_required_member_cannot_be_instantiated():
    # Negative: omitting a required concern fails at instantiation.
    class Incomplete(Environment):
        task_source = _Tasks()
        initial_state = _Initial()
        observations = _Obs()
        backend = _Backend()
        state = _State()
        rubric = _Rubric()
        # termination omitted

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_an_environment_may_supply_the_optional_members():
    class WithTools(_Headless):
        @property
        def tools(self) -> ToolProvider:
            class _Static(ToolProvider):
                def tools(self) -> Sequence[ToolSpec]:
                    return [ToolSpec(name="close_ticket")]

            return _Static()

    assert [t.name for t in WithTools().tools.tools()] == ["close_ticket"]


def test_episode_controller_is_not_part_of_the_facade():
    # A controller drives an environment from outside; folding it in would
    # imply every environment owns its own loop.
    assert not hasattr(_Headless(), "episode_controller")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/contracts/test_environment_facade.py -v`
Expected: FAIL with `ImportError: cannot import name 'Environment' from 'forge.contracts'`

- [ ] **Step 3: Write the facade**

```python
# forge/contracts/environment.py
"""The composed contract a complete Forge environment satisfies."""
from __future__ import annotations

from abc import ABC, abstractmethod

from forge.contracts.backend import ExecutionBackend
from forge.contracts.dataset import TaskSource
from forge.contracts.initial_state import InitialStateProvider
from forge.contracts.observation import ObservationEncoder
from forge.contracts.prompting import PromptTemplate
from forge.contracts.reward import Rubric
from forge.contracts.state import StateManager
from forge.contracts.termination import TerminationPolicy
from forge.contracts.tools import ToolProvider
from forge.contracts.transport import Transport


class Environment(ABC):
    """Ten of the eleven concerns, composed.

    Seven members are required because every environment family has them. The
    three optional ones are exactly those a family can legitimately lack: a
    shell environment has no tool schema, a pure-Python environment has no
    transport, and an environment driven by a trainer that supplies its own
    prompting has no template.

    EpisodeController is deliberately absent — see forge/contracts/episode.py.
    """

    @property
    @abstractmethod
    def task_source(self) -> TaskSource: ...

    @property
    @abstractmethod
    def initial_state(self) -> InitialStateProvider: ...

    @property
    @abstractmethod
    def observations(self) -> ObservationEncoder: ...

    @property
    @abstractmethod
    def backend(self) -> ExecutionBackend: ...

    @property
    @abstractmethod
    def state(self) -> StateManager: ...

    @property
    @abstractmethod
    def rubric(self) -> Rubric: ...

    @property
    @abstractmethod
    def termination(self) -> TerminationPolicy: ...

    # ------------------------------------------------------------------
    # Optional concerns
    # ------------------------------------------------------------------

    @property
    def prompt(self) -> PromptTemplate | None:
        return None

    @property
    def tools(self) -> ToolProvider | None:
        return None

    @property
    def transport(self) -> Transport | None:
        return None
```

- [ ] **Step 4: Export from `forge/contracts/__init__.py`**

```python
from forge.contracts.environment import Environment
```

Add `"Environment"` to `__all__`.

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/contracts/ -v`
Expected: PASS

- [ ] **Step 6: Run the full suite — phase 1 gate**

Run: `.venv/bin/python -m pytest -q`
Expected: 1265 existing passed, plus the new contract tests, zero failures.

- [ ] **Step 7: Commit**

```bash
git add forge/contracts tests/contracts
git commit -m "Add the Environment facade composing ten contracts

Seven required members every family has; three optional ones a family can
legitimately lack. EpisodeController stays out because the same environment may
be driven by a trainer, a benchmark harness, or a rollout worker."
```

---

# Phase 2 — The rebase

Phase 2 moves the existing implementations onto the contracts. Order matters: the leaf types first, the registries next, the environments and runners last, and the code generators after the runtime they emit calls into.

### Task 9: `StateStore` becomes `InProcessStateManager`; add `HttpStateManager`

**Files:**
- Modify: `forge/runtime/state.py`
- Create: `forge/runtime/http_state.py`
- Test: `tests/runtime/test_state_manager.py` (create)

**Interfaces:**
- Consumes: `StateManager` from `forge.contracts`.
- Produces: `InProcessStateManager(StateManager)` in `forge/runtime/state.py`, with `StateStore` retained as an alias. `HttpStateManager(StateManager)` in `forge/runtime/http_state.py`, taking `base_url: str` and `client: httpx.Client`.

- [ ] **Step 1: Write the failing test**

```python
# tests/runtime/test_state_manager.py
from __future__ import annotations

import httpx
import pytest

from forge.contracts import StateManager
from forge.runtime.http_state import HttpStateManager
from forge.runtime.state import InProcessStateManager, StateStore


def test_in_process_manager_satisfies_the_contract():
    assert isinstance(InProcessStateManager({}), StateManager)


def test_state_store_remains_importable_as_an_alias():
    # False-positive guard: renaming must not break existing imports.
    assert StateStore is InProcessStateManager


def test_get_returns_a_copy_not_a_live_reference():
    # Negative: a caller mutating the returned dict must not corrupt state.
    manager = InProcessStateManager({"tickets": []})
    manager.get()["tickets"].append("leaked")
    assert manager.get() == {"tickets": []}


def test_hash_is_stable_across_key_order():
    a = InProcessStateManager({"x": 1, "y": 2})
    b = InProcessStateManager({"y": 2, "x": 1})
    assert a.hash() == b.hash()


def test_in_process_manager_rejects_named_slots():
    # Only the container family supports slots.
    with pytest.raises(NotImplementedError):
        InProcessStateManager({}).snapshot("s1")


def test_http_manager_reads_state_over_the_wire():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/forge/state"
        return httpx.Response(200, json={"tickets": [{"id": "t1"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    manager = HttpStateManager("http://env", client=client)
    assert manager.get() == {"tickets": [{"id": "t1"}]}


def test_http_manager_supports_named_slots():
    # False-positive guard: the container family does support slots, so the
    # base class's refusal must be overridden here.
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    manager = HttpStateManager("http://env", client=client)
    manager.snapshot("s1")
    manager.restore("s1")
    assert calls == ["/forge/snapshot", "/forge/restore/s1"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/runtime/test_state_manager.py -v`
Expected: FAIL with `ImportError: cannot import name 'InProcessStateManager'`

- [ ] **Step 3: Rewrite `forge/runtime/state.py`**

```python
from __future__ import annotations

import copy
import hashlib
import json

from forge.contracts import StateManager


class InProcessStateManager(StateManager):
    """State held in memory, hashed with sorted keys for cross-run stability."""

    def __init__(self, initial_state: dict) -> None:
        self._state = copy.deepcopy(initial_state)

    def get(self) -> dict:
        return copy.deepcopy(self._state)

    def apply(self, new_state: dict) -> None:
        self._state = copy.deepcopy(new_state)

    def hash(self) -> str:
        serialized = json.dumps(self._state, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        return f"sha256:{digest}"


# The pre-contracts name. Retained so existing imports and generated packages
# keep working.
StateStore = InProcessStateManager
```

- [ ] **Step 4: Write `forge/runtime/http_state.py`**

```python
# forge/runtime/http_state.py
"""State manager for container-backed environments.

The container app's SQLite database is the source of truth; this reads and
writes it through the Forge control endpoints rather than holding a copy.
"""
from __future__ import annotations

import hashlib
import json

import httpx

from forge.contracts import StateManager


class HttpStateManager(StateManager):
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=15.0)

    def get(self) -> dict:
        response = self._client.get(f"{self._base_url}/forge/state")
        response.raise_for_status()
        return response.json()

    def apply(self, new_state: dict) -> None:
        response = self._client.post(
            f"{self._base_url}/forge/restore-state", json=new_state
        )
        response.raise_for_status()

    def hash(self) -> str:
        serialized = json.dumps(self.get(), sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(serialized.encode()).hexdigest()}"

    def snapshot(self, slot: str) -> None:
        response = self._client.post(
            f"{self._base_url}/forge/snapshot", json={"slot": slot}
        )
        response.raise_for_status()

    def restore(self, slot: str) -> None:
        response = self._client.post(f"{self._base_url}/forge/restore/{slot}")
        response.raise_for_status()
```

- [ ] **Step 5: Run the new test and the full suite**

Run: `.venv/bin/python -m pytest tests/runtime/test_state_manager.py -v && .venv/bin/python -m pytest -q`
Expected: new tests PASS; full suite unchanged at 1265 + contract tests.

- [ ] **Step 6: Commit**

```bash
git add forge/runtime/state.py forge/runtime/http_state.py tests/runtime/test_state_manager.py
git commit -m "Bind state management to the StateManager contract

StateStore becomes InProcessStateManager with the old name kept as an alias.
HttpStateManager covers the container family, including the named slots the
in-process manager legitimately refuses."
```

---

### Task 10: `TerminationMonitor` becomes `ThresholdTerminationPolicy`

**Files:**
- Modify: `forge/contracts/termination.py`
- Modify: `forge/envgen/episode_base.py`
- Test: `tests/contracts/test_threshold_termination.py` (create)

**Interfaces:**
- Consumes: `TerminationPolicy`, `StepOutcome`, `Termination`, `BaseEpisodeConfig`.
- Produces: `ThresholdTerminationPolicy(config: BaseEpisodeConfig)` with `check(outcome) -> Termination | None` and the legacy `observe(score, marker=None) -> str | None`. `MaxStepsTerminationPolicy(max_steps: int)`. `TerminationMonitor` alias.

- [ ] **Step 1: Write the failing test pinning current behavior**

```python
# tests/contracts/test_threshold_termination.py
"""ThresholdTerminationPolicy must reproduce TerminationMonitor exactly."""
from __future__ import annotations

from forge.contracts import BaseEpisodeConfig, StepOutcome
from forge.contracts.termination import (
    MaxStepsTerminationPolicy,
    ThresholdTerminationPolicy,
)


def _config(**kwargs) -> BaseEpisodeConfig:
    return BaseEpisodeConfig(objective="close the ticket", **kwargs)


def test_reaching_the_success_threshold_terminates():
    policy = ThresholdTerminationPolicy(_config())
    assert policy.check(StepOutcome(step_index=0, score=0.95)).reason == "success"


def test_an_unchanged_marker_for_the_patience_window_is_a_dead_end():
    policy = ThresholdTerminationPolicy(_config(dead_end_patience=3))
    outcomes = [
        policy.check(StepOutcome(step_index=i, score=0.5, state_hash="same"))
        for i in range(3)
    ]
    assert outcomes[-1].reason == "dead_end"


def test_sustained_low_scores_diverge():
    policy = ThresholdTerminationPolicy(
        _config(divergence_threshold=0.2, consecutive_below_threshold=2, dead_end_patience=99)
    )
    policy.check(StepOutcome(step_index=0, score=0.1, state_hash="a"))
    assert policy.check(StepOutcome(step_index=1, score=0.1, state_hash="b")).reason == "diverged"


def test_success_outranks_dead_end():
    # Negative: a high score on an unchanged state is a success, not a dead end.
    policy = ThresholdTerminationPolicy(_config(dead_end_patience=1))
    assert policy.check(StepOutcome(step_index=0, score=0.99, state_hash="same")).reason == "success"


def test_a_recovering_score_resets_the_divergence_counter():
    # False-positive guard: one good step must clear the streak.
    policy = ThresholdTerminationPolicy(
        _config(divergence_threshold=0.2, consecutive_below_threshold=2, dead_end_patience=99)
    )
    policy.check(StepOutcome(step_index=0, score=0.1, state_hash="a"))
    policy.check(StepOutcome(step_index=1, score=0.5, state_hash="b"))
    assert policy.check(StepOutcome(step_index=2, score=0.1, state_hash="c")) is None


def test_the_legacy_observe_api_still_works():
    # The three runners call observe(); it must survive the rename.
    policy = ThresholdTerminationPolicy(_config())
    assert policy.observe(0.95) == "success"
    assert policy.observe(0.5) is None


def test_max_steps_truncates_rather_than_terminates():
    policy = MaxStepsTerminationPolicy(max_steps=3)
    assert policy.check(StepOutcome(step_index=1)) is None
    decision = policy.check(StepOutcome(step_index=2))
    assert decision.reason == "max_steps"
    assert decision.truncated is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/contracts/test_threshold_termination.py -v`
Expected: FAIL with `ImportError: cannot import name 'ThresholdTerminationPolicy'`

- [ ] **Step 3: Append the implementations to `forge/contracts/termination.py`**

```python
class ThresholdTerminationPolicy(TerminationPolicy):
    """Success / dead-end / divergence, in that priority order.

    The priority matters: a high score on an unchanged state is a success, not
    a dead end. `observe` is the pre-contracts API the three runners call and is
    kept as a thin wrapper.
    """

    def __init__(self, config: "BaseEpisodeConfig") -> None:
        self._cfg = config
        self._markers: list[object] = []
        self._below_threshold_count = 0

    def check(self, outcome: StepOutcome) -> Termination | None:
        reason = self.observe(outcome.score, outcome.state_hash)
        return Termination(reason=reason) if reason else None

    def observe(self, score: float, marker: object = None) -> str | None:
        self._markers.append(marker if marker is not None else round(score, 2))

        if score >= self._cfg.success_threshold:
            return "success"

        if len(self._markers) >= self._cfg.dead_end_patience:
            recent = self._markers[-self._cfg.dead_end_patience:]
            if len(set(recent)) == 1:
                return "dead_end"

        if score < self._cfg.divergence_threshold:
            self._below_threshold_count += 1
        else:
            self._below_threshold_count = 0
        if self._below_threshold_count >= self._cfg.consecutive_below_threshold:
            return "diverged"
        return None


class MaxStepsTerminationPolicy(TerminationPolicy):
    """The step budget, made explicit.

    Each runner previously inlined this as `step_index == max_steps - 1`.
    """

    def __init__(self, max_steps: int) -> None:
        self._max_steps = max_steps

    def check(self, outcome: StepOutcome) -> Termination | None:
        if outcome.step_index >= self._max_steps - 1:
            return Termination(reason="max_steps", truncated=True)
        return None


# The pre-contracts name, kept so the runners and their tests keep working.
TerminationMonitor = ThresholdTerminationPolicy
```

Add at the top of the file, under the existing imports:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.contracts.episode import BaseEpisodeConfig
```

- [ ] **Step 4: Re-export from `forge/envgen/episode_base.py`**

Delete the `TerminationMonitor` class body and extend the existing re-export block:

```python
from forge.contracts.termination import (  # noqa: F401
    MaxStepsTerminationPolicy,
    TerminationMonitor,
    ThresholdTerminationPolicy,
)
```

- [ ] **Step 5: Export from `forge/contracts/__init__.py`**

```python
from forge.contracts.termination import (
    MaxStepsTerminationPolicy,
    TerminationMonitor,
    TerminationPolicy,
    ThresholdTerminationPolicy,
)
```

Add the three new names to `__all__`.

- [ ] **Step 6: Run the new test, then the full suite**

Run: `.venv/bin/python -m pytest tests/contracts/test_threshold_termination.py -v && .venv/bin/python -m pytest -q`
Expected: new tests PASS; full suite unchanged. The runners still call `observe()` through the alias.

- [ ] **Step 7: Commit**

```bash
git add forge/contracts/termination.py forge/envgen/episode_base.py tests/contracts/test_threshold_termination.py
git commit -m "Bind termination to the TerminationPolicy contract

TerminationMonitor becomes ThresholdTerminationPolicy, keeping observe() as the
alias the runners call. MaxStepsTerminationPolicy makes explicit the step
budget each runner previously inlined."
```

---

### Task 11: Typed registries in the three engines

This is the defect the whole plan exists to fix: `register` currently accepts any callable, so a generated handler with the wrong arity is accepted at build time and fails mid-episode.

**Files:**
- Modify: `forge/runtime/transition.py`
- Modify: `forge/runtime/verifier.py`
- Modify: `forge/runtime/reward.py`
- Test: `tests/runtime/test_typed_registries.py` (create)

**Interfaces:**
- Consumes: `TransitionHandler`, `Verifier`, `Rubric`, `Action` from `forge.contracts`.
- Produces: `TransitionEngine.register(action_type, handler: TransitionHandler)`, `VerifierEngine.register(verifier_id, verifier: Verifier)`, `RewardEngine.register(task_name, rubric: Rubric)` and `set_default(rubric: Rubric)` — each raising `TypeError` on a non-conforming argument. Plus `FunctionTransitionHandler`, `FunctionVerifier`, `FunctionRubric` adapters that wrap a plain callable, used by `hooks.py` in Task 12.

- [ ] **Step 1: Write the failing test**

```python
# tests/runtime/test_typed_registries.py
"""Registries must reject non-conforming handlers at registration time."""
from __future__ import annotations

import pytest

from forge.contracts import Action, RewardBreakdown, RewardComponent, Rubric, Verifier
from forge.contracts.backend import TransitionHandler
from forge.runtime.reward import FunctionRubric, RewardEngine
from forge.runtime.transition import (
    FunctionTransitionHandler,
    TransitionEngine,
    TransitionResult,
)
from forge.runtime.verifier import FunctionVerifier, VerifierEngine


class _Close(TransitionHandler):
    def apply(self, state: dict, action: Action, ctx) -> TransitionResult:
        return TransitionResult(state={**state, "closed": True}, events=[])


def test_a_conforming_handler_registers_and_applies():
    engine = TransitionEngine()
    engine.register("close_ticket", _Close())
    result = engine.apply({}, {"type": "close_ticket"}, None)
    assert result.state == {"closed": True}


def test_registering_a_bare_function_raises_at_registration():
    # This is the bug the contracts exist to prevent: a wrong-arity handler
    # used to be accepted here and blow up mid-episode instead.
    engine = TransitionEngine()
    with pytest.raises(TypeError, match="TransitionHandler"):
        engine.register("close_ticket", lambda state, action: state)


def test_a_plain_function_can_be_adapted_explicitly():
    # False-positive guard: wrapping stays available, so the customization
    # hooks API does not become less ergonomic.
    engine = TransitionEngine()
    engine.register(
        "close_ticket",
        FunctionTransitionHandler(
            lambda state, action, ctx: TransitionResult(state={"ok": True}, events=[])
        ),
    )
    assert engine.apply({}, {"type": "close_ticket"}, None).state == {"ok": True}


def test_verifier_engine_rejects_a_bare_function():
    with pytest.raises(TypeError, match="Verifier"):
        VerifierEngine().register("v1", lambda s, t, task: None)


def test_reward_engine_rejects_a_bare_function():
    with pytest.raises(TypeError, match="Rubric"):
        RewardEngine().register("t1", lambda *args: None)


def test_reward_engine_default_still_scores_from_verifier_results():
    # Behavior preservation: the documented fallback is unchanged.
    engine = RewardEngine()

    class _Passed:
        passed = True

    breakdown = engine.compute({}, None, [_Passed()], None)
    assert breakdown.total_reward == 1.0
    assert breakdown.components[0].name == "task_success"


def test_reward_engine_default_scores_zero_with_no_passing_verifier():
    # Negative: nothing passing must score zero, not a vacuous one.
    assert RewardEngine().compute({}, None, [], None).total_reward == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/runtime/test_typed_registries.py -v`
Expected: FAIL with `ImportError: cannot import name 'FunctionTransitionHandler'`

- [ ] **Step 3: Rewrite `forge/runtime/transition.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from forge.contracts import Action
from forge.contracts.backend import TransitionHandler
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError


@dataclass
class TransitionResult:
    state: dict
    events: list[dict] = field(default_factory=list)


class FunctionTransitionHandler(TransitionHandler):
    """Adapts a plain `(state, action, ctx) -> TransitionResult` function.

    Used by the customization hooks so a decorated function stays as easy to
    write as it was before the contract existed.
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    def apply(self, state: dict, action: Action, ctx: RuntimeContext) -> TransitionResult:
        return self._fn(state, action.to_dict(), ctx)


class TransitionEngine:
    def __init__(self) -> None:
        self._handlers: dict[str, TransitionHandler] = {}

    def register(self, action_type: str, handler: TransitionHandler) -> None:
        if not isinstance(handler, TransitionHandler):
            raise TypeError(
                f"Handler for {action_type!r} must be a TransitionHandler, got "
                f"{type(handler).__name__}. Wrap a plain function in "
                f"FunctionTransitionHandler."
            )
        self._handlers[action_type] = handler

    @property
    def action_types(self) -> set[str]:
        return set(self._handlers.keys())

    def apply(self, state: dict, action: dict, ctx: RuntimeContext) -> TransitionResult:
        handler = self._handlers.get(action.get("type", ""))
        if handler is None:
            raise InvalidActionError(
                f"Unknown action type: '{action.get('type')}'. Valid: {sorted(self._handlers)}",
                code="UNKNOWN_ACTION_TYPE",
            )
        return handler.apply(state, Action.from_dict(action), ctx)
```

- [ ] **Step 4: Rewrite `forge/runtime/verifier.py`**

```python
from __future__ import annotations

from typing import Callable

from forge.contracts import VerificationResult, Verifier


class FunctionVerifier(Verifier):
    """Adapts a plain `(state, trajectory, task) -> VerificationResult` function."""

    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    def verify(self, state: dict, trajectory, task) -> VerificationResult:
        return self._fn(state, trajectory, task)


class VerifierEngine:
    def __init__(self) -> None:
        self._verifiers: dict[str, Verifier] = {}

    def register(self, verifier_id: str, verifier: Verifier) -> None:
        if not isinstance(verifier, Verifier):
            raise TypeError(
                f"Verifier {verifier_id!r} must be a Verifier, got "
                f"{type(verifier).__name__}. Wrap a plain function in FunctionVerifier."
            )
        self._verifiers[verifier_id] = verifier

    def run_all(
        self, state: dict, trajectory, task: dict | None
    ) -> list[VerificationResult]:
        if task is None:
            return []
        verifier_id = task.get("verifier_id")
        if not verifier_id or verifier_id not in self._verifiers:
            return []
        return [self._verifiers[verifier_id].verify(state, trajectory, task)]
```

- [ ] **Step 5: Rewrite `forge/runtime/reward.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from forge.contracts import RewardBreakdown, RewardComponent, Rubric

if TYPE_CHECKING:
    from forge.runtime.trajectory import Trajectory


class FunctionRubric(Rubric):
    """Adapts a plain `(state, trajectory, verifier_results, task)` function."""

    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    def score(self, state, trajectory, verifier_results, task) -> RewardBreakdown:
        return self._fn(state, trajectory, verifier_results, task)


class TaskSuccessRubric(Rubric):
    """The default: 1.0 if any verifier passed, else 0.0."""

    def score(self, state, trajectory, verifier_results, task) -> RewardBreakdown:
        passed = any(vr.passed for vr in verifier_results)
        value = 1.0 if passed else 0.0
        return RewardBreakdown(
            total_reward=value,
            components=[RewardComponent(name="task_success", value=value)],
        )


class RewardEngine:
    def __init__(self) -> None:
        self._task_rubrics: dict[str, Rubric] = {}
        self._default: Rubric | None = None

    def register(self, task_name: str, rubric: Rubric) -> None:
        if not isinstance(rubric, Rubric):
            raise TypeError(
                f"Rubric for {task_name!r} must be a Rubric, got "
                f"{type(rubric).__name__}. Wrap a plain function in FunctionRubric."
            )
        self._task_rubrics[task_name] = rubric

    def set_default(self, rubric: Rubric) -> None:
        if not isinstance(rubric, Rubric):
            raise TypeError(
                f"Default rubric must be a Rubric, got {type(rubric).__name__}. "
                f"Wrap a plain function in FunctionRubric."
            )
        self._default = rubric

    def compute(
        self,
        state: dict,
        trajectory: "Trajectory",
        verifier_results: list,
        task: dict | None = None,
    ) -> RewardBreakdown:
        task_name = task.get("name") if task else None
        rubric = self._task_rubrics.get(task_name) if task_name else None
        rubric = rubric or self._default or TaskSuccessRubric()
        return rubric.score(state, trajectory, verifier_results, task)
```

- [ ] **Step 6: Re-export `RewardComponent` / `RewardBreakdown` from their old home**

`forge/runtime/reward.py` already imports both from `forge.contracts`, so `from forge.runtime.reward import RewardBreakdown` keeps working. Verify with:

Run: `.venv/bin/python -c "from forge.runtime.reward import RewardBreakdown, RewardComponent; from forge.runtime.verification import VerificationResult, CheckResult; print('ok')"`
Expected: `ok`

- [ ] **Step 7: Run the new test, then the full suite**

Run: `.venv/bin/python -m pytest tests/runtime/test_typed_registries.py -v`
Expected: PASS

Run: `.venv/bin/python -m pytest -q`
Expected: failures in tests that register bare functions. Those are the call sites Task 12 fixes; note them and proceed to Task 12 before committing this task. If any failure is NOT a bare-function registration, stop — it means behavior changed.

- [ ] **Step 8: Do not commit yet**

This task and Task 12 land together, because tightening `register` breaks the hook decorators until they wrap. Proceed to Task 12.

---

### Task 12: Customization hooks wrap plain functions into the contracts

**Files:**
- Modify: `forge/customization/hooks.py`
- Modify: `forge/customization/loader.py`
- Test: `tests/customization/test_hooks_wrap_contracts.py` (create)

**Interfaces:**
- Consumes: `FunctionTransitionHandler`, `FunctionVerifier`, `FunctionRubric` from Task 11.
- Produces: unchanged decorator API — `@override_transition(name)`, `@verifier(name)`, `@reward(name)` still decorate plain functions; the registry now stores contract instances.

- [ ] **Step 1: Write the failing test**

```python
# tests/customization/test_hooks_wrap_contracts.py
"""The decorator API stays plain-function; the registry stores contracts."""
from __future__ import annotations

from forge.contracts import Rubric, Verifier
from forge.contracts.backend import TransitionHandler
from forge.customization.hooks import (
    clear_registry,
    get_registry,
    override_transition,
    reward,
    verifier,
)
from forge.runtime.transition import TransitionEngine, TransitionResult


def test_a_decorated_transition_is_stored_as_a_contract_instance():
    clear_registry()

    @override_transition("close_ticket")
    def _close(state, action, ctx):
        return TransitionResult(state={"closed": True}, events=[])

    stored = get_registry()["transitions"]["close_ticket"]
    assert isinstance(stored, TransitionHandler)


def test_a_decorated_transition_registers_without_a_type_error():
    # This is the integration the tightened registry would otherwise break.
    clear_registry()

    @override_transition("close_ticket")
    def _close(state, action, ctx):
        return TransitionResult(state={"closed": True}, events=[])

    engine = TransitionEngine()
    engine.register("close_ticket", get_registry()["transitions"]["close_ticket"])
    assert engine.apply({}, {"type": "close_ticket"}, None).state == {"closed": True}


def test_the_decorator_returns_the_original_function():
    # False-positive guard: decorating must not replace the author's function,
    # or the module's own callers would break.
    clear_registry()

    def _close(state, action, ctx):
        return TransitionResult(state={}, events=[])

    assert override_transition("close_ticket")(_close) is _close


def test_decorated_verifiers_and_rewards_are_also_wrapped():
    clear_registry()

    @verifier("task_a")
    def _v(state, trajectory, task):
        return None

    @reward("task_a")
    def _r(state, trajectory, verifier_results, task):
        return None

    assert isinstance(get_registry()["verifiers"]["task_a"], Verifier)
    assert isinstance(get_registry()["rewards"]["task_a"], Rubric)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/customization/test_hooks_wrap_contracts.py -v`
Expected: FAIL — `get_registry()["transitions"]["close_ticket"]` is a bare function, not a `TransitionHandler`.

- [ ] **Step 3: Wrap in the decorators**

In `forge/customization/hooks.py`, add the imports and change the three decorators to store a wrapped instance while returning the original function:

```python
from forge.runtime.reward import FunctionRubric
from forge.runtime.transition import FunctionTransitionHandler
from forge.runtime.verifier import FunctionVerifier


def override_transition(action_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        # The registry stores the contract instance; the author keeps their
        # plain function, so the decorator stays transparent at the call site.
        _registry["transitions"][action_name] = FunctionTransitionHandler(fn)
        return fn
    return decorator


def verifier(task_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _registry["verifiers"][task_name] = FunctionVerifier(fn)
        return fn
    return decorator


def reward(task_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _registry["rewards"][task_name] = FunctionRubric(fn)
        return fn
    return decorator
```

Leave `observation_transform` and `policy_rule` unchanged — they have no contract in this plan.

- [ ] **Step 4: Verify `CustomizationLoader.apply` needs no change**

`loader.py` passes registry values straight into `engine.register(...)`. Those values are now contract instances, so the calls satisfy the tightened registries with no edit. Confirm by reading `forge/customization/loader.py:13-31` and running the loader tests.

Run: `.venv/bin/python -m pytest tests/customization/ -v`
Expected: PASS

- [ ] **Step 5: Run the full suite — the Task 11 + 12 gate**

Run: `.venv/bin/python -m pytest -q`
Expected: all green. Any remaining failure is a call site still registering a bare function; fix it by wrapping in the matching `Function*` adapter.

- [ ] **Step 6: Commit Tasks 11 and 12 together**

```bash
git add forge/runtime/transition.py forge/runtime/verifier.py forge/runtime/reward.py \
        forge/customization/hooks.py tests/runtime/test_typed_registries.py \
        tests/customization/test_hooks_wrap_contracts.py
git commit -m "Reject non-conforming handlers at registration instead of mid-episode

The three engines accepted any callable, so a generated handler with the wrong
arity was accepted at build time and failed during a rollout. They now require
TransitionHandler, Verifier, and Rubric instances.

The customization decorators wrap the author's plain function into the matching
contract and return the original, so the documented hooks API is unchanged."
```

---

### Task 13: `ForgeEnv` takes contract-typed collaborators

**Files:**
- Modify: `forge/runtime/env.py:33` (drop `InitialStateFactory`), `:44-46` (constructor params), `:164` (the `create` call)
- Modify: `forge/runtime/env_builder.py`
- Test: `tests/runtime/test_forge_env_contracts.py` (create)

**Interfaces:**
- Consumes: `InitialStateProvider` from `forge.contracts`.
- Produces: `ForgeEnv(initial_state_provider: InitialStateProvider, ...)`, with `InitialStateFactory` retained as a deprecated alias of `InitialStateProvider`.

- [ ] **Step 1: Write the failing test**

```python
# tests/runtime/test_forge_env_contracts.py
from __future__ import annotations

from collections.abc import Mapping

import pytest

from forge.contracts import Action, InitialStateProvider
from forge.contracts.backend import TransitionHandler
from forge.runtime.env import ForgeEnv
from forge.runtime.reward import RewardEngine
from forge.runtime.snapshot import EnvironmentSpec
from forge.runtime.transition import TransitionEngine, TransitionResult
from forge.runtime.verifier import VerifierEngine


class _Initial(InitialStateProvider):
    def reset(self, ctx, *, seed: int | None, options: Mapping[str, object]) -> dict:
        return {"tickets": [], "seed": seed}


class _Close(TransitionHandler):
    def apply(self, state: dict, action: Action, ctx) -> TransitionResult:
        return TransitionResult(state={**state, "closed": True}, events=[])


def _env() -> ForgeEnv:
    engine = TransitionEngine()
    engine.register("close_ticket", _Close())
    return ForgeEnv(
        env_spec=EnvironmentSpec(name="t", domain="support", max_steps=5),
        initial_state_provider=_Initial(),
        transition_engine=engine,
        verifier_engine=VerifierEngine(),
        reward_engine=RewardEngine(),
    )


def test_reset_threads_the_seed_to_the_provider():
    obs, info = _env().reset(seed=11)
    assert obs["seed"] == 11
    assert info["seed"] == 11


def test_an_unseeded_reset_still_produces_a_seed():
    # False-positive guard: gym requires a usable seed even when none is given,
    # so the provider must receive the derived one rather than None.
    obs, info = _env().reset()
    assert obs["seed"] == info["seed"]
    assert isinstance(info["seed"], int)


def test_step_applies_the_registered_handler():
    env = _env()
    env.reset(seed=1)
    obs, _reward, _term, _trunc, _info = env.step({"type": "close_ticket"})
    assert obs["closed"] is True


def test_initial_state_factory_remains_importable_as_an_alias():
    from forge.runtime.env import InitialStateFactory

    assert InitialStateFactory is InitialStateProvider
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/runtime/test_forge_env_contracts.py -v`
Expected: FAIL with `TypeError: ForgeEnv.__init__() got an unexpected keyword argument 'initial_state_provider'`

- [ ] **Step 3: Edit `forge/runtime/env.py`**

Replace the `InitialStateFactory` Protocol definition (lines 33-34) with:

```python
from forge.contracts import InitialStateProvider

# The pre-contracts name. Retained so existing imports keep working.
InitialStateFactory = InitialStateProvider
```

Rename the constructor parameter `initial_state_factory` to `initial_state_provider` and the attribute `self._factory` to `self._initial_state`.

Replace the `create` call in `reset` (line 164):

```python
initial_state = self._initial_state.reset(
    self._ctx, seed=actual_seed, options=opts
)
```

Replace `self._state_store = StateStore(initial_state)` with `self._state_store = InProcessStateManager(initial_state)` and update the import.

- [ ] **Step 4: Update `forge/runtime/env_builder.py`**

Change its `ForgeEnv(...)` construction to pass `initial_state_provider=`, and its import of `InitialStateFactory` to `InitialStateProvider`.

- [ ] **Step 5: Run the new test and the full suite**

Run: `.venv/bin/python -m pytest tests/runtime/test_forge_env_contracts.py -v && .venv/bin/python -m pytest -q`
Expected: both PASS. `tests/backend/test_env_loader_determinism.py` is the key regression check here — it exercises the reset path being changed.

- [ ] **Step 6: Commit**

```bash
git add forge/runtime/env.py forge/runtime/env_builder.py tests/runtime/test_forge_env_contracts.py
git commit -m "Bind ForgeEnv to the InitialStateProvider contract

The seed reaches the provider as an explicit keyword instead of through
options or ctx, so a seeded reset and an unseeded one are no longer ambiguous.
InitialStateFactory stays as an alias."
```

---

### Task 14: `ContainerEnvBase` implements `Environment`

**Files:**
- Modify: `forge/envgen/container_env_base.py`
- Test: `tests/envgen/test_container_env_contracts.py` (create)

**Interfaces:**
- Consumes: `Environment`, `HttpStateManager`, `ThresholdTerminationPolicy`, `TaskSuccessRubric`.
- Produces: `ContainerEnvBase(gymnasium.Env, Environment)` exposing the seven required facade members plus `tools` and `transport`.

- [ ] **Step 1: Write the failing test**

```python
# tests/envgen/test_container_env_contracts.py
from __future__ import annotations

import httpx

from forge.contracts import Environment, StateManager, Transport
from forge.envgen.container_env_base import ContainerEnvBase


def _env() -> ContainerEnvBase:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tickets": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ContainerEnvBase("http://env", client=client)


def test_a_container_env_satisfies_the_environment_facade():
    assert isinstance(_env(), Environment)


def test_it_exposes_an_http_state_manager():
    assert isinstance(_env().state, StateManager)


def test_a_container_env_has_a_transport_because_it_is_over_a_wire():
    # False-positive guard: the optional members are optional in general, but
    # this family genuinely has one and must expose it.
    assert isinstance(_env().transport, Transport)


def test_reset_and_step_still_work_over_http():
    env = _env()
    obs, _info = env.reset()
    assert obs == {"tickets": []}
    obs, reward, terminated, truncated, info = env.step({"type": "close_ticket"})
    assert info["status_code"] == 200
    assert reward == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/envgen/test_container_env_contracts.py -v`
Expected: FAIL — `ContainerEnvBase` is not an `Environment`.

- [ ] **Step 3: Make `ContainerEnvBase` implement the facade**

Change the class declaration to `class ContainerEnvBase(gymnasium.Env, Environment):` and add the facade members. Keep every existing method body unchanged — `reset`, `step`, `_observe`, `action_endpoint`, and `compute_reward` all stay as they are. Add:

```python
    @property
    def state(self) -> StateManager:
        return self._state_manager

    @property
    def transport(self) -> Transport:
        return self._transport

    @property
    def task_source(self) -> TaskSource:
        return self._task_source

    @property
    def initial_state(self) -> InitialStateProvider:
        return self._initial_state

    @property
    def observations(self) -> ObservationEncoder:
        return self._observations

    @property
    def backend(self) -> ExecutionBackend:
        return self._backend

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    @property
    def termination(self) -> TerminationPolicy:
        return self._termination
```

Build the collaborators in `__init__` after `self.client` is set, using `HttpStateManager(self.base_url, client=self.client)` for state, `RestTransport(self.base_url, client=self.client)` for transport, `TaskSuccessRubric()` for the rubric, and `MaxStepsTerminationPolicy(max_steps=...)` for termination.

- [ ] **Step 4: Add `RestTransport` to `forge/runtime/http_state.py`'s sibling module**

Create `forge/runtime/rest_transport.py`:

```python
# forge/runtime/rest_transport.py
"""HTTP transport for container-backed environments."""
from __future__ import annotations

import httpx

from forge.contracts import Transport, TransportRequest, TransportResponse


class RestTransport(Transport):
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=15.0)

    def call(self, request: TransportRequest) -> TransportResponse:
        try:
            response = self._client.request(
                request.method,
                f"{self._base_url}{request.target}",
                json=request.payload or None,
                timeout=request.timeout,
            )
        except httpx.HTTPError as exc:
            # In-band, so one wire failure costs a step rather than the episode.
            return TransportResponse(status=0, body={}, error=str(exc))
        body = response.json() if response.content else {}
        return TransportResponse(status=response.status_code, body=body)

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 5: Run the new test and the full suite**

Run: `.venv/bin/python -m pytest tests/envgen/test_container_env_contracts.py -v && .venv/bin/python -m pytest -q`
Expected: both PASS. `tests/envgen/test_container_env_base.py` and `tests/envgen/test_container_seed_determinism.py` are the regression checks.

- [ ] **Step 6: Commit**

```bash
git add forge/envgen/container_env_base.py forge/runtime/rest_transport.py \
        tests/envgen/test_container_env_contracts.py
git commit -m "Make ContainerEnvBase implement the Environment facade

Adds RestTransport, which reports wire failures in-band so a flaky container
costs one step rather than the whole episode."
```

---

### Task 15: The three runners become `EpisodeController`s

**Files:**
- Modify: `forge/envgen/episode_runner.py:98`, `forge/envgen/cli_runner.py:53,102`, `forge/envgen/browser_runner.py:34,110`
- Test: `tests/envgen/test_runners_are_controllers.py` (create)

**Interfaces:**
- Consumes: `EpisodeController` from `forge.contracts`.
- Produces: all three runners subclass `EpisodeController`; `CliEpisodeRunner.run_episode` and `BrowserEpisodeRunner.run_episode` gain `seed: int | None = None`, accepted and ignored.

- [ ] **Step 1: Write the failing test**

```python
# tests/envgen/test_runners_are_controllers.py
from __future__ import annotations

import inspect

import pytest

from forge.contracts import EpisodeController
from forge.envgen.browser_runner import BrowserEpisodeRunner
from forge.envgen.cli_runner import CliEpisodeRunner
from forge.envgen.episode_runner import ContainerEpisodeRunner

RUNNERS = [ContainerEpisodeRunner, CliEpisodeRunner, BrowserEpisodeRunner]


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda r: r.__name__)
def test_every_runner_is_an_episode_controller(runner):
    assert issubclass(runner, EpisodeController)


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda r: r.__name__)
def test_every_runner_accepts_the_same_keywords(runner):
    params = inspect.signature(runner.run_episode).parameters
    assert {"agent", "episode_id", "seed", "jsonl_path"} <= set(params)


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda r: r.__name__)
def test_seed_is_keyword_only(runner):
    # False-positive guard: uniform keywords must not become positional, or a
    # caller could pass jsonl_path into seed.
    params = inspect.signature(runner.run_episode).parameters
    assert params["seed"].kind is inspect.Parameter.KEYWORD_ONLY
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/envgen/test_runners_are_controllers.py -v`
Expected: FAIL — none of the three is a subclass, and two lack `seed`.

- [ ] **Step 3: Edit the three runners**

For each: add `EpisodeController` as a base class and make `run_episode`'s parameters keyword-only after `agent`.

`forge/envgen/episode_runner.py` — `class ContainerEpisodeRunner(EpisodeController):`, and change the signature to:

```python
    def run_episode(
        self,
        agent: ContainerAgentBase,
        *,
        episode_id: str | None = None,
        seed: int | None = None,
        jsonl_path: Path | None = None,
    ) -> EpisodeResult:
```

`forge/envgen/cli_runner.py` — `class CliEpisodeRunner(EpisodeController):` and:

```python
    def run_episode(
        self,
        agent,
        *,
        episode_id: str | None = None,
        seed: int | None = None,
        jsonl_path: Path | None = None,
    ) -> CliEpisodeResult:
        # A CLI sandbox has no seeded reset: the container's filesystem is its
        # initial state. The keyword is accepted for a uniform controller
        # signature and deliberately unused.
        del seed
```

`forge/envgen/browser_runner.py` — the same treatment, with the comment noting a browser session's initial state is the loaded page.

- [ ] **Step 4: Fix the call sites that pass these positionally**

Run: `grep -rn "run_episode(" --include='*.py' forge/ backend/ tests/ | grep -v __pycache__`

Update any call passing `episode_id` or `jsonl_path` positionally to use keywords.

- [ ] **Step 5: Run the new test and the full suite**

Run: `.venv/bin/python -m pytest tests/envgen/test_runners_are_controllers.py -v && .venv/bin/python -m pytest -q`
Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add forge/envgen/episode_runner.py forge/envgen/cli_runner.py \
        forge/envgen/browser_runner.py tests/envgen/test_runners_are_controllers.py
git commit -m "Make the three episode runners EpisodeControllers

All three now take the same keyword-only signature. The CLI and browser runners
accept and ignore seed: their initial state is the container filesystem and the
loaded page, but a uniform signature means callers need no per-family case."
```

---

### Task 16: Code generators emit the contract shape

**Files:**
- Modify: `forge/templates/gym_wrapper.py.j2`
- Modify: `forge/templates/transition.py.j2`
- Modify: `forge/templates/verifier.py.j2`
- Modify: `forge/templates/reward.py.j2`
- Modify: `forge/templates/initial_state.py.j2`
- Test: `tests/compiler/test_generated_package_contracts.py` (create)

**Interfaces:**
- Consumes: every contract from phases 1-2.
- Produces: generated packages whose transitions are `TransitionHandler` subclasses, verifiers are `Verifier` subclasses, rewards are `Rubric` subclasses, and the initial-state factory is an `InitialStateProvider`.

- [ ] **Step 1: Write the failing test**

```python
# tests/compiler/test_generated_package_contracts.py
"""A generated package must satisfy the contracts it is built against."""
from __future__ import annotations

from forge.compiler.generators.gym_wrapper import GymWrapperGenerator
from forge.compiler.generators.initial_state import InitialStateGenerator
from forge.compiler.generators.transition import TransitionGenerator
from forge.extraction.schemas import ActionDef, CompilerInput, EntityDef, FieldDef


def _compiler_input() -> CompilerInput:
    return CompilerInput(
        project_name="ticket_env",
        domain="support",
        entities=[
            EntityDef(name="ticket", fields=[FieldDef(name="id", type="string")])
        ],
        actions=[ActionDef(name="close_ticket", params=[])],
        tasks=[],
    )


def test_generated_transitions_expose_a_handler_class():
    # generate() returns {action_name: source}, one entry per declared action.
    sources = TransitionGenerator().generate(_compiler_input())
    source = sources["close_ticket"]
    assert "class CloseTicketHandler(TransitionHandler):" in source
    assert "def apply(self, state: dict, action: Action, ctx" in source


def test_generated_transitions_keep_the_plain_function():
    # False-positive guard: the function form is what customization overrides
    # and hand-written callers use. Wrapping must not remove it.
    sources = TransitionGenerator().generate(_compiler_input())
    assert "def apply_close_ticket(" in sources["close_ticket"]


def test_generated_wrapper_registers_handler_instances_not_functions():
    source = GymWrapperGenerator().generate(_compiler_input())
    # Negative: registering a bare function would now raise at build time.
    assert 'te.register("close_ticket", CloseTicketHandler())' in source
    assert "apply_close_ticket)" not in source


def test_generated_wrapper_uses_the_provider_keyword():
    source = GymWrapperGenerator().generate(_compiler_input())
    assert "initial_state_provider=" in source
    assert "initial_state_factory=" not in source


def test_generated_initial_state_implements_the_provider_contract():
    source = InitialStateGenerator().generate(_compiler_input())
    assert "InitialStateProvider" in source
    assert "def reset(self, ctx: RuntimeContext, *, seed: int | None, options: dict)" in source
    assert "def create(" not in source
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/compiler/test_generated_package_contracts.py -v`
Expected: FAIL — the templates still emit bare functions and `initial_state_factory=`.

- [ ] **Step 3: Update `forge/templates/transition.py.j2`**

The template owns the whole generated function body, so do not touch it. Leave `apply_{{ action.name }}` exactly as it is and make two additive edits.

Add to the import block at the top:

```jinja
from forge.contracts import Action
from forge.contracts.backend import TransitionHandler
```

Append at the end of the file, after the existing `return TransitionResult(...)`:

```jinja


class {{ action.name | pascal_case }}Handler(TransitionHandler):
    """Contract form of the generated transition.

    The plain function above stays the implementation — it is what
    customization overrides replace and what hand-written callers import.
    """

    def apply(self, state: dict, action: Action, ctx: RuntimeContext) -> TransitionResult:
        return apply_{{ action.name }}(state, action.to_dict(), ctx)
```

Keeping the function as the implementation and the class as a thin wrapper means the body template — which carries all the entity, param, and event logic — is unchanged, so this edit cannot alter generated behavior.

- [ ] **Step 4: Update `forge/templates/gym_wrapper.py.j2`**

Change the imports to pull the handler classes, the registration lines to instantiate them, and the `ForgeEnv(...)` call to use `initial_state_provider=`:

```jinja
{% for action in actions %}
from generated_envs.{{ project_name }}.transitions.{{ action.name }} import {{ action.name | pascal_case }}Handler
{% endfor %}
...
    te = TransitionEngine()
{% for action in actions %}
    te.register("{{ action.name }}", {{ action.name | pascal_case }}Handler())
{% endfor %}
...
    return ForgeEnv(
        env_spec=spec,
        initial_state_provider={{ project_name | pascal_case }}InitialStateFactory(),
        transition_engine=te,
        verifier_engine=ve,
        reward_engine=re,
    )
```

- [ ] **Step 5: Update `verifier.py.j2` and `reward.py.j2` the same additive way**

Both keep their generated function as the implementation and gain a thin contract wrapper appended at the end.

`verifier.py.j2` — add `from forge.contracts import Verifier, VerificationResult` to the imports and append:

```jinja


class {{ task.name | pascal_case }}Verifier(Verifier):
    def verify(self, state: dict, trajectory, task) -> VerificationResult:
        return verify_{{ task.name }}(state, trajectory, task)
```

`reward.py.j2` — add `from forge.contracts import RewardBreakdown, Rubric` and append:

```jinja


class {{ task.name | pascal_case }}Rubric(Rubric):
    def score(self, state, trajectory, verifier_results, task) -> RewardBreakdown:
        return compute_{{ task.name }}_reward(state, trajectory, verifier_results, task)
```

- [ ] **Step 6: Update `initial_state.py.j2` — the one non-additive template edit**

This template has no wrapper option, because the facade calls `reset` by name. Change two lines and leave the entire state-literal body untouched:

```jinja
from forge.contracts import InitialStateProvider
from forge.runtime.context import RuntimeContext


class {{ project_name | pascal_case }}InitialStateFactory(InitialStateProvider):
    def reset(self, ctx: RuntimeContext, *, seed: int | None, options: dict) -> dict:
```

The generated body uses `ctx.id_generator` and `ctx.clock` and never reads a seed, so `seed` is accepted and unused here. `ForgeEnv` derives `ctx` from the same seed it passes, so behavior is identical.

No change is needed in `forge/compiler/generators/` — each generator passes the template its own variables and returns whatever the template renders, so appending classes to the templates changes nothing in the generator classes themselves. Verify with `.venv/bin/python -m pytest tests/compiler/test_generators.py -v` before moving on.

- [ ] **Step 7: Run the new test and the full suite**

Run: `.venv/bin/python -m pytest tests/compiler/ -v && .venv/bin/python -m pytest -q`
Expected: both PASS.

- [ ] **Step 8: Commit**

```bash
git add forge/templates forge/compiler/generators tests/compiler/test_generated_package_contracts.py
git commit -m "Emit contract-conforming code from the package generators

Generated transitions, verifiers, rewards, and initial-state factories are now
contract subclasses, so the tightened registries accept them and a malformed
generated handler fails at build time rather than during a rollout."
```

---

### Task 17: Migrate `examples/gmail_env`

**Files:**
- Modify: `examples/gmail_env/gym_wrapper.py`, `initial_state.py`, `transitions/*.py`, `verifiers/*.py`, `rewards/base.py`
- Test: `tests/runtime/test_gmail_example_contracts.py` (create)

**Interfaces:**
- Consumes: everything from Tasks 9-16.
- Produces: `examples/gmail_env` in the generator's new output shape.

`generated_envs/` holds only `.gitkeep`, so this is the only checked-in package in the template's shape and the whole migration surface.

- [ ] **Step 1: Write the failing test**

```python
# tests/runtime/test_gmail_example_contracts.py
"""The worked example must stay in the shape the generator now emits."""
from __future__ import annotations

from forge.contracts import InitialStateProvider, Rubric, Verifier
from forge.contracts.backend import TransitionHandler
from examples.gmail_env.gym_wrapper import build_gmail_env
from examples.gmail_env.initial_state import GmailEnvInitialStateFactory
from examples.gmail_env.transitions.archive_email import ArchiveEmailHandler


def test_the_initial_state_factory_is_a_provider():
    assert isinstance(GmailEnvInitialStateFactory(), InitialStateProvider)


def test_a_transition_is_a_handler():
    assert isinstance(ArchiveEmailHandler(), TransitionHandler)


def test_the_example_env_builds_and_resets():
    env = build_gmail_env()
    obs, info = env.reset(seed=3)
    assert isinstance(obs, dict)
    assert info["seed"] == 3


def test_the_example_env_takes_a_step():
    # False-positive guard: building is not the same as working. Exercise a
    # real registered action end to end.
    env = build_gmail_env()
    env.reset(seed=3)
    obs, _reward, _term, _trunc, info = env.step(
        {"type": "archive_email", "email_id": "e_0001"}
    )
    assert "error" not in info
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/runtime/test_gmail_example_contracts.py -v`
Expected: FAIL with `ImportError: cannot import name 'ArchiveEmailHandler'`

- [ ] **Step 3: Migrate the package**

Apply to `examples/gmail_env` exactly the shape the templates now emit: each `transitions/<action>.py` gains a `<Action>Handler(TransitionHandler)` class wrapping its existing body, each `verifiers/<task>.py` gains a `<Task>Verifier(Verifier)`, `rewards/base.py` gains a `Rubric` subclass, `initial_state.py`'s `create` becomes `reset(self, ctx, *, seed, options)`, and `gym_wrapper.py` registers instances and passes `initial_state_provider=`.

- [ ] **Step 4: Run the new test and the full suite**

Run: `.venv/bin/python -m pytest tests/runtime/test_gmail_example_contracts.py -v && .venv/bin/python -m pytest -q`
Expected: both PASS.

- [ ] **Step 5: Commit — phase 2 gate**

```bash
git add examples/gmail_env tests/runtime/test_gmail_example_contracts.py
git commit -m "Migrate the gmail example to the contract shape

The only checked-in package in the generator's output shape, so this completes
the phase 2 rebase."
```

---

# Phase 3 — envgen adoption

Phase 3 is what makes the contracts help generation rather than just tidy the runtime.

### Task 18: Specialist prompts cite the contracts

**Files:**
- Modify: `forge/envgen/agents/app_generator.py` (the `_BACKEND_SYSTEM` state-management section)
- Modify: `forge/envgen/agents/state_bridge.py`
- Modify: `forge/envgen/agents/reward.py`
- Test: `tests/envgen/test_prompts_cite_contracts.py` (create)

**Interfaces:**
- Consumes: the contract names from phase 1.
- Produces: prompts naming the exact contracts and method signatures generated code must satisfy.

- [ ] **Step 1: Write the failing test**

```python
# tests/envgen/test_prompts_cite_contracts.py
"""Specialists must name the contracts, not describe them in prose."""
from __future__ import annotations

from forge.envgen.agents.app_generator import AppGeneratorPrompts


def test_the_backend_prompt_names_the_state_manager_contract():
    prompt = AppGeneratorPrompts.backend(with_ui=False)
    assert "StateManager" in prompt
    assert "reset_state" in prompt


def test_the_state_bridge_prompt_names_the_environment_facade():
    from forge.envgen.agents.state_bridge import StateBridgePrompts

    assert "Environment" in StateBridgePrompts.SYSTEM
    assert "forge.contracts" in StateBridgePrompts.SYSTEM


def test_the_reward_prompt_names_the_rubric_contract():
    from forge.envgen.agents.reward import RewardPrompts

    assert "Rubric" in RewardPrompts.SYSTEM
    assert "def score(" in RewardPrompts.SYSTEM
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/envgen/test_prompts_cite_contracts.py -v`
Expected: FAIL — none of the three prompts mentions a contract.

- [ ] **Step 3: Add a contract section to each prompt**

Add to `_BACKEND_SYSTEM`, inside the existing "STATE-MANAGEMENT CLASS" block:

```
"  The state class mirrors the forge.contracts.StateManager contract:\n"
"    get() -> dict, apply(state) -> None, hash() -> str, plus reset_state()\n"
"    and seed_state(seed). Keeping the same names means the generated app and\n"
"    the Forge runtime describe state the same way.\n"
```

Add to the state-bridge system prompt the requirement that the generated `ContainerForgeEnv` subclass `forge.contracts.Environment` and supply the seven required members, and to the reward prompt that the generated reward be a `forge.contracts.Rubric` subclass with `def score(self, state, trajectory, verifier_results, task) -> RewardBreakdown`.

Read each file first to place the text inside the existing prompt structure rather than appending to the end.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/envgen/test_prompts_cite_contracts.py -v && .venv/bin/python -m pytest -q`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add forge/envgen/agents tests/envgen/test_prompts_cite_contracts.py
git commit -m "Point the generation specialists at the contracts

The specialists described these concerns in prose, which left each generation
free to invent its own shape. They now name the contract and its signature."
```

---

### Task 19: The reviewer gate checks contract conformance

**Files:**
- Modify: `forge/envgen/agents/reviewer.py`
- Test: `tests/envgen/test_reviewer_contract_check.py` (create)

**Interfaces:**
- Consumes: `ReviewIssue`, `ReviewSeverity` from `forge/envgen/agents/reviewer.py`.
- Produces: a static AST check in `ReviewerAgent.run` that the state bridge subclasses `Environment` and the reward code defines a `Rubric` subclass with a `score` method.

- [ ] **Step 1: Write the failing test**

```python
# tests/envgen/test_reviewer_contract_check.py
from __future__ import annotations

import pytest

from forge.envgen.agents.reviewer import ReviewerAgent
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


@pytest.mark.asyncio
async def test_a_conforming_generation_passes_the_contract_check():
    bus = await _bus(_CONFORMING_BRIDGE, _CONFORMING_REWARD)
    await ReviewerAgent(semantic_review=False).run(_ctx(), bus)
    review = bus.get("review_report")
    assert not [i for i in review.issues if i.category == "contract"]


@pytest.mark.asyncio
async def test_a_bridge_that_ignores_the_facade_is_rejected():
    # Negative: the check must actually fire.
    bus = await _bus("class ContainerForgeEnv:\n    pass\n", _CONFORMING_REWARD)
    await ReviewerAgent(semantic_review=False).run(_ctx(), bus)
    review = bus.get("review_report")
    assert any(i.category == "contract" for i in review.issues)
    assert review.approved is False


@pytest.mark.asyncio
async def test_a_rubric_without_score_is_rejected():
    reward = "from forge.contracts import Rubric\nclass R(Rubric):\n    pass\n"
    bus = await _bus(_CONFORMING_BRIDGE, reward)
    await ReviewerAgent(semantic_review=False).run(_ctx(), bus)
    assert any(i.category == "contract" for i in bus.get("review_report").issues)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/envgen/test_reviewer_contract_check.py -v`
Expected: FAIL — no `contract` category issue is ever produced.

- [ ] **Step 3: Add the check to `ReviewerAgent.run`**

After the existing Forge-endpoint loop, add:

```python
        # Contract conformance. Static, like the determinism gate — it asks
        # whether the generated code declares the shape the runtime requires,
        # which the semantic reviewer cannot check reliably.
        issues.extend(self._contract_issues(
            artifacts["state_bridge_code"] or "",
            artifacts["reward_fn_code"] or "",
        ))
```

And the helper:

```python
    @staticmethod
    def _subclasses(source: str, base: str) -> list[ast.ClassDef]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []  # a syntax issue is already reported by the parse check
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(
                (isinstance(b, ast.Name) and b.id == base)
                or (isinstance(b, ast.Attribute) and b.attr == base)
                for b in node.bases
            )
        ]

    def _contract_issues(
        self, state_bridge_code: str, reward_fn_code: str
    ) -> list[ReviewIssue]:
        issues: list[ReviewIssue] = []

        if state_bridge_code and not self._subclasses(state_bridge_code, "Environment"):
            issues.append(self._error(
                "contract",
                "The state bridge must subclass forge.contracts.Environment",
                "state_bridge_code",
            ))

        if reward_fn_code:
            rubrics = self._subclasses(reward_fn_code, "Rubric")
            if not rubrics:
                issues.append(self._error(
                    "contract",
                    "The reward must subclass forge.contracts.Rubric",
                    "reward_fn_code",
                ))
            elif not any(
                isinstance(item, ast.FunctionDef) and item.name == "score"
                for rubric in rubrics
                for item in rubric.body
            ):
                issues.append(self._error(
                    "contract",
                    "The Rubric subclass must define score(); without it the "
                    "reward cannot be registered",
                    "reward_fn_code",
                ))

        return issues
```

- [ ] **Step 4: Route contract findings to the right specialist**

In `forge/envgen/repair.py`, `FindingRouter.route` already maps `state_bridge_code` and `reward_fn_code` to their producers via `self._producers`, so contract findings route correctly with no change. Confirm with:

Run: `.venv/bin/python -m pytest tests/envgen/test_repair_loop.py -v`
Expected: PASS

- [ ] **Step 5: Run the new test and the full suite**

Run: `.venv/bin/python -m pytest tests/envgen/test_reviewer_contract_check.py -v && .venv/bin/python -m pytest -q`
Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add forge/envgen/agents/reviewer.py tests/envgen/test_reviewer_contract_check.py
git commit -m "Gate generated environments on contract conformance

A static AST check, like the determinism gate: it asks whether generated code
declares the shape the runtime requires. Findings route to the state-bridge and
reward specialists through the existing repair loop."
```

---

### Task 20: Document the extension surface

**Files:**
- Modify: `README.md`
- Modify: `specs/2026-08-30-forge-contracts-design.md` (status line)

- [ ] **Step 1: Add a contracts section to the README**

Insert after the "Environment Types" table:

```markdown
### Environment Contracts

Every Forge environment implements a shared set of interfaces in
`forge/contracts/`, one per concern:

| Contract | Answers |
|---|---|
| `TaskSource` | What problems should the model solve? |
| `InitialStateProvider` | How is per-episode state set up? |
| `PromptTemplate` | How is the task presented to the model? |
| `ToolProvider` | What can the model do? |
| `ObservationEncoder` | What does the model see back? |
| `ExecutionBackend` | Where do actions actually run? |
| `StateManager` | How is state tracked across turns? |
| `Rubric` / `Verifier` | How is the model's behavior scored? |
| `TerminationPolicy` | How does the episode end? |
| `EpisodeController` | Who drives the multi-turn loop? |
| `Transport` | How does the model talk to the environment? |

`Environment` composes ten of them; `EpisodeController` stays separate because a
controller drives an environment from outside. To author an environment by
hand, implement `Environment` and hand it to any controller.

Environments generated before this release use the pre-contract shape and must
be regenerated.
```

- [ ] **Step 2: Update the spec status**

Change the spec's status line to `**Status:** Implemented (phases 1-3)`.

- [ ] **Step 3: Run the full suite one final time**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add README.md specs/2026-08-30-forge-contracts-design.md
git commit -m "Document forge/contracts as the environment extension surface"
```

---

## Self-review

**Spec coverage.** Every section of `specs/2026-08-30-forge-contracts-design.md` maps to a task: layout and types → Task 1; the eleven contracts → Tasks 2-7; the facade → Task 8; the phase-2 rebase list, items 1-8 → Tasks 9-17; phase 3 → Tasks 18-20. The import-direction rule is enforced from Task 1 onward. The `TYPE_CHECKING` resolution for `RuntimeContext`, `Trajectory`, and `TransitionResult` appears in Tasks 3, 5, and 6.

**Corrections made during self-review.** Two references in the first draft of Task 16 did not survive checking against the code. `TransitionGenerator.generate` takes only `compiler_input` and returns `{action_name: source}`, not the two-argument form the draft test called. And the templates own their entire generated body, so the draft's `{{ body }}` splice did not exist; Task 16 now makes purely additive template edits — the generated function stays the implementation and a thin contract class wraps it — except in `initial_state.py.j2`, where the facade calls `reset` by name and a rename is unavoidable. The additive form means the transition, verifier, and reward templates cannot change generated behavior.

**Deviation from the spec, deliberate.** The spec listed `AgentAdapter` as moving to `contracts/types.py`; this plan does that in Task 1 but does **not** change `forge/runtime/agents/base.py` to re-export it, because nothing in the plan imports it from there. If a later task needs it, add the one-line re-export then.

**Type consistency.** `InitialStateProvider.reset(ctx, *, seed, options)` is identical in Tasks 3, 13, 16, and 17. `Rubric.score(state, trajectory, verifier_results, task)` is identical in Tasks 6, 11, 16, 18, and 19. `TransitionHandler.apply(state, action, ctx)` is identical in Tasks 5, 11, 16, and 17. `StateManager.get/apply/hash/snapshot/restore` is identical in Tasks 2, 9, and 14.

**Known ordering constraint.** Tasks 11 and 12 must land in one commit: tightening `register` breaks the customization hooks until they wrap. Task 11 step 8 says so explicitly.
