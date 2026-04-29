# M1: Runtime Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic Gymnasium-compatible runtime kernel and a hand-written Gmail environment that proves it works end-to-end.

**Architecture:** The kernel is a pure Python package (`forge/runtime/`) with no web, LLM, or compiler dependencies. A `RuntimeContext` owns all non-determinism (clock, RNG, ID generator) so that the same seed + same action sequence always produces an identical trajectory hash. `ForgeEnv` subclasses `gym.Env` and wires together `TransitionEngine`, `VerifierEngine`, `RewardEngine`, and `TrajectoryStore`. The Gmail environment in `examples/gmail_env/` is the reference implementation and M1 acceptance test.

**Tech Stack:** Python 3.11+, Gymnasium 0.29+, Pydantic v2, pytest, hatchling

---

## File Map

```
forge/
  __init__.py
  runtime/
    __init__.py
    context.py        RuntimeContext, SimClock, IDGenerator
    state.py          StateStore (mutable world state + sha256 hash)
    diff.py           compute_diff (flat key-path state diff)
    snapshot.py       StepSnapshot, Event, InvalidActionError, EnvironmentSpec
    verification.py   CheckResult, VerificationResult
    reward.py         RewardComponent, RewardBreakdown, RewardEngine
    transition.py     TransitionResult, TransitionEngine
    action.py         ActionValidator
    verifier.py       VerifierEngine
    trajectory.py     Trajectory, TrajectoryStore
    env.py            ForgeEnv (gym.Env subclass)
examples/
  gmail_env/
    __init__.py
    state_models.py       User, Email, Thread, Label field definitions
    action_models.py      Typed dicts for all 6 Gmail actions
    initial_state.py      GmailInitialStateFactory
    transitions/
      __init__.py
      reply_email.py
      send_email.py
      archive_email.py
      apply_label.py
      mark_read.py
      escalate_thread.py
    verifiers/
      __init__.py
      reply_to_customer.py
      label_urgent_request.py
      archive_newsletter.py
      escalate_billing_complaint.py
    rewards/
      __init__.py
      base.py
    gym_wrapper.py        build_gmail_env() factory
tests/
  runtime/
    test_context.py
    test_state.py
    test_diff.py
    test_transition.py
    test_verifier.py
    test_reward.py
    test_trajectory.py
    test_env.py
  gmail_env/
    test_transitions.py
    test_verifiers.py
    test_determinism.py
pyproject.toml
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `forge/__init__.py`
- Create: `forge/runtime/__init__.py`
- Create: `examples/__init__.py`
- Create: `examples/gmail_env/__init__.py`
- Create: `examples/gmail_env/transitions/__init__.py`
- Create: `examples/gmail_env/verifiers/__init__.py`
- Create: `examples/gmail_env/rewards/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/runtime/__init__.py`
- Create: `tests/gmail_env/__init__.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "forge"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "gymnasium>=0.29.0",
    "pydantic>=2.0.0",
    "numpy>=1.26.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.hatch.build.targets.wheel]
packages = ["forge", "examples"]
```

- [ ] **Step 2: Create all empty `__init__.py` files**

```bash
touch forge/__init__.py
touch forge/runtime/__init__.py
touch examples/__init__.py
touch "examples/gmail_env/__init__.py"
touch "examples/gmail_env/transitions/__init__.py"
touch "examples/gmail_env/verifiers/__init__.py"
touch "examples/gmail_env/rewards/__init__.py"
touch tests/__init__.py
touch tests/runtime/__init__.py
touch tests/gmail_env/__init__.py
```

- [ ] **Step 3: Install package in dev mode**

```bash
pip install -e ".[dev]"
```

Expected: `Successfully installed forge-0.1.0`

- [ ] **Step 4: Verify import works**

```bash
python -c "import forge; print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml forge/ examples/ tests/
git commit -m "feat: add project scaffold and package structure"
```

---

## Task 2: RuntimeContext

**Files:**
- Create: `forge/runtime/context.py`
- Create: `tests/runtime/test_context.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/runtime/test_context.py
from forge.runtime.context import RuntimeContext


def test_same_seed_produces_same_rng_sequence():
    ctx1 = RuntimeContext(seed=42)
    ctx2 = RuntimeContext(seed=42)
    assert [ctx1.rng.random() for _ in range(10)] == [ctx2.rng.random() for _ in range(10)]


def test_different_seeds_produce_different_sequences():
    ctx1 = RuntimeContext(seed=1)
    ctx2 = RuntimeContext(seed=2)
    assert ctx1.rng.random() != ctx2.rng.random()


def test_id_generator_is_sequential_and_deterministic():
    ctx = RuntimeContext(seed=0)
    assert ctx.id_generator.next("email") == "email_0000"
    assert ctx.id_generator.next("email") == "email_0001"
    assert ctx.id_generator.next("thread") == "thread_0000"


def test_same_seed_produces_same_id_sequence():
    ctx1 = RuntimeContext(seed=99)
    ctx2 = RuntimeContext(seed=99)
    ids1 = [ctx1.id_generator.next("x") for _ in range(5)]
    ids2 = [ctx2.id_generator.next("x") for _ in range(5)]
    assert ids1 == ids2


def test_clock_starts_at_epoch_and_advances():
    ctx = RuntimeContext(seed=0)
    t0 = ctx.clock.now()
    ctx.clock.advance()
    t1 = ctx.clock.now()
    assert t1 > t0


def test_same_seed_produces_same_clock_sequence():
    ctx1 = RuntimeContext(seed=5)
    ctx2 = RuntimeContext(seed=5)
    for _ in range(3):
        ctx1.clock.advance()
        ctx2.clock.advance()
    assert ctx1.clock.now() == ctx2.clock.now()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/runtime/test_context.py -v
```

Expected: `ModuleNotFoundError: No module named 'forge.runtime.context'`

- [ ] **Step 3: Implement `forge/runtime/context.py`**

```python
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class SimClock:
    _base: int = 1_700_000_000  # fixed epoch, not wall-clock
    _tick: int = 0

    def now(self) -> datetime:
        return datetime.fromtimestamp(self._base + self._tick, tz=timezone.utc)

    def advance(self, seconds: int = 1) -> None:
        self._tick += seconds


@dataclass
class IDGenerator:
    _counters: dict[str, int] = field(default_factory=dict)

    def next(self, prefix: str) -> str:
        count = self._counters.get(prefix, 0)
        self._counters[prefix] = count + 1
        return f"{prefix}_{count:04d}"


@dataclass
class RuntimeContext:
    seed: int
    actor_id: str = "agent"
    clock: SimClock = field(default_factory=SimClock)
    id_generator: IDGenerator = field(default_factory=IDGenerator)
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/runtime/test_context.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add forge/runtime/context.py tests/runtime/test_context.py
git commit -m "feat: add RuntimeContext with deterministic clock, RNG, and ID generator"
```

---

## Task 3: StateStore

**Files:**
- Create: `forge/runtime/state.py`
- Create: `tests/runtime/test_state.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/runtime/test_state.py
import copy
from forge.runtime.state import StateStore


SAMPLE_STATE = {
    "emails": {
        "e_0000": {"id": "e_0000", "labels": ["inbox"], "archived": False}
    },
    "users": {
        "u_0000": {"id": "u_0000", "email": "agent@example.com"}
    },
}


def test_get_returns_deep_copy():
    store = StateStore(SAMPLE_STATE)
    s1 = store.get()
    s1["emails"]["e_0000"]["archived"] = True
    s2 = store.get()
    assert s2["emails"]["e_0000"]["archived"] is False


def test_apply_updates_state():
    store = StateStore(SAMPLE_STATE)
    new_state = copy.deepcopy(SAMPLE_STATE)
    new_state["emails"]["e_0000"]["archived"] = True
    store.apply(new_state)
    assert store.get()["emails"]["e_0000"]["archived"] is True


def test_hash_is_stable_for_same_state():
    store = StateStore(SAMPLE_STATE)
    assert store.hash() == store.hash()


def test_hash_changes_after_mutation():
    store = StateStore(SAMPLE_STATE)
    h1 = store.hash()
    new_state = copy.deepcopy(SAMPLE_STATE)
    new_state["emails"]["e_0000"]["archived"] = True
    store.apply(new_state)
    assert store.hash() != h1


def test_hash_starts_with_sha256_prefix():
    store = StateStore(SAMPLE_STATE)
    assert store.hash().startswith("sha256:")


def test_same_state_always_produces_same_hash():
    s1 = StateStore(SAMPLE_STATE)
    s2 = StateStore(SAMPLE_STATE)
    assert s1.hash() == s2.hash()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/runtime/test_state.py -v
```

Expected: `ModuleNotFoundError: No module named 'forge.runtime.state'`

- [ ] **Step 3: Implement `forge/runtime/state.py`**

```python
import copy
import hashlib
import json


class StateStore:
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/runtime/test_state.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add forge/runtime/state.py tests/runtime/test_state.py
git commit -m "feat: add StateStore with deterministic sha256 hashing"
```

---

## Task 4: StateDiff Engine

**Files:**
- Create: `forge/runtime/diff.py`
- Create: `tests/runtime/test_diff.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/runtime/test_diff.py
from forge.runtime.diff import compute_diff


def test_no_change_returns_empty_diff():
    state = {"emails": {"e_0": {"id": "e_0", "archived": False}}}
    diff = compute_diff(state, state)
    assert diff == {"added": {}, "changed": {}, "removed": {}}


def test_added_entity_appears_in_added():
    before = {"emails": {"e_0": {"id": "e_0"}}}
    after = {"emails": {"e_0": {"id": "e_0"}, "e_1": {"id": "e_1"}}}
    diff = compute_diff(before, after)
    assert "emails.e_1" in diff["added"]
    assert diff["added"]["emails.e_1"] == {"id": "e_1"}


def test_removed_entity_appears_in_removed():
    before = {"emails": {"e_0": {"id": "e_0"}, "e_1": {"id": "e_1"}}}
    after = {"emails": {"e_0": {"id": "e_0"}}}
    diff = compute_diff(before, after)
    assert "emails.e_1" in diff["removed"]


def test_changed_field_appears_in_changed_with_before_and_after():
    before = {"emails": {"e_0": {"id": "e_0", "labels": ["inbox"]}}}
    after = {"emails": {"e_0": {"id": "e_0", "labels": ["inbox", "urgent"]}}}
    diff = compute_diff(before, after)
    assert "emails.e_0.labels" in diff["changed"]
    assert diff["changed"]["emails.e_0.labels"]["before"] == ["inbox"]
    assert diff["changed"]["emails.e_0.labels"]["after"] == ["inbox", "urgent"]


def test_multiple_changes_all_captured():
    before = {
        "emails": {"e_0": {"id": "e_0", "archived": False, "labels": ["inbox"]}},
        "threads": {"t_0": {"id": "t_0", "escalated": False}},
    }
    after = {
        "emails": {"e_0": {"id": "e_0", "archived": True, "labels": ["inbox"]}},
        "threads": {"t_0": {"id": "t_0", "escalated": True}},
    }
    diff = compute_diff(before, after)
    assert "emails.e_0.archived" in diff["changed"]
    assert "threads.t_0.escalated" in diff["changed"]


def test_unchanged_fields_not_in_changed():
    before = {"emails": {"e_0": {"id": "e_0", "archived": False, "labels": ["inbox"]}}}
    after = {"emails": {"e_0": {"id": "e_0", "archived": True, "labels": ["inbox"]}}}
    diff = compute_diff(before, after)
    assert "emails.e_0.id" not in diff["changed"]
    assert "emails.e_0.labels" not in diff["changed"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/runtime/test_diff.py -v
```

Expected: `ModuleNotFoundError: No module named 'forge.runtime.diff'`

- [ ] **Step 3: Implement `forge/runtime/diff.py`**

```python
def compute_diff(before: dict, after: dict) -> dict:
    added: dict = {}
    changed: dict = {}
    removed: dict = {}

    all_collections = set(before.keys()) | set(after.keys())

    for collection in all_collections:
        before_col = before.get(collection, {})
        after_col = after.get(collection, {})

        before_ids = set(before_col.keys())
        after_ids = set(after_col.keys())

        for entity_id in after_ids - before_ids:
            added[f"{collection}.{entity_id}"] = after_col[entity_id]

        for entity_id in before_ids - after_ids:
            removed[f"{collection}.{entity_id}"] = before_col[entity_id]

        for entity_id in before_ids & after_ids:
            b_entity = before_col[entity_id]
            a_entity = after_col[entity_id]
            all_fields = set(b_entity.keys()) | set(a_entity.keys())
            for field in all_fields:
                b_val = b_entity.get(field)
                a_val = a_entity.get(field)
                if b_val != a_val:
                    changed[f"{collection}.{entity_id}.{field}"] = {
                        "before": b_val,
                        "after": a_val,
                    }

    return {"added": added, "changed": changed, "removed": removed}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/runtime/test_diff.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add forge/runtime/diff.py tests/runtime/test_diff.py
git commit -m "feat: add StateDiff engine with flat key-path format"
```

---

## Task 5: Core Schemas

**Files:**
- Create: `forge/runtime/snapshot.py`
- Create: `forge/runtime/verification.py`
- Create: `forge/runtime/reward.py`

No separate tests — these are Pydantic models; their behaviour is covered by integration tests in later tasks.

- [ ] **Step 1: Write `forge/runtime/snapshot.py`**

```python
from __future__ import annotations
from pydantic import BaseModel


class InvalidActionError(Exception):
    def __init__(self, detail: str, code: str = "INVALID_ACTION") -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail

    def to_dict(self) -> dict:
        return {"error": "INVALID_ACTION", "code": self.code, "detail": self.detail}


class EnvironmentSpec(BaseModel):
    name: str
    domain: str
    max_steps: int = 50
    default_task: dict | None = None


class StepSnapshot(BaseModel):
    episode_id: str
    step_index: int
    state_hash_before: str
    state_hash_after: str
    action: dict
    events: list[dict]
    reward: float
    verifier_results: list[dict]
    diff: dict
    terminated: bool
    truncated: bool
```

- [ ] **Step 2: Write `forge/runtime/verification.py`**

```python
from __future__ import annotations
from pydantic import BaseModel


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
    def from_checks(cls, verifier_id: str, checks: list[CheckResult]) -> "VerificationResult":
        passed = all(c.passed for c in checks)
        score = sum(c.score for c in checks) / len(checks) if checks else 0.0
        return cls(verifier_id=verifier_id, passed=passed, score=score, checks=checks)
```

- [ ] **Step 3: Write `forge/runtime/reward.py`**

```python
from __future__ import annotations
from typing import Callable
from pydantic import BaseModel


class RewardComponent(BaseModel):
    name: str
    value: float


class RewardBreakdown(BaseModel):
    total_reward: float
    components: list[RewardComponent]


class RewardEngine:
    def __init__(self) -> None:
        self._task_fns: dict[str, Callable] = {}
        self._default_fn: Callable | None = None

    def register(self, task_name: str, fn: Callable) -> None:
        self._task_fns[task_name] = fn

    def set_default(self, fn: Callable) -> None:
        self._default_fn = fn

    def compute(
        self,
        state: dict,
        trajectory: "Trajectory",
        verifier_results: list,
        task: dict | None = None,
    ) -> RewardBreakdown:
        task_name = task.get("name") if task else None
        fn = self._task_fns.get(task_name) if task_name else None
        fn = fn or self._default_fn

        if fn is None:
            passed = any(vr.passed for vr in verifier_results)
            value = 1.0 if passed else 0.0
            return RewardBreakdown(
                total_reward=value,
                components=[RewardComponent(name="task_success", value=value)],
            )

        return fn(state, trajectory, verifier_results, task)
```

- [ ] **Step 4: Verify imports work**

```bash
python -c "from forge.runtime.snapshot import StepSnapshot, EnvironmentSpec, InvalidActionError; print('ok')"
python -c "from forge.runtime.verification import VerificationResult, CheckResult; print('ok')"
python -c "from forge.runtime.reward import RewardEngine, RewardBreakdown; print('ok')"
```

Expected: three lines of `ok`

- [ ] **Step 5: Commit**

```bash
git add forge/runtime/snapshot.py forge/runtime/verification.py forge/runtime/reward.py
git commit -m "feat: add core schemas (StepSnapshot, VerificationResult, RewardBreakdown)"
```

---

## Task 6: TransitionEngine

**Files:**
- Create: `forge/runtime/transition.py`
- Create: `tests/runtime/test_transition.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/runtime/test_transition.py
import copy
import pytest
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError
from forge.runtime.transition import TransitionEngine, TransitionResult


def make_noop_transition(state, action, ctx):
    return TransitionResult(state=copy.deepcopy(state), events=[])


def make_mutation_transition(state, action, ctx):
    new_state = copy.deepcopy(state)
    new_state["counter"] = new_state.get("counter", 0) + 1
    return TransitionResult(
        state=new_state,
        events=[{"type": "counter_incremented", "entity_id": "counter"}],
    )


def test_registered_action_dispatches_correctly():
    engine = TransitionEngine()
    engine.register("increment", make_mutation_transition)
    ctx = RuntimeContext(seed=0)
    result = engine.apply({"counter": 0}, {"type": "increment"}, ctx)
    assert result.state["counter"] == 1
    assert result.events[0]["type"] == "counter_incremented"


def test_unknown_action_raises_invalid_action_error():
    engine = TransitionEngine()
    ctx = RuntimeContext(seed=0)
    with pytest.raises(InvalidActionError) as exc_info:
        engine.apply({}, {"type": "nonexistent"}, ctx)
    assert exc_info.value.code == "UNKNOWN_ACTION_TYPE"


def test_action_types_returns_registered_types():
    engine = TransitionEngine()
    engine.register("a", make_noop_transition)
    engine.register("b", make_noop_transition)
    assert engine.action_types == {"a", "b"}


def test_transition_does_not_mutate_original_state():
    engine = TransitionEngine()
    engine.register("increment", make_mutation_transition)
    ctx = RuntimeContext(seed=0)
    original = {"counter": 0}
    engine.apply(original, {"type": "increment"}, ctx)
    assert original["counter"] == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/runtime/test_transition.py -v
```

Expected: `ModuleNotFoundError: No module named 'forge.runtime.transition'`

- [ ] **Step 3: Implement `forge/runtime/transition.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError


@dataclass
class TransitionResult:
    state: dict
    events: list[dict] = field(default_factory=list)


class TransitionEngine:
    def __init__(self) -> None:
        self._handlers: dict[str, Callable] = {}

    def register(self, action_type: str, handler: Callable) -> None:
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
        return handler(state, action, ctx)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/runtime/test_transition.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add forge/runtime/transition.py tests/runtime/test_transition.py
git commit -m "feat: add TransitionEngine with handler registration and dispatch"
```

---

## Task 7: ActionValidator

**Files:**
- Create: `forge/runtime/action.py`

Tests for this are folded into the `ForgeEnv` tests in Task 11.

- [ ] **Step 1: Implement `forge/runtime/action.py`**

```python
from __future__ import annotations


class ActionValidator:
    def __init__(self, valid_action_types: set[str]) -> None:
        self._valid_types = valid_action_types

    def validate(self, action: object) -> dict | None:
        if not isinstance(action, dict):
            return {
                "error": "INVALID_ACTION",
                "code": "INVALID_FORMAT",
                "detail": f"Action must be a dict, got {type(action).__name__}",
            }
        if "type" not in action:
            return {
                "error": "INVALID_ACTION",
                "code": "MISSING_TYPE",
                "detail": "Action must have a 'type' field",
            }
        if action["type"] not in self._valid_types:
            return {
                "error": "INVALID_ACTION",
                "code": "UNKNOWN_TYPE",
                "detail": (
                    f"Unknown action type: '{action['type']}'. "
                    f"Valid types: {sorted(self._valid_types)}"
                ),
            }
        return None
```

- [ ] **Step 2: Verify import**

```bash
python -c "from forge.runtime.action import ActionValidator; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add forge/runtime/action.py
git commit -m "feat: add ActionValidator"
```

---

## Task 8: VerifierEngine

**Files:**
- Create: `forge/runtime/verifier.py`
- Create: `tests/runtime/test_verifier.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/runtime/test_verifier.py
from forge.runtime.verification import CheckResult, VerificationResult
from forge.runtime.verifier import VerifierEngine


def passing_verifier(state, trajectory, task):
    return VerificationResult.from_checks(
        "always_pass", [CheckResult(name="check", passed=True, score=1.0)]
    )


def failing_verifier(state, trajectory, task):
    return VerificationResult.from_checks(
        "always_fail",
        [CheckResult(name="check", passed=False, score=0.0, evidence="never passes")],
    )


def test_registered_verifier_runs():
    engine = VerifierEngine()
    engine.register("always_pass", passing_verifier)
    results = engine.run_all({}, None, {"name": "t", "verifier_id": "always_pass"})
    assert len(results) == 1
    assert results[0].passed is True


def test_no_task_returns_empty_results():
    engine = VerifierEngine()
    engine.register("v", passing_verifier)
    assert engine.run_all({}, None, None) == []


def test_unknown_verifier_id_returns_empty():
    engine = VerifierEngine()
    results = engine.run_all({}, None, {"name": "t", "verifier_id": "nonexistent"})
    assert results == []


def test_failing_verifier_result_has_passed_false():
    engine = VerifierEngine()
    engine.register("always_fail", failing_verifier)
    results = engine.run_all({}, None, {"name": "t", "verifier_id": "always_fail"})
    assert results[0].passed is False
    assert results[0].checks[0].evidence == "never passes"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/runtime/test_verifier.py -v
```

Expected: `ModuleNotFoundError: No module named 'forge.runtime.verifier'`

- [ ] **Step 3: Implement `forge/runtime/verifier.py`**

```python
from __future__ import annotations
from typing import Callable
from forge.runtime.verification import VerificationResult


class VerifierEngine:
    def __init__(self) -> None:
        self._verifiers: dict[str, Callable] = {}

    def register(self, verifier_id: str, fn: Callable) -> None:
        self._verifiers[verifier_id] = fn

    def run_all(
        self, state: dict, trajectory, task: dict | None
    ) -> list[VerificationResult]:
        if task is None:
            return []
        verifier_id = task.get("verifier_id")
        if not verifier_id or verifier_id not in self._verifiers:
            return []
        return [self._verifiers[verifier_id](state, trajectory, task)]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/runtime/test_verifier.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add forge/runtime/verifier.py tests/runtime/test_verifier.py
git commit -m "feat: add VerifierEngine with verifier_id dispatch"
```

---

## Task 9: TrajectoryStore and Trajectory

**Files:**
- Create: `forge/runtime/trajectory.py`
- Create: `tests/runtime/test_trajectory.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/runtime/test_trajectory.py
import json
from forge.runtime.snapshot import StepSnapshot
from forge.runtime.trajectory import Trajectory, TrajectoryStore


def make_snapshot(episode_id: str, step_index: int, action_type: str) -> StepSnapshot:
    return StepSnapshot(
        episode_id=episode_id,
        step_index=step_index,
        state_hash_before="sha256:aaa",
        state_hash_after="sha256:bbb",
        action={"type": action_type},
        events=[{"type": "test_event", "entity_id": "x"}],
        reward=0.5,
        verifier_results=[],
        diff={"added": {}, "changed": {}, "removed": {}},
        terminated=False,
        truncated=False,
    )


def test_record_and_retrieve_steps():
    store = TrajectoryStore("ep_0000")
    store.record(make_snapshot("ep_0000", 0, "action_a"))
    store.record(make_snapshot("ep_0000", 1, "action_b"))
    traj = store.to_trajectory()
    assert len(traj.steps) == 2
    assert traj.steps[0].action["type"] == "action_a"


def test_events_flattens_all_step_events():
    store = TrajectoryStore("ep_0000")
    store.record(make_snapshot("ep_0000", 0, "a"))
    store.record(make_snapshot("ep_0000", 1, "b"))
    traj = store.to_trajectory()
    assert len(traj.events) == 2
    assert all(e["type"] == "test_event" for e in traj.events)


def test_to_jsonl_produces_one_json_object_per_line():
    store = TrajectoryStore("ep_0000")
    store.record(make_snapshot("ep_0000", 0, "a"))
    store.record(make_snapshot("ep_0000", 1, "b"))
    jsonl = store.to_jsonl()
    lines = jsonl.strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert obj["episode_id"] == "ep_0000"


def test_empty_trajectory_has_no_policy_violations():
    store = TrajectoryStore("ep_0000")
    assert store.to_trajectory().has_policy_violation is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/runtime/test_trajectory.py -v
```

Expected: `ModuleNotFoundError: No module named 'forge.runtime.trajectory'`

- [ ] **Step 3: Implement `forge/runtime/trajectory.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from forge.runtime.snapshot import StepSnapshot


@dataclass
class Trajectory:
    episode_id: str
    steps: list[StepSnapshot]

    @property
    def events(self) -> list[dict]:
        return [event for step in self.steps for event in step.events]

    @property
    def has_policy_violation(self) -> bool:
        return any(e.get("type") == "policy_violation" for e in self.events)

    @property
    def step_count(self) -> int:
        return len(self.steps)


class TrajectoryStore:
    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self._steps: list[StepSnapshot] = []

    def record(self, snapshot: StepSnapshot) -> None:
        self._steps.append(snapshot)

    def to_trajectory(self) -> Trajectory:
        return Trajectory(episode_id=self.episode_id, steps=list(self._steps))

    def to_jsonl(self) -> str:
        return "\n".join(step.model_dump_json() for step in self._steps)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/runtime/test_trajectory.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add forge/runtime/trajectory.py tests/runtime/test_trajectory.py
git commit -m "feat: add TrajectoryStore and Trajectory with JSONL export"
```

---

## Task 10: ForgeEnv

**Files:**
- Create: `forge/runtime/env.py`
- Create: `tests/runtime/test_env.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/runtime/test_env.py
import copy
import pytest
from forge.runtime.context import RuntimeContext
from forge.runtime.env import ForgeEnv
from forge.runtime.snapshot import EnvironmentSpec
from forge.runtime.transition import TransitionEngine, TransitionResult
from forge.runtime.verifier import VerifierEngine
from forge.runtime.reward import RewardEngine
from forge.runtime.verification import CheckResult, VerificationResult


class FixedStateFactory:
    def create(self, ctx: RuntimeContext, options: dict) -> dict:
        ctx.actor_id = "u_0000"
        return {"counter": {"c_0": {"id": "c_0", "value": 0}}}


def increment_transition(state, action, ctx):
    new_state = copy.deepcopy(state)
    new_state["counter"]["c_0"]["value"] += 1
    return TransitionResult(state=new_state, events=[{"type": "incremented", "entity_id": "c_0"}])


def check_counter_verifier(state, trajectory, task):
    passed = state["counter"]["c_0"]["value"] >= task["inputs"]["target"]
    return VerificationResult.from_checks(
        "check_counter",
        [CheckResult(name="counter_reached", passed=passed, score=1.0 if passed else 0.0)],
    )


def build_env(max_steps: int = 10) -> ForgeEnv:
    spec = EnvironmentSpec(name="test_env", domain="test", max_steps=max_steps)
    te = TransitionEngine()
    te.register("increment", increment_transition)
    ve = VerifierEngine()
    ve.register("check_counter", check_counter_verifier)
    re = RewardEngine()
    return ForgeEnv(
        env_spec=spec,
        initial_state_factory=FixedStateFactory(),
        transition_engine=te,
        verifier_engine=ve,
        reward_engine=re,
    )


def test_reset_returns_observation_and_info():
    env = build_env()
    obs, info = env.reset(seed=1)
    assert isinstance(obs, dict)
    assert "episode_id" in info
    assert "seed" in info


def test_step_returns_five_tuple():
    env = build_env()
    env.reset(seed=1)
    task = {"name": "reach_3", "verifier_id": "check_counter", "inputs": {"target": 3}}
    result = env.step({"type": "increment", "task": task})
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, dict)
    assert isinstance(reward, float)


def test_task_completion_terminates_episode():
    env = build_env()
    task = {"name": "reach_1", "verifier_id": "check_counter", "inputs": {"target": 1}}
    env.reset(seed=1, options={"task": task})
    _, _, terminated, _, _ = env.step({"type": "increment"})
    assert terminated is True


def test_invalid_action_does_not_mutate_state():
    env = build_env()
    env.reset(seed=1)
    obs_before, _ = env.reset(seed=1)
    obs_after, reward, _, _, info = env.step({"type": "nonexistent_action"})
    assert obs_after == obs_before
    assert reward == 0.0
    assert "error" in info


def test_step_before_reset_raises():
    env = build_env()
    with pytest.raises(RuntimeError, match="reset()"):
        env.step({"type": "increment"})


def test_truncation_at_max_steps():
    env = build_env(max_steps=2)
    task = {"name": "impossible", "verifier_id": "check_counter", "inputs": {"target": 999}}
    env.reset(seed=1, options={"task": task})
    env.step({"type": "increment"})
    _, _, _, truncated, _ = env.step({"type": "increment"})
    assert truncated is True


def test_same_seed_produces_same_initial_observation():
    env = build_env()
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert obs1 == obs2
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/runtime/test_env.py -v
```

Expected: `ModuleNotFoundError: No module named 'forge.runtime.env'`

- [ ] **Step 3: Implement `forge/runtime/env.py`**

```python
from __future__ import annotations
from typing import Protocol
import gymnasium as gym
from forge.runtime.action import ActionValidator
from forge.runtime.context import RuntimeContext
from forge.runtime.diff import compute_diff
from forge.runtime.reward import RewardEngine
from forge.runtime.snapshot import EnvironmentSpec, InvalidActionError, StepSnapshot
from forge.runtime.state import StateStore
from forge.runtime.trajectory import TrajectoryStore
from forge.runtime.transition import TransitionEngine
from forge.runtime.verifier import VerifierEngine


class InitialStateFactory(Protocol):
    def create(self, ctx: RuntimeContext, options: dict) -> dict: ...


class ForgeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_spec: EnvironmentSpec,
        initial_state_factory: InitialStateFactory,
        transition_engine: TransitionEngine,
        verifier_engine: VerifierEngine,
        reward_engine: RewardEngine,
    ) -> None:
        super().__init__()
        self.env_spec = env_spec
        self._factory = initial_state_factory
        self._transition_engine = transition_engine
        self._verifier_engine = verifier_engine
        self._reward_engine = reward_engine
        self._action_validator = ActionValidator(transition_engine.action_types)

        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})

        self._ctx: RuntimeContext | None = None
        self._state_store: StateStore | None = None
        self._traj_store: TrajectoryStore | None = None
        self._current_task: dict | None = None
        self._step_count: int = 0
        self._episode_id: str | None = None

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        actual_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        opts = options or {}

        self._ctx = RuntimeContext(seed=actual_seed)
        self._episode_id = self._ctx.id_generator.next("ep")
        initial_state = self._factory.create(self._ctx, opts)
        self._state_store = StateStore(initial_state)
        self._traj_store = TrajectoryStore(self._episode_id)
        self._current_task = opts.get("task", self.env_spec.default_task)
        self._step_count = 0

        return self._state_store.get(), {
            "episode_id": self._episode_id,
            "task": self._current_task,
            "seed": actual_seed,
        }

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        if self._ctx is None:
            raise RuntimeError("Must call reset() before step()")

        state_before = self._state_store.get()
        hash_before = self._state_store.hash()

        validation_error = self._action_validator.validate(action)
        if validation_error:
            self._record_invalid_step(hash_before, action)
            return state_before, 0.0, False, False, {"error": validation_error}

        try:
            result = self._transition_engine.apply(state_before, action, self._ctx)
        except InvalidActionError as exc:
            self._record_invalid_step(hash_before, action)
            return state_before, 0.0, False, False, {"error": exc.to_dict()}

        self._state_store.apply(result.state)
        state_after = self._state_store.get()
        hash_after = self._state_store.hash()
        self._ctx.clock.advance()

        diff = compute_diff(state_before, state_after)
        trajectory = self._traj_store.to_trajectory()
        verifier_results = self._verifier_engine.run_all(
            state_after, trajectory, self._current_task
        )
        reward_breakdown = self._reward_engine.compute(
            state_after, trajectory, verifier_results, self._current_task
        )

        self._step_count += 1
        terminated = any(vr.passed for vr in verifier_results)
        truncated = self._step_count >= self.env_spec.max_steps

        snapshot = StepSnapshot(
            episode_id=self._episode_id,
            step_index=self._step_count - 1,
            state_hash_before=hash_before,
            state_hash_after=hash_after,
            action=action,
            events=result.events,
            reward=reward_breakdown.total_reward,
            verifier_results=[vr.model_dump() for vr in verifier_results],
            diff=diff,
            terminated=terminated,
            truncated=truncated,
        )
        self._traj_store.record(snapshot)

        return state_after, reward_breakdown.total_reward, terminated, truncated, {
            "episode_id": self._episode_id,
            "verifier_results": [vr.model_dump() for vr in verifier_results],
            "reward_breakdown": reward_breakdown.model_dump(),
            "events": result.events,
        }

    def _record_invalid_step(self, hash_before: str, action: dict) -> None:
        self._step_count += 1
        self._traj_store.record(
            StepSnapshot(
                episode_id=self._episode_id,
                step_index=self._step_count - 1,
                state_hash_before=hash_before,
                state_hash_after=hash_before,
                action=action,
                events=[],
                reward=0.0,
                verifier_results=[],
                diff={"added": {}, "changed": {}, "removed": {}},
                terminated=False,
                truncated=False,
            )
        )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/runtime/test_env.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Run all runtime tests**

```bash
pytest tests/runtime/ -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add forge/runtime/env.py tests/runtime/test_env.py
git commit -m "feat: add ForgeEnv — Gymnasium-compatible env wrapping all runtime components"
```

---

## Task 11: Gmail State and Action Models

**Files:**
- Create: `examples/gmail_env/state_models.py`
- Create: `examples/gmail_env/action_models.py`

- [ ] **Step 1: Write `examples/gmail_env/state_models.py`**

These are plain dicts at runtime (the compiler generates Pydantic models in M2). For M1, this file documents the shape as typed dicts and provides factory helpers.

```python
from typing import TypedDict


class User(TypedDict):
    id: str
    email: str
    role: str


class Email(TypedDict):
    id: str
    from_: str
    to: str
    subject: str
    body: str
    labels: list[str]
    archived: bool
    thread_id: str
    read: bool
    created_at: str
    escalated: bool


class Thread(TypedDict):
    id: str
    email_ids: list[str]
    escalated: bool


class Label(TypedDict):
    id: str
    name: str


class GmailState(TypedDict):
    users: dict[str, User]
    emails: dict[str, Email]
    threads: dict[str, Thread]
    labels: dict[str, Label]
    actor_id: str
```

- [ ] **Step 2: Write `examples/gmail_env/action_models.py`**

```python
from typing import TypedDict


class ReplyEmailAction(TypedDict):
    type: str          # "reply_email"
    thread_id: str
    body: str


class SendEmailAction(TypedDict):
    type: str          # "send_email"
    to: str
    subject: str
    body: str


class ArchiveEmailAction(TypedDict):
    type: str          # "archive_email"
    email_id: str


class ApplyLabelAction(TypedDict):
    type: str          # "apply_label"
    email_id: str
    label: str


class MarkReadAction(TypedDict):
    type: str          # "mark_read"
    email_id: str


class EscalateThreadAction(TypedDict):
    type: str          # "escalate_thread"
    thread_id: str
```

- [ ] **Step 3: Verify imports**

```bash
python -c "from examples.gmail_env.state_models import GmailState; from examples.gmail_env.action_models import ReplyEmailAction; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add examples/gmail_env/state_models.py examples/gmail_env/action_models.py
git commit -m "feat: add Gmail state and action model type definitions"
```

---

## Task 12: Gmail Initial State Factory

**Files:**
- Create: `examples/gmail_env/initial_state.py`

- [ ] **Step 1: Implement `examples/gmail_env/initial_state.py`**

```python
from forge.runtime.context import RuntimeContext


class GmailInitialStateFactory:
    def create(self, ctx: RuntimeContext, options: dict) -> dict:
        user_id = ctx.id_generator.next("user")
        ctx.actor_id = user_id

        thread_id = ctx.id_generator.next("thread")
        email_id = ctx.id_generator.next("email")
        now = ctx.clock.now().isoformat()

        scenario = options.get("scenario", "refund_request")

        if scenario == "newsletter":
            subject = "Your weekly digest"
            body = "Here are this week's top stories."
            sender = "newsletter@digest.com"
            labels = ["inbox"]
        elif scenario == "billing_complaint":
            subject = "Billing issue - urgent"
            body = "I have been charged incorrectly. This is urgent."
            sender = "customer@example.com"
            labels = ["inbox"]
        else:
            subject = "Refund request"
            body = "I was charged twice for my order."
            sender = "customer@example.com"
            labels = ["inbox"]

        return {
            "users": {
                user_id: {
                    "id": user_id,
                    "email": "agent@example.com",
                    "role": "support_agent",
                }
            },
            "emails": {
                email_id: {
                    "id": email_id,
                    "from_": sender,
                    "to": "support@example.com",
                    "subject": subject,
                    "body": body,
                    "labels": labels,
                    "archived": False,
                    "thread_id": thread_id,
                    "read": False,
                    "created_at": now,
                    "escalated": False,
                }
            },
            "threads": {
                thread_id: {
                    "id": thread_id,
                    "email_ids": [email_id],
                    "escalated": False,
                }
            },
            "labels": {
                "inbox": {"id": "inbox", "name": "Inbox"},
                "sent": {"id": "sent", "name": "Sent"},
                "urgent": {"id": "urgent", "name": "Urgent"},
                "newsletter": {"id": "newsletter", "name": "Newsletter"},
            },
            "actor_id": user_id,
        }
```

- [ ] **Step 2: Commit**

```bash
git add examples/gmail_env/initial_state.py
git commit -m "feat: add GmailInitialStateFactory with scenario support"
```

---

## Task 13: Gmail Transitions (reply_email, send_email, archive_email)

**Files:**
- Create: `examples/gmail_env/transitions/reply_email.py`
- Create: `examples/gmail_env/transitions/send_email.py`
- Create: `examples/gmail_env/transitions/archive_email.py`
- Create: `tests/gmail_env/test_transitions.py` (partial — add more in Task 14)

- [ ] **Step 1: Write the failing tests for the first three transitions**

```python
# tests/gmail_env/test_transitions.py
import pytest
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError
from examples.gmail_env.initial_state import GmailInitialStateFactory
from examples.gmail_env.transitions.reply_email import apply_reply_email
from examples.gmail_env.transitions.send_email import apply_send_email
from examples.gmail_env.transitions.archive_email import apply_archive_email


def make_state(seed: int = 0) -> tuple[dict, RuntimeContext]:
    ctx = RuntimeContext(seed=seed)
    state = GmailInitialStateFactory().create(ctx, {})
    return state, ctx


def get_first_thread_id(state: dict) -> str:
    return next(iter(state["threads"]))


def get_first_email_id(state: dict) -> str:
    return next(iter(state["emails"]))


# --- reply_email ---

def test_reply_email_adds_new_email_to_thread():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_reply_email(state, {"type": "reply_email", "thread_id": thread_id, "body": "Hello"}, ctx)
    thread_email_ids = result.state["threads"][thread_id]["email_ids"]
    assert len(thread_email_ids) == 2


def test_reply_email_new_email_has_sent_label():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_reply_email(state, {"type": "reply_email", "thread_id": thread_id, "body": "Hi"}, ctx)
    thread = result.state["threads"][thread_id]
    new_email_id = thread["email_ids"][-1]
    assert "sent" in result.state["emails"][new_email_id]["labels"]


def test_reply_email_emits_email_replied_event():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_reply_email(state, {"type": "reply_email", "thread_id": thread_id, "body": "Hi"}, ctx)
    assert any(e["type"] == "email_replied" for e in result.events)


def test_reply_email_raises_for_unknown_thread():
    state, ctx = make_state()
    with pytest.raises(InvalidActionError):
        apply_reply_email(state, {"type": "reply_email", "thread_id": "bad_id", "body": "Hi"}, ctx)


def test_reply_email_does_not_mutate_original_state():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    original_count = len(state["threads"][thread_id]["email_ids"])
    apply_reply_email(state, {"type": "reply_email", "thread_id": thread_id, "body": "Hi"}, ctx)
    assert len(state["threads"][thread_id]["email_ids"]) == original_count


# --- send_email ---

def test_send_email_creates_new_thread_and_email():
    state, ctx = make_state()
    result = apply_send_email(
        state,
        {"type": "send_email", "to": "other@example.com", "subject": "Hello", "body": "Hi there"},
        ctx,
    )
    assert len(result.state["threads"]) == 2
    assert len(result.state["emails"]) == 2


def test_send_email_emits_email_sent_event():
    state, ctx = make_state()
    result = apply_send_email(
        state,
        {"type": "send_email", "to": "x@x.com", "subject": "S", "body": "B"},
        ctx,
    )
    assert any(e["type"] == "email_sent" for e in result.events)


# --- archive_email ---

def test_archive_email_sets_archived_true():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_archive_email(state, {"type": "archive_email", "email_id": email_id}, ctx)
    assert result.state["emails"][email_id]["archived"] is True


def test_archive_email_emits_email_archived_event():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_archive_email(state, {"type": "archive_email", "email_id": email_id}, ctx)
    assert any(e["type"] == "email_archived" for e in result.events)


def test_archive_email_raises_for_unknown_email():
    state, ctx = make_state()
    with pytest.raises(InvalidActionError):
        apply_archive_email(state, {"type": "archive_email", "email_id": "bad_id"}, ctx)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/gmail_env/test_transitions.py -v
```

Expected: `ModuleNotFoundError` for transition modules

- [ ] **Step 3: Implement `examples/gmail_env/transitions/reply_email.py`**

```python
import copy
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError
from forge.runtime.transition import TransitionResult


def apply_reply_email(state: dict, action: dict, ctx: RuntimeContext) -> TransitionResult:
    thread_id = action["thread_id"]
    body = action["body"]

    if thread_id not in state["threads"]:
        raise InvalidActionError(f"Thread '{thread_id}' not found", code="ENTITY_NOT_FOUND")

    thread = state["threads"][thread_id]
    original_email_id = thread["email_ids"][0]
    original_email = state["emails"][original_email_id]

    new_email_id = ctx.id_generator.next("email")
    new_email = {
        "id": new_email_id,
        "from_": state["users"][ctx.actor_id]["email"],
        "to": original_email["from_"],
        "subject": f"Re: {original_email['subject']}",
        "body": body,
        "labels": ["sent"],
        "archived": False,
        "thread_id": thread_id,
        "read": True,
        "created_at": ctx.clock.now().isoformat(),
        "escalated": False,
    }

    new_state = copy.deepcopy(state)
    new_state["emails"][new_email_id] = new_email
    new_state["threads"][thread_id]["email_ids"].append(new_email_id)

    return TransitionResult(
        state=new_state,
        events=[{
            "type": "email_replied",
            "entity_id": thread_id,
            "payload": {"email_id": new_email_id},
            "timestamp": ctx.clock.now().isoformat(),
        }],
    )
```

- [ ] **Step 4: Implement `examples/gmail_env/transitions/send_email.py`**

```python
import copy
from forge.runtime.context import RuntimeContext
from forge.runtime.transition import TransitionResult


def apply_send_email(state: dict, action: dict, ctx: RuntimeContext) -> TransitionResult:
    to = action["to"]
    subject = action["subject"]
    body = action["body"]
    now = ctx.clock.now().isoformat()

    thread_id = ctx.id_generator.next("thread")
    email_id = ctx.id_generator.next("email")

    new_email = {
        "id": email_id,
        "from_": state["users"][ctx.actor_id]["email"],
        "to": to,
        "subject": subject,
        "body": body,
        "labels": ["sent"],
        "archived": False,
        "thread_id": thread_id,
        "read": True,
        "created_at": now,
        "escalated": False,
    }
    new_thread = {
        "id": thread_id,
        "email_ids": [email_id],
        "escalated": False,
    }

    new_state = copy.deepcopy(state)
    new_state["emails"][email_id] = new_email
    new_state["threads"][thread_id] = new_thread

    return TransitionResult(
        state=new_state,
        events=[{
            "type": "email_sent",
            "entity_id": thread_id,
            "payload": {"email_id": email_id, "to": to},
            "timestamp": now,
        }],
    )
```

- [ ] **Step 5: Implement `examples/gmail_env/transitions/archive_email.py`**

```python
import copy
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError
from forge.runtime.transition import TransitionResult


def apply_archive_email(state: dict, action: dict, ctx: RuntimeContext) -> TransitionResult:
    email_id = action["email_id"]

    if email_id not in state["emails"]:
        raise InvalidActionError(f"Email '{email_id}' not found", code="ENTITY_NOT_FOUND")

    new_state = copy.deepcopy(state)
    new_state["emails"][email_id]["archived"] = True

    return TransitionResult(
        state=new_state,
        events=[{
            "type": "email_archived",
            "entity_id": email_id,
            "payload": {},
            "timestamp": ctx.clock.now().isoformat(),
        }],
    )
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
pytest tests/gmail_env/test_transitions.py -v
```

Expected: all 12 pass

- [ ] **Step 7: Commit**

```bash
git add examples/gmail_env/transitions/reply_email.py examples/gmail_env/transitions/send_email.py examples/gmail_env/transitions/archive_email.py tests/gmail_env/test_transitions.py
git commit -m "feat: add Gmail reply_email, send_email, archive_email transitions"
```

---

## Task 14: Gmail Transitions (apply_label, mark_read, escalate_thread)

**Files:**
- Create: `examples/gmail_env/transitions/apply_label.py`
- Create: `examples/gmail_env/transitions/mark_read.py`
- Create: `examples/gmail_env/transitions/escalate_thread.py`
- Modify: `tests/gmail_env/test_transitions.py` (append tests)

- [ ] **Step 1: Append the failing tests to `tests/gmail_env/test_transitions.py`**

Add these at the end of the file:

```python
from examples.gmail_env.transitions.apply_label import apply_apply_label
from examples.gmail_env.transitions.mark_read import apply_mark_read
from examples.gmail_env.transitions.escalate_thread import apply_escalate_thread


# --- apply_label ---

def test_apply_label_adds_label_to_email():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_apply_label(state, {"type": "apply_label", "email_id": email_id, "label": "urgent"}, ctx)
    assert "urgent" in result.state["emails"][email_id]["labels"]


def test_apply_label_is_idempotent():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result1 = apply_apply_label(state, {"type": "apply_label", "email_id": email_id, "label": "urgent"}, ctx)
    result2 = apply_apply_label(result1.state, {"type": "apply_label", "email_id": email_id, "label": "urgent"}, ctx)
    assert result2.state["emails"][email_id]["labels"].count("urgent") == 1


def test_apply_label_emits_label_applied_event():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_apply_label(state, {"type": "apply_label", "email_id": email_id, "label": "urgent"}, ctx)
    assert any(e["type"] == "label_applied" for e in result.events)


def test_apply_label_raises_for_unknown_email():
    state, ctx = make_state()
    with pytest.raises(InvalidActionError):
        apply_apply_label(state, {"type": "apply_label", "email_id": "bad", "label": "x"}, ctx)


# --- mark_read ---

def test_mark_read_sets_read_true():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    assert state["emails"][email_id]["read"] is False
    result = apply_mark_read(state, {"type": "mark_read", "email_id": email_id}, ctx)
    assert result.state["emails"][email_id]["read"] is True


def test_mark_read_emits_email_read_event():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_mark_read(state, {"type": "mark_read", "email_id": email_id}, ctx)
    assert any(e["type"] == "email_read" for e in result.events)


# --- escalate_thread ---

def test_escalate_thread_sets_escalated_true_on_thread():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_escalate_thread(state, {"type": "escalate_thread", "thread_id": thread_id}, ctx)
    assert result.state["threads"][thread_id]["escalated"] is True


def test_escalate_thread_emits_thread_escalated_event():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_escalate_thread(state, {"type": "escalate_thread", "thread_id": thread_id}, ctx)
    assert any(e["type"] == "thread_escalated" for e in result.events)


def test_escalate_thread_raises_for_unknown_thread():
    state, ctx = make_state()
    with pytest.raises(InvalidActionError):
        apply_escalate_thread(state, {"type": "escalate_thread", "thread_id": "bad"}, ctx)
```

- [ ] **Step 2: Run tests to confirm the new ones fail**

```bash
pytest tests/gmail_env/test_transitions.py -v
```

Expected: 12 pass, 9 fail with `ModuleNotFoundError`

- [ ] **Step 3: Implement `examples/gmail_env/transitions/apply_label.py`**

```python
import copy
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError
from forge.runtime.transition import TransitionResult


def apply_apply_label(state: dict, action: dict, ctx: RuntimeContext) -> TransitionResult:
    email_id = action["email_id"]
    label = action["label"]

    if email_id not in state["emails"]:
        raise InvalidActionError(f"Email '{email_id}' not found", code="ENTITY_NOT_FOUND")

    new_state = copy.deepcopy(state)
    labels = new_state["emails"][email_id]["labels"]
    if label not in labels:
        labels.append(label)

    return TransitionResult(
        state=new_state,
        events=[{
            "type": "label_applied",
            "entity_id": email_id,
            "payload": {"label": label},
            "timestamp": ctx.clock.now().isoformat(),
        }],
    )
```

- [ ] **Step 4: Implement `examples/gmail_env/transitions/mark_read.py`**

```python
import copy
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError
from forge.runtime.transition import TransitionResult


def apply_mark_read(state: dict, action: dict, ctx: RuntimeContext) -> TransitionResult:
    email_id = action["email_id"]

    if email_id not in state["emails"]:
        raise InvalidActionError(f"Email '{email_id}' not found", code="ENTITY_NOT_FOUND")

    new_state = copy.deepcopy(state)
    new_state["emails"][email_id]["read"] = True

    return TransitionResult(
        state=new_state,
        events=[{
            "type": "email_read",
            "entity_id": email_id,
            "payload": {},
            "timestamp": ctx.clock.now().isoformat(),
        }],
    )
```

- [ ] **Step 5: Implement `examples/gmail_env/transitions/escalate_thread.py`**

```python
import copy
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError
from forge.runtime.transition import TransitionResult


def apply_escalate_thread(state: dict, action: dict, ctx: RuntimeContext) -> TransitionResult:
    thread_id = action["thread_id"]

    if thread_id not in state["threads"]:
        raise InvalidActionError(f"Thread '{thread_id}' not found", code="ENTITY_NOT_FOUND")

    new_state = copy.deepcopy(state)
    new_state["threads"][thread_id]["escalated"] = True
    for email_id in new_state["threads"][thread_id]["email_ids"]:
        new_state["emails"][email_id]["escalated"] = True

    return TransitionResult(
        state=new_state,
        events=[{
            "type": "thread_escalated",
            "entity_id": thread_id,
            "payload": {},
            "timestamp": ctx.clock.now().isoformat(),
        }],
    )
```

- [ ] **Step 6: Run all transition tests to confirm they pass**

```bash
pytest tests/gmail_env/test_transitions.py -v
```

Expected: `21 passed`

- [ ] **Step 7: Commit**

```bash
git add examples/gmail_env/transitions/apply_label.py examples/gmail_env/transitions/mark_read.py examples/gmail_env/transitions/escalate_thread.py tests/gmail_env/test_transitions.py
git commit -m "feat: add Gmail apply_label, mark_read, escalate_thread transitions"
```

---

## Task 15: Gmail Verifiers

**Files:**
- Create: `examples/gmail_env/verifiers/reply_to_customer.py`
- Create: `examples/gmail_env/verifiers/label_urgent_request.py`
- Create: `examples/gmail_env/verifiers/archive_newsletter.py`
- Create: `examples/gmail_env/verifiers/escalate_billing_complaint.py`
- Create: `tests/gmail_env/test_verifiers.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/gmail_env/test_verifiers.py
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import StepSnapshot
from forge.runtime.trajectory import Trajectory
from examples.gmail_env.initial_state import GmailInitialStateFactory
from examples.gmail_env.transitions.reply_email import apply_reply_email
from examples.gmail_env.transitions.apply_label import apply_apply_label
from examples.gmail_env.transitions.archive_email import apply_archive_email
from examples.gmail_env.transitions.escalate_thread import apply_escalate_thread
from examples.gmail_env.verifiers.reply_to_customer import verify_reply_to_customer
from examples.gmail_env.verifiers.label_urgent_request import verify_label_urgent_request
from examples.gmail_env.verifiers.archive_newsletter import verify_archive_newsletter
from examples.gmail_env.verifiers.escalate_billing_complaint import verify_escalate_billing_complaint


def make_state(seed: int = 0):
    ctx = RuntimeContext(seed=seed)
    state = GmailInitialStateFactory().create(ctx, {})
    return state, ctx


def empty_trajectory() -> Trajectory:
    return Trajectory(episode_id="ep_0", steps=[])


def trajectory_with_events(events: list[dict]) -> Trajectory:
    step = StepSnapshot(
        episode_id="ep_0", step_index=0,
        state_hash_before="sha256:a", state_hash_after="sha256:b",
        action={"type": "noop"}, events=events, reward=0.0,
        verifier_results=[], diff={"added": {}, "changed": {}, "removed": {}},
        terminated=False, truncated=False,
    )
    return Trajectory(episode_id="ep_0", steps=[step])


def get_first_thread_id(state):
    return next(iter(state["threads"]))


def get_first_email_id(state):
    return next(iter(state["emails"]))


# --- reply_to_customer ---

def test_reply_to_customer_passes_when_reply_sent():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_reply_email(state, {"type": "reply_email", "thread_id": thread_id, "body": "Hi"}, ctx)
    traj = trajectory_with_events(result.events)
    task = {"name": "reply_to_customer", "verifier_id": "reply_to_customer", "inputs": {"thread_id": thread_id}}
    vr = verify_reply_to_customer(result.state, traj, task)
    assert vr.passed is True


def test_reply_to_customer_fails_with_no_reply():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    task = {"name": "reply_to_customer", "verifier_id": "reply_to_customer", "inputs": {"thread_id": thread_id}}
    vr = verify_reply_to_customer(state, empty_trajectory(), task)
    assert vr.passed is False
    assert vr.checks[0].evidence is not None


# --- label_urgent_request ---

def test_label_urgent_passes_when_urgent_label_applied():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_apply_label(state, {"type": "apply_label", "email_id": email_id, "label": "urgent"}, ctx)
    task = {"name": "label_urgent_request", "verifier_id": "label_urgent_request", "inputs": {"email_id": email_id}}
    vr = verify_label_urgent_request(result.state, empty_trajectory(), task)
    assert vr.passed is True


def test_label_urgent_fails_without_label():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    task = {"name": "label_urgent_request", "verifier_id": "label_urgent_request", "inputs": {"email_id": email_id}}
    vr = verify_label_urgent_request(state, empty_trajectory(), task)
    assert vr.passed is False


# --- archive_newsletter ---

def test_archive_newsletter_passes_when_archived():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    result = apply_archive_email(state, {"type": "archive_email", "email_id": email_id}, ctx)
    task = {"name": "archive_newsletter", "verifier_id": "archive_newsletter", "inputs": {"email_id": email_id}}
    vr = verify_archive_newsletter(result.state, empty_trajectory(), task)
    assert vr.passed is True


def test_archive_newsletter_fails_when_not_archived():
    state, ctx = make_state()
    email_id = get_first_email_id(state)
    task = {"name": "archive_newsletter", "verifier_id": "archive_newsletter", "inputs": {"email_id": email_id}}
    vr = verify_archive_newsletter(state, empty_trajectory(), task)
    assert vr.passed is False


# --- escalate_billing_complaint ---

def test_escalate_billing_passes_when_thread_escalated():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    result = apply_escalate_thread(state, {"type": "escalate_thread", "thread_id": thread_id}, ctx)
    task = {"name": "escalate_billing_complaint", "verifier_id": "escalate_billing_complaint", "inputs": {"thread_id": thread_id}}
    vr = verify_escalate_billing_complaint(result.state, empty_trajectory(), task)
    assert vr.passed is True


def test_escalate_billing_fails_when_not_escalated():
    state, ctx = make_state()
    thread_id = get_first_thread_id(state)
    task = {"name": "escalate_billing_complaint", "verifier_id": "escalate_billing_complaint", "inputs": {"thread_id": thread_id}}
    vr = verify_escalate_billing_complaint(state, empty_trajectory(), task)
    assert vr.passed is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/gmail_env/test_verifiers.py -v
```

Expected: `ModuleNotFoundError` for verifier modules

- [ ] **Step 3: Implement `examples/gmail_env/verifiers/reply_to_customer.py`**

```python
from forge.runtime.verification import CheckResult, VerificationResult


def verify_reply_to_customer(state: dict, trajectory, task: dict) -> VerificationResult:
    thread_id = task["inputs"]["thread_id"]

    replies = [
        e for e in trajectory.events
        if e["type"] == "email_replied" and e["entity_id"] == thread_id
    ]

    passed = len(replies) > 0
    return VerificationResult.from_checks(
        "reply_to_customer",
        [CheckResult(
            name="reply_sent",
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence=None if passed else f"No reply sent to thread '{thread_id}'.",
        )],
    )
```

- [ ] **Step 4: Implement `examples/gmail_env/verifiers/label_urgent_request.py`**

```python
from forge.runtime.verification import CheckResult, VerificationResult


def verify_label_urgent_request(state: dict, trajectory, task: dict) -> VerificationResult:
    email_id = task["inputs"]["email_id"]
    email = state["emails"].get(email_id, {})
    passed = "urgent" in email.get("labels", [])

    return VerificationResult.from_checks(
        "label_urgent_request",
        [CheckResult(
            name="urgent_label_applied",
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence=None if passed else f"Email '{email_id}' does not have 'urgent' label.",
        )],
    )
```

- [ ] **Step 5: Implement `examples/gmail_env/verifiers/archive_newsletter.py`**

```python
from forge.runtime.verification import CheckResult, VerificationResult


def verify_archive_newsletter(state: dict, trajectory, task: dict) -> VerificationResult:
    email_id = task["inputs"]["email_id"]
    email = state["emails"].get(email_id, {})
    passed = email.get("archived", False) is True

    return VerificationResult.from_checks(
        "archive_newsletter",
        [CheckResult(
            name="email_archived",
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence=None if passed else f"Email '{email_id}' has not been archived.",
        )],
    )
```

- [ ] **Step 6: Implement `examples/gmail_env/verifiers/escalate_billing_complaint.py`**

```python
from forge.runtime.verification import CheckResult, VerificationResult


def verify_escalate_billing_complaint(state: dict, trajectory, task: dict) -> VerificationResult:
    thread_id = task["inputs"]["thread_id"]
    thread = state["threads"].get(thread_id, {})
    passed = thread.get("escalated", False) is True

    return VerificationResult.from_checks(
        "escalate_billing_complaint",
        [CheckResult(
            name="thread_escalated",
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence=None if passed else f"Thread '{thread_id}' has not been escalated.",
        )],
    )
```

- [ ] **Step 7: Run verifier tests to confirm they pass**

```bash
pytest tests/gmail_env/test_verifiers.py -v
```

Expected: `10 passed`

- [ ] **Step 8: Commit**

```bash
git add examples/gmail_env/verifiers/ tests/gmail_env/test_verifiers.py
git commit -m "feat: add Gmail verifiers for all four tasks"
```

---

## Task 16: Gmail Reward Function and Gym Wrapper

**Files:**
- Create: `examples/gmail_env/rewards/base.py`
- Create: `examples/gmail_env/gym_wrapper.py`

- [ ] **Step 1: Implement `examples/gmail_env/rewards/base.py`**

```python
from forge.runtime.reward import RewardBreakdown, RewardComponent


def compute_gmail_reward(
    state: dict,
    trajectory,
    verifier_results: list,
    task: dict | None = None,
) -> RewardBreakdown:
    pass_rate = (
        sum(vr.score for vr in verifier_results) / len(verifier_results)
        if verifier_results
        else 0.0
    )
    step_count = trajectory.step_count
    has_violation = trajectory.has_policy_violation

    task_success = pass_rate * 1.0
    step_penalty = 0.01 * step_count
    violation_penalty = 1.0 if has_violation else 0.0

    total = task_success - step_penalty - violation_penalty
    total = max(-1.0, min(1.0, total))

    return RewardBreakdown(
        total_reward=total,
        components=[
            RewardComponent(name="task_success", value=task_success),
            RewardComponent(name="step_efficiency", value=-step_penalty),
            RewardComponent(name="policy_compliance", value=-violation_penalty),
        ],
    )
```

- [ ] **Step 2: Implement `examples/gmail_env/gym_wrapper.py`**

```python
from forge.runtime.env import ForgeEnv
from forge.runtime.reward import RewardEngine
from forge.runtime.snapshot import EnvironmentSpec
from forge.runtime.transition import TransitionEngine
from forge.runtime.verifier import VerifierEngine
from examples.gmail_env.initial_state import GmailInitialStateFactory
from examples.gmail_env.rewards.base import compute_gmail_reward
from examples.gmail_env.transitions.apply_label import apply_apply_label
from examples.gmail_env.transitions.archive_email import apply_archive_email
from examples.gmail_env.transitions.escalate_thread import apply_escalate_thread
from examples.gmail_env.transitions.mark_read import apply_mark_read
from examples.gmail_env.transitions.reply_email import apply_reply_email
from examples.gmail_env.transitions.send_email import apply_send_email
from examples.gmail_env.verifiers.archive_newsletter import verify_archive_newsletter
from examples.gmail_env.verifiers.escalate_billing_complaint import verify_escalate_billing_complaint
from examples.gmail_env.verifiers.label_urgent_request import verify_label_urgent_request
from examples.gmail_env.verifiers.reply_to_customer import verify_reply_to_customer


def build_gmail_env(max_steps: int = 20) -> ForgeEnv:
    spec = EnvironmentSpec(name="gmail_env", domain="email", max_steps=max_steps)

    te = TransitionEngine()
    te.register("reply_email", apply_reply_email)
    te.register("send_email", apply_send_email)
    te.register("archive_email", apply_archive_email)
    te.register("apply_label", apply_apply_label)
    te.register("mark_read", apply_mark_read)
    te.register("escalate_thread", apply_escalate_thread)

    ve = VerifierEngine()
    ve.register("reply_to_customer", verify_reply_to_customer)
    ve.register("label_urgent_request", verify_label_urgent_request)
    ve.register("archive_newsletter", verify_archive_newsletter)
    ve.register("escalate_billing_complaint", verify_escalate_billing_complaint)

    re = RewardEngine()
    re.set_default(compute_gmail_reward)

    return ForgeEnv(
        env_spec=spec,
        initial_state_factory=GmailInitialStateFactory(),
        transition_engine=te,
        verifier_engine=ve,
        reward_engine=re,
    )
```

- [ ] **Step 3: Verify the wrapper builds and imports correctly**

```bash
python -c "from examples.gmail_env.gym_wrapper import build_gmail_env; env = build_gmail_env(); print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add examples/gmail_env/rewards/base.py examples/gmail_env/gym_wrapper.py
git commit -m "feat: add Gmail reward function and gym_wrapper factory"
```

---

## Task 17: End-to-End Determinism Test

**Files:**
- Create: `tests/gmail_env/test_determinism.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/gmail_env/test_determinism.py
import json
from examples.gmail_env.gym_wrapper import build_gmail_env


def run_episode(seed: int, actions: list[dict], task: dict) -> dict:
    env = build_gmail_env()
    obs, info = env.reset(seed=seed, options={"task": task})
    episode_id = info["episode_id"]

    step_hashes = []
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        snapshots = env._traj_store._steps
        step_hashes.append(snapshots[-1].state_hash_after)
        if terminated or truncated:
            break

    trajectory_jsonl = env._traj_store.to_jsonl()
    return {
        "episode_id": episode_id,
        "step_hashes": step_hashes,
        "trajectory_jsonl": trajectory_jsonl,
        "final_obs": obs,
    }


def test_same_seed_and_actions_produce_identical_trajectory():
    task = {
        "name": "reply_to_customer",
        "verifier_id": "reply_to_customer",
        "inputs": {},  # thread_id filled in after reset
    }

    env1 = build_gmail_env()
    obs1, info1 = env1.reset(seed=42, options={"task": task})
    thread_id = next(iter(obs1["threads"]))
    task_with_id = {**task, "inputs": {"thread_id": thread_id}}

    env1._current_task = task_with_id
    obs1_a, r1, t1, tr1, i1 = env1.step({"type": "reply_email", "thread_id": thread_id, "body": "Hello"})
    hash1 = env1._traj_store._steps[-1].state_hash_after

    env2 = build_gmail_env()
    obs2, info2 = env2.reset(seed=42, options={"task": task})
    env2._current_task = task_with_id
    obs2_a, r2, t2, tr2, i2 = env2.step({"type": "reply_email", "thread_id": thread_id, "body": "Hello"})
    hash2 = env2._traj_store._steps[-1].state_hash_after

    assert hash1 == hash2
    assert r1 == r2
    assert t1 == t2


def test_different_seeds_produce_different_episode_ids():
    env = build_gmail_env()
    _, info1 = env.reset(seed=1)
    _, info2 = env.reset(seed=2)
    assert info1["episode_id"] != info2["episode_id"]


def test_trajectory_exports_valid_jsonl():
    env = build_gmail_env()
    task = {"name": "archive_newsletter", "verifier_id": "archive_newsletter", "inputs": {}}
    obs, info = env.reset(seed=7, options={"task": task})
    email_id = next(iter(obs["emails"]))
    env._current_task = {**task, "inputs": {"email_id": email_id}}
    env.step({"type": "archive_email", "email_id": email_id})

    jsonl = env._traj_store.to_jsonl()
    lines = jsonl.strip().split("\n")
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["episode_id"] == info["episode_id"]
    assert "state_hash_before" in record
    assert "state_hash_after" in record
    assert "diff" in record
    assert "reward" in record


def test_invalid_action_does_not_corrupt_trajectory_hashes():
    env = build_gmail_env()
    obs, info = env.reset(seed=5)
    email_id = next(iter(obs["emails"]))

    env.step({"type": "mark_read", "email_id": email_id})
    hash_after_valid = env._traj_store._steps[-1].state_hash_after

    env.step({"type": "nonexistent_action"})
    hash_after_invalid = env._traj_store._steps[-1].state_hash_after

    assert hash_after_valid == hash_after_invalid


def test_episode_task_completion_produces_positive_reward():
    env = build_gmail_env()
    task = {"name": "reply_to_customer", "verifier_id": "reply_to_customer", "inputs": {}}
    obs, info = env.reset(seed=10, options={"task": task})
    thread_id = next(iter(obs["threads"]))
    env._current_task = {**task, "inputs": {"thread_id": thread_id}}

    _, reward, terminated, _, _ = env.step(
        {"type": "reply_email", "thread_id": thread_id, "body": "Thank you for reaching out."}
    )
    assert terminated is True
    assert reward > 0.0


def test_five_seeds_all_produce_distinct_state_hashes():
    env = build_gmail_env()
    hashes = set()
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        from forge.runtime.state import StateStore
        store = StateStore(obs)
        hashes.add(store.hash())
    assert len(hashes) == 5
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/gmail_env/test_determinism.py -v
```

Expected: failures (module import errors or logic failures)

- [ ] **Step 3: Run tests and fix until all pass**

```bash
pytest tests/gmail_env/test_determinism.py -v
```

Expected: `6 passed`

- [ ] **Step 4: Run the full test suite**

```bash
pytest -v
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/gmail_env/test_determinism.py
git commit -m "test: add end-to-end determinism tests for Gmail env — M1 acceptance criteria"
```

---

## Task 18: M1 Acceptance Verification

- [ ] **Step 1: Confirm determinism — same seed and actions produce identical trajectory hash**

```bash
pytest tests/gmail_env/test_determinism.py::test_same_seed_and_actions_produce_identical_trajectory -v
```

Expected: `PASSED`

- [ ] **Step 2: Confirm every action produces a state diff**

```bash
pytest tests/gmail_env/test_determinism.py::test_trajectory_exports_valid_jsonl -v
```

Expected: `PASSED` (diff key present in JSONL output)

- [ ] **Step 3: Confirm invalid actions fail cleanly**

```bash
pytest tests/runtime/test_env.py::test_invalid_action_does_not_mutate_state -v
pytest tests/gmail_env/test_determinism.py::test_invalid_action_does_not_corrupt_trajectory_hashes -v
```

Expected: both `PASSED`

- [ ] **Step 4: Confirm verifiers run after every step**

```bash
pytest tests/gmail_env/test_determinism.py::test_episode_task_completion_produces_positive_reward -v
```

Expected: `PASSED`

- [ ] **Step 5: Confirm JSONL export of complete episode**

```bash
pytest tests/gmail_env/test_determinism.py::test_trajectory_exports_valid_jsonl -v
```

Expected: `PASSED`

- [ ] **Step 6: Confirm basic Gymnasium training loop works**

```bash
python -c "
from examples.gmail_env.gym_wrapper import build_gmail_env

env = build_gmail_env()
task = {'name': 'reply_to_customer', 'verifier_id': 'reply_to_customer', 'inputs': {}}
obs, info = env.reset(seed=0, options={'task': task})
thread_id = next(iter(obs['threads']))
env._current_task = {**task, 'inputs': {'thread_id': thread_id}}

for _ in range(5):
    action = {'type': 'reply_email', 'thread_id': thread_id, 'body': 'Hello'}
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'reward={reward:.3f} terminated={terminated}')
    if terminated or truncated:
        break

print('Gymnasium loop: OK')
"
```

Expected: prints reward lines and `Gymnasium loop: OK`

- [ ] **Step 7: Final commit**

```bash
git add .
git commit -m "feat: M1 complete — deterministic runtime kernel + Gmail reference environment"
```
