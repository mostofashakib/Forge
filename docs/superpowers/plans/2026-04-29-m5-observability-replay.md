# M5: Observability & Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add episode trace storage, branch replay, failure clustering, a live WebSocket episode runner, and three frontend pages (Dashboard, Episode Replay, Environment Graph).

**Architecture:** `TelemetryClient` writes `StepSnapshot` records to SQLite + JSONL after each `ForgeEnv.step()`. `ReplayService` loads stored episodes for fast-forward branching. A FastAPI WebSocket endpoint spawns asyncio background tasks that run `ForgeEnv` and stream step events to connected clients. `EpisodeService` aggregates stats and drives `FailureClusterer`. Three Next.js 16 pages visualise episode data.

**Tech Stack:** Python 3.11, SQLAlchemy (sync), FastAPI, asyncio, Next.js 16, @xyflow/react, Tailwind CSS, shadcn/ui.

---

## File Map

| Path | Change | Purpose |
|------|--------|---------|
| `backend/app/models.py` | MODIFY | Add `Episode` + `EpisodeStep` ORM models |
| `forge/runtime/telemetry.py` | CREATE | `TelemetryClient` — writes steps to SQLite + JSONL |
| `forge/runtime/policy.py` | CREATE | `RandomPolicy` — picks random action type |
| `forge/runtime/replay.py` | CREATE | `ReplayService` — load episode, branch_from |
| `forge/runtime/clustering.py` | CREATE | `FailureClusterer` — group failed episodes by first failed check |
| `forge/runtime/env.py` | MODIFY | Inject `TelemetryClient`, expose `action_types`, track `_total_reward` |
| `backend/app/services/episode_service.py` | CREATE | CRUD + stats aggregation |
| `backend/app/services/runner_service.py` | CREATE | asyncio background task that runs `ForgeEnv` |
| `backend/app/api/episodes.py` | CREATE | REST endpoints + WebSocket stream |
| `backend/app/api/envs.py` | MODIFY | Add `/stats` and `/compiler-input` endpoints |
| `backend/app/main.py` | MODIFY | Include episodes router |
| `frontend/components/RewardBreakdown.tsx` | CREATE | Reward component breakdown table |
| `frontend/components/EpisodeTimeline.tsx` | CREATE | Step timeline with branch button |
| `frontend/app/environments/[env_name]/replay/[episode_id]/page.tsx` | CREATE | Episode Replay page |
| `frontend/app/dashboard/page.tsx` | CREATE | Dashboard page (stats + failures + recent episodes) |
| `frontend/components/EnvironmentGraph.tsx` | CREATE | React Flow live graph, WebSocket-fed |
| `frontend/app/environments/[env_name]/graph/page.tsx` | CREATE | Environment Graph page |
| `tests/runtime/test_telemetry.py` | CREATE | TelemetryClient tests |
| `tests/runtime/test_replay.py` | CREATE | ReplayService tests |
| `tests/runtime/test_clustering.py` | CREATE | FailureClusterer tests |
| `tests/backend/test_episode_api.py` | MODIFY | Episode API + stats tests (file already exists for other tests) |

---

## Task 1: Episode + EpisodeStep SQLAlchemy Models

**Files:**
- Modify: `backend/app/models.py`
- Modify: `tests/backend/test_episode_api.py` (create if needed — add to existing file)

- [ ] **Step 1: Write failing tests for Episode and EpisodeStep table creation**

Create `tests/backend/test_episode_api.py` (a new file — do NOT modify existing test files):

```python
# tests/backend/test_episode_api.py
from __future__ import annotations
import json
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base


def make_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_episode_model_can_be_created():
    from backend.app.models import Episode
    db = make_memory_db()
    ep = Episode(
        id="ep_00000001",
        env_name="test_env",
        task_name="test_task",
        seed=1,
        agent_id="random_policy",
        status="running",
        total_steps=0,
        total_reward=0.0,
        passed=False,
        started_at=datetime.now(timezone.utc),
    )
    db.add(ep)
    db.commit()
    fetched = db.get(Episode, "ep_00000001")
    assert fetched.env_name == "test_env"
    assert fetched.status == "running"
    db.close()


def test_episode_step_model_can_be_created():
    from backend.app.models import Episode, EpisodeStep
    db = make_memory_db()
    ep = Episode(
        id="ep_00000001",
        env_name="test_env",
        task_name="test_task",
        seed=1,
        agent_id="random_policy",
        status="running",
        total_steps=0,
        total_reward=0.0,
        passed=False,
        started_at=datetime.now(timezone.utc),
    )
    db.add(ep)
    db.flush()
    step = EpisodeStep(
        episode_id="ep_00000001",
        step_index=0,
        action='{"type": "increment"}',
        reward=0.5,
        verifier_results="[]",
        diff="{}",
        events="[]",
        state_hash_before="abc",
        state_hash_after="def",
        terminated=False,
        truncated=False,
    )
    db.add(step)
    db.commit()
    fetched = db.query(EpisodeStep).filter_by(episode_id="ep_00000001").first()
    assert fetched.step_index == 0
    assert fetched.reward == 0.5
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /path/to/worktree
pytest tests/backend/test_episode_api.py::test_episode_model_can_be_created tests/backend/test_episode_api.py::test_episode_step_model_can_be_created -v
```

Expected: `ImportError` or `AttributeError` because `Episode` and `EpisodeStep` don't exist yet.

- [ ] **Step 3: Add Episode and EpisodeStep to models.py**

Open `backend/app/models.py`. The file currently imports `String, Text, DateTime` and defines `CompileJob`. Add the following imports and classes:

```python
# backend/app/models.py — full file after modification
from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, Integer, Boolean, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.database import Base


class CompileJob(Base):
    __tablename__ = "compile_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_name: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="pending")
    prompt: Mapped[str] = mapped_column(Text)
    compiler_input_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_path: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    env_name: Mapped[str] = mapped_column(String, index=True)
    task_name: Mapped[str] = mapped_column(String)
    seed: Mapped[int] = mapped_column(Integer)
    agent_id: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="running")
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    total_reward: Mapped[float] = mapped_column(Float, default=0.0)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    jsonl_path: Mapped[str | None] = mapped_column(String, nullable=True)


class EpisodeStep(Base):
    __tablename__ = "episode_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_id: Mapped[str] = mapped_column(String, ForeignKey("episodes.id"), index=True)
    step_index: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(Text)
    reward: Mapped[float] = mapped_column(Float)
    verifier_results: Mapped[str] = mapped_column(Text)
    diff: Mapped[str] = mapped_column(Text)
    events: Mapped[str] = mapped_column(Text)
    state_hash_before: Mapped[str] = mapped_column(String)
    state_hash_after: Mapped[str] = mapped_column(String)
    terminated: Mapped[bool] = mapped_column(Boolean)
    truncated: Mapped[bool] = mapped_column(Boolean)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/backend/test_episode_api.py::test_episode_model_can_be_created tests/backend/test_episode_api.py::test_episode_step_model_can_be_created -v
```

Expected: 2 passed.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
pytest --tb=short -q
```

Expected: all existing tests pass (200 total).

- [ ] **Step 6: Commit**

```bash
git add backend/app/models.py tests/backend/test_episode_api.py
git commit -m "feat: add Episode and EpisodeStep SQLAlchemy models"
```

---

## Task 2: TelemetryClient

**Files:**
- Create: `forge/runtime/telemetry.py`
- Create: `tests/runtime/test_telemetry.py`

**Design:** `TelemetryClient` does NOT create the `Episode` row — that is the caller's responsibility (RunnerService). It receives an existing `episode_id` and writes `EpisodeStep` rows + JSONL. `complete_episode` updates the `Episode` row fields.

- [ ] **Step 1: Write failing tests**

Create `tests/runtime/test_telemetry.py`:

```python
# tests/runtime/test_telemetry.py
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.models import Episode, EpisodeStep
from forge.runtime.snapshot import StepSnapshot


def make_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    # Pre-create Episode row (as RunnerService would before constructing TelemetryClient)
    ep = Episode(
        id="ep_00000001",
        env_name="test_env",
        task_name="test_task",
        seed=1,
        agent_id="random_policy",
        status="running",
        total_steps=0,
        total_reward=0.0,
        passed=False,
        started_at=datetime.now(timezone.utc),
    )
    db.add(ep)
    db.commit()
    return db


def make_snapshot(step_index: int = 0, terminated: bool = False) -> StepSnapshot:
    return StepSnapshot(
        episode_id="ep_00000001",
        step_index=step_index,
        state_hash_before="abc",
        state_hash_after="def",
        action={"type": "increment"},
        events=[],
        reward=0.5,
        verifier_results=[],
        diff={"added": {}, "changed": {}, "removed": {}},
        terminated=terminated,
        truncated=False,
    )


def make_client(db, jsonl_path=None):
    from forge.runtime.telemetry import TelemetryClient
    return TelemetryClient(
        episode_id="ep_00000001",
        db_session=db,
        jsonl_path=jsonl_path,
    )


def test_record_step_writes_sqlite_row():
    db = make_db()
    client = make_client(db)
    client.record_step(make_snapshot())
    step = db.query(EpisodeStep).filter_by(episode_id="ep_00000001").first()
    assert step is not None
    assert step.step_index == 0
    assert json.loads(step.action) == {"type": "increment"}
    assert step.reward == 0.5
    assert step.state_hash_before == "abc"


def test_record_step_writes_jsonl_line(tmp_path):
    db = make_db()
    jsonl_path = tmp_path / "ep_00000001.jsonl"
    client = make_client(db, jsonl_path=jsonl_path)
    client.record_step(make_snapshot())
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["episode_id"] == "ep_00000001"
    assert data["step_index"] == 0


def test_record_step_appends_multiple_jsonl_lines(tmp_path):
    db = make_db()
    jsonl_path = tmp_path / "ep_00000001.jsonl"
    client = make_client(db, jsonl_path=jsonl_path)
    client.record_step(make_snapshot(step_index=0))
    client.record_step(make_snapshot(step_index=1))
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_record_step_no_jsonl_path_does_not_raise():
    db = make_db()
    client = make_client(db, jsonl_path=None)
    client.record_step(make_snapshot())  # should not raise


def test_complete_episode_updates_row():
    db = make_db()
    client = make_client(db)
    client.complete_episode(total_reward=1.5, passed=True, total_steps=7)
    ep = db.get(Episode, "ep_00000001")
    assert ep.status == "completed"
    assert ep.total_reward == 1.5
    assert ep.passed is True
    assert ep.total_steps == 7
    assert ep.completed_at is not None


def test_complete_episode_failed_sets_passed_false():
    db = make_db()
    client = make_client(db)
    client.complete_episode(total_reward=-0.5, passed=False, total_steps=50)
    ep = db.get(Episode, "ep_00000001")
    assert ep.status == "completed"
    assert ep.passed is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/runtime/test_telemetry.py -v
```

Expected: `ImportError: cannot import name 'TelemetryClient'`.

- [ ] **Step 3: Implement TelemetryClient**

Create `forge/runtime/telemetry.py`:

```python
# forge/runtime/telemetry.py
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from backend.app.models import Episode, EpisodeStep
from forge.runtime.snapshot import StepSnapshot


class TelemetryClient:
    def __init__(
        self,
        episode_id: str,
        db_session,
        jsonl_path: Path | None = None,
    ) -> None:
        self._episode_id = episode_id
        self._db = db_session
        self._jsonl_path = jsonl_path

    def record_step(self, snapshot: StepSnapshot) -> None:
        step = EpisodeStep(
            episode_id=self._episode_id,
            step_index=snapshot.step_index,
            action=json.dumps(snapshot.action),
            reward=snapshot.reward,
            verifier_results=json.dumps(snapshot.verifier_results),
            diff=json.dumps(snapshot.diff),
            events=json.dumps(snapshot.events),
            state_hash_before=snapshot.state_hash_before,
            state_hash_after=snapshot.state_hash_after,
            terminated=snapshot.terminated,
            truncated=snapshot.truncated,
        )
        self._db.add(step)
        self._db.commit()
        if self._jsonl_path is not None:
            with open(self._jsonl_path, "a") as f:
                f.write(snapshot.model_dump_json() + "\n")

    def complete_episode(
        self, total_reward: float, passed: bool, total_steps: int
    ) -> None:
        ep = self._db.get(Episode, self._episode_id)
        ep.status = "completed"
        ep.total_reward = total_reward
        ep.passed = passed
        ep.total_steps = total_steps
        ep.completed_at = datetime.now(timezone.utc)
        self._db.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/runtime/test_telemetry.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Run full suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add forge/runtime/telemetry.py tests/runtime/test_telemetry.py
git commit -m "feat: add TelemetryClient for step recording and episode completion"
```

---

## Task 3: ForgeEnv Telemetry Injection

**Files:**
- Modify: `forge/runtime/env.py`
- Modify: `tests/runtime/test_env.py` (append new tests only — do not change existing tests or `build_env`)

**Changes to ForgeEnv:**
1. Add `telemetry: TelemetryClient | None = None` to `__init__`
2. Add `self._total_reward: float = 0.0` to `__init__`
3. Add `action_types` property (`frozenset[str]`)
4. In `reset()`: reset `self._total_reward = 0.0`
5. In `step()`: accumulate `_total_reward`, call `record_step`, call `complete_episode` when `terminated or truncated`
6. In `_record_invalid_step()`: call `record_step` if telemetry is set

- [ ] **Step 1: Write failing tests**

Append these tests to the **end** of `tests/runtime/test_env.py`. Do NOT modify any existing code in that file.

```python
# --- Appended to tests/runtime/test_env.py ---

class MockTelemetry:
    def __init__(self):
        self.steps = []
        self.completions = []

    def record_step(self, snapshot):
        self.steps.append(snapshot)

    def complete_episode(self, total_reward, passed, total_steps):
        self.completions.append({"total_reward": total_reward, "passed": passed, "total_steps": total_steps})


def build_env_with_telemetry(telemetry, max_steps: int = 10) -> ForgeEnv:
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
        telemetry=telemetry,
    )


def test_forgeenv_telemetry_defaults_to_none():
    env = build_env()
    assert env._telemetry is None


def test_forgeenv_action_types_returns_registered_types():
    env = build_env()
    assert "increment" in env.action_types


def test_forgeenv_telemetry_records_step_on_valid_action():
    telemetry = MockTelemetry()
    env = build_env_with_telemetry(telemetry)
    env.reset(seed=1)
    env.step({"type": "increment"})
    assert len(telemetry.steps) == 1
    assert telemetry.steps[0].step_index == 0
    assert telemetry.steps[0].action == {"type": "increment"}


def test_forgeenv_telemetry_records_step_on_invalid_action():
    telemetry = MockTelemetry()
    env = build_env_with_telemetry(telemetry)
    env.reset(seed=1)
    env.step({"type": "nonexistent_action"})
    assert len(telemetry.steps) == 1


def test_forgeenv_telemetry_calls_complete_episode_on_truncation():
    telemetry = MockTelemetry()
    env = build_env_with_telemetry(telemetry, max_steps=1)
    env.reset(seed=1)
    _, reward, terminated, truncated, _ = env.step({"type": "increment"})
    assert truncated is True
    assert len(telemetry.completions) == 1
    assert telemetry.completions[0]["passed"] is False
    assert telemetry.completions[0]["total_steps"] == 1


def test_forgeenv_telemetry_none_does_not_raise():
    env = build_env()  # no telemetry
    env.reset(seed=1)
    env.step({"type": "increment"})  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/runtime/test_env.py::test_forgeenv_telemetry_defaults_to_none tests/runtime/test_env.py::test_forgeenv_action_types_returns_registered_types -v
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'telemetry'` and `AttributeError`.

- [ ] **Step 3: Implement changes to ForgeEnv**

Replace `forge/runtime/env.py` with the following (every change is marked `# M5`):

```python
# forge/runtime/env.py
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol
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

if TYPE_CHECKING:
    from forge.runtime.telemetry import TelemetryClient


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
        telemetry: "TelemetryClient | None" = None,  # M5
    ) -> None:
        super().__init__()
        self.env_spec = env_spec
        self._factory = initial_state_factory
        self._transition_engine = transition_engine
        self._verifier_engine = verifier_engine
        self._reward_engine = reward_engine
        self._action_validator = ActionValidator(transition_engine.action_types)
        self._telemetry = telemetry  # M5

        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})

        self._ctx: RuntimeContext | None = None
        self._state_store: StateStore | None = None
        self._traj_store: TrajectoryStore | None = None
        self._current_task: dict | None = None
        self._step_count: int = 0
        self._episode_id: str | None = None
        self._invalid_action_count: int = 0
        self._total_reward: float = 0.0  # M5

    @property
    def action_types(self) -> frozenset[str]:  # M5
        return frozenset(self._action_validator._valid_types)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        actual_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        opts = options or {}

        self._ctx = RuntimeContext(seed=actual_seed)
        self._episode_id = f"ep_{actual_seed:08x}"
        initial_state = self._factory.create(self._ctx, opts)
        self._state_store = StateStore(initial_state)
        self._traj_store = TrajectoryStore(self._episode_id)
        self._current_task = opts.get("task", self.env_spec.default_task)
        self._step_count = 0
        self._invalid_action_count = 0
        self._total_reward = 0.0  # M5

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
        trajectory = self._traj_store.to_trajectory_with_events(result.events)
        verifier_results = self._verifier_engine.run_all(
            state_after, trajectory, self._current_task
        )
        task_with_meta = {**(self._current_task or {}), "invalid_action_count": self._invalid_action_count}
        reward_breakdown = self._reward_engine.compute(
            state_after, trajectory, verifier_results, task_with_meta
        )

        self._step_count += 1
        self._total_reward += reward_breakdown.total_reward  # M5
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

        if self._telemetry:  # M5
            self._telemetry.record_step(snapshot)
        if (terminated or truncated) and self._telemetry:  # M5
            self._telemetry.complete_episode(self._total_reward, terminated, self._step_count)

        return state_after, reward_breakdown.total_reward, terminated, truncated, {
            "episode_id": self._episode_id,
            "verifier_results": [vr.model_dump() for vr in verifier_results],
            "reward_breakdown": reward_breakdown.model_dump(),
            "events": result.events,
        }

    def _record_invalid_step(self, hash_before: str, action: dict) -> None:
        self._step_count += 1
        self._invalid_action_count += 1
        snapshot = StepSnapshot(
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
        self._traj_store.record(snapshot)
        if self._telemetry:  # M5
            self._telemetry.record_step(snapshot)
```

- [ ] **Step 4: Run new telemetry tests**

```bash
pytest tests/runtime/test_env.py -k "telemetry or action_types" -v
```

Expected: 6 passed.

- [ ] **Step 5: Run full test suite**

```bash
pytest --tb=short -q
```

Expected: all existing tests pass plus new ones.

- [ ] **Step 6: Commit**

```bash
git add forge/runtime/env.py tests/runtime/test_env.py
git commit -m "feat: inject TelemetryClient into ForgeEnv; expose action_types; track total_reward"
```

---

## Task 4: RandomPolicy + ReplayService

**Files:**
- Create: `forge/runtime/policy.py`
- Create: `forge/runtime/replay.py`
- Create: `tests/runtime/test_replay.py`

- [ ] **Step 1: Write failing tests for ReplayService**

Create `tests/runtime/test_replay.py`:

```python
# tests/runtime/test_replay.py
from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.models import Episode, EpisodeStep
from forge.runtime.replay import ReplayService


def make_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def insert_episode_with_steps(db, n_steps: int, episode_id: str = "ep_00000001") -> str:
    ep = Episode(
        id=episode_id,
        env_name="test_env",
        task_name="test_task",
        seed=1,
        agent_id="random_policy",
        status="completed",
        total_steps=n_steps,
        total_reward=0.5 * n_steps,
        passed=False,
        started_at=datetime.now(timezone.utc),
    )
    db.add(ep)
    for i in range(n_steps):
        db.add(EpisodeStep(
            episode_id=episode_id,
            step_index=i,
            action=f'{{"type": "action_{i}"}}',
            reward=0.5,
            verifier_results="[]",
            diff="{}",
            events="[]",
            state_hash_before="abc",
            state_hash_after="def",
            terminated=False,
            truncated=(i == n_steps - 1),
        ))
    db.commit()
    return episode_id


def test_load_episode_returns_steps_in_order():
    db = make_db()
    insert_episode_with_steps(db, 3)
    record = ReplayService().load_episode("ep_00000001", db)
    assert [s.step_index for s in record.steps] == [0, 1, 2]


def test_load_episode_returns_episode_metadata():
    db = make_db()
    insert_episode_with_steps(db, 2)
    record = ReplayService().load_episode("ep_00000001", db)
    assert record.episode.env_name == "test_env"
    assert record.episode.total_steps == 2


def test_branch_from_returns_n_actions():
    db = make_db()
    insert_episode_with_steps(db, 5)
    actions = ReplayService().branch_from("ep_00000001", 3, db)
    assert len(actions) == 3
    assert all(isinstance(a, dict) for a in actions)


def test_branch_from_returns_correct_action_sequence():
    db = make_db()
    insert_episode_with_steps(db, 5)
    actions = ReplayService().branch_from("ep_00000001", 2, db)
    assert actions[0] == {"type": "action_0"}
    assert actions[1] == {"type": "action_1"}


def test_branch_from_step_0_returns_empty():
    db = make_db()
    insert_episode_with_steps(db, 3)
    actions = ReplayService().branch_from("ep_00000001", 0, db)
    assert actions == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/runtime/test_replay.py -v
```

Expected: `ImportError: cannot import name 'ReplayService'`.

- [ ] **Step 3: Implement RandomPolicy**

Create `forge/runtime/policy.py`:

```python
# forge/runtime/policy.py
from __future__ import annotations
import random


class RandomPolicy:
    def __init__(self, action_types: frozenset[str] | set[str]) -> None:
        self._action_types = sorted(action_types)

    def act(self, obs: dict) -> dict:
        return {"type": random.choice(self._action_types)}
```

- [ ] **Step 4: Implement ReplayService**

Create `forge/runtime/replay.py`:

```python
# forge/runtime/replay.py
from __future__ import annotations
import json
from dataclasses import dataclass
from sqlalchemy.orm import Session
from backend.app.models import Episode, EpisodeStep


@dataclass
class EpisodeRecord:
    episode: Episode
    steps: list[EpisodeStep]


class ReplayService:
    def load_episode(self, episode_id: str, db: Session) -> EpisodeRecord:
        ep = db.get(Episode, episode_id)
        steps = (
            db.query(EpisodeStep)
            .filter_by(episode_id=episode_id)
            .order_by(EpisodeStep.step_index)
            .all()
        )
        return EpisodeRecord(episode=ep, steps=steps)

    def branch_from(self, episode_id: str, step_n: int, db: Session) -> list[dict]:
        steps = (
            db.query(EpisodeStep)
            .filter(
                EpisodeStep.episode_id == episode_id,
                EpisodeStep.step_index < step_n,
            )
            .order_by(EpisodeStep.step_index)
            .all()
        )
        return [json.loads(s.action) for s in steps]
```

- [ ] **Step 5: Run replay tests**

```bash
pytest tests/runtime/test_replay.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Run full suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add forge/runtime/policy.py forge/runtime/replay.py tests/runtime/test_replay.py
git commit -m "feat: add RandomPolicy stub and ReplayService"
```

---

## Task 5: FailureClusterer

**Files:**
- Create: `forge/runtime/clustering.py`
- Create: `tests/runtime/test_clustering.py`

- [ ] **Step 1: Write failing tests**

Create `tests/runtime/test_clustering.py`:

```python
# tests/runtime/test_clustering.py
from __future__ import annotations
import json
from unittest.mock import MagicMock
from forge.runtime.clustering import FailureClusterer
from forge.runtime.replay import EpisodeRecord


def make_record(episode_id: str, passed: bool, first_failed_check: str | None) -> EpisodeRecord:
    ep = MagicMock()
    ep.id = episode_id
    ep.passed = passed

    if first_failed_check is not None:
        vr_json = json.dumps([{
            "verifier_id": "v1",
            "passed": False,
            "score": 0.0,
            "checks": [{"name": first_failed_check, "passed": False, "score": 0.0}],
        }])
        step = MagicMock()
        step.verifier_results = vr_json
        steps = [step]
    else:
        steps = []

    return EpisodeRecord(episode=ep, steps=steps)


def test_cluster_groups_by_first_failed_check():
    records = [
        make_record("ep_1", passed=False, first_failed_check="ticket_solved"),
        make_record("ep_2", passed=False, first_failed_check="ticket_solved"),
        make_record("ep_3", passed=False, first_failed_check="ticket_solved"),
        make_record("ep_4", passed=False, first_failed_check="comment_added"),
        make_record("ep_5", passed=False, first_failed_check="comment_added"),
    ]
    clusters = FailureClusterer().cluster(records)
    assert clusters[0].check_name == "ticket_solved"
    assert clusters[0].count == 3
    assert clusters[1].check_name == "comment_added"
    assert clusters[1].count == 2


def test_cluster_skips_passed_episodes():
    records = [
        make_record("ep_1", passed=True, first_failed_check="ticket_solved"),
        make_record("ep_2", passed=False, first_failed_check="ticket_solved"),
    ]
    clusters = FailureClusterer().cluster(records)
    assert len(clusters) == 1
    assert clusters[0].count == 1


def test_cluster_returns_at_most_5_clusters():
    records = [
        make_record(f"ep_{i}", passed=False, first_failed_check=f"check_{i}")
        for i in range(10)
    ]
    clusters = FailureClusterer().cluster(records)
    assert len(clusters) <= 5


def test_cluster_episode_ids_capped_at_5():
    records = [
        make_record(f"ep_{i}", passed=False, first_failed_check="same_check")
        for i in range(10)
    ]
    clusters = FailureClusterer().cluster(records)
    assert len(clusters[0].episode_ids) <= 5


def test_cluster_sorted_by_count_descending():
    records = [
        make_record("ep_1", passed=False, first_failed_check="rare"),
        make_record("ep_2", passed=False, first_failed_check="common"),
        make_record("ep_3", passed=False, first_failed_check="common"),
        make_record("ep_4", passed=False, first_failed_check="common"),
    ]
    clusters = FailureClusterer().cluster(records)
    assert clusters[0].check_name == "common"


def test_cluster_skips_episodes_with_no_failed_checks():
    records = [
        make_record("ep_1", passed=False, first_failed_check=None),
    ]
    clusters = FailureClusterer().cluster(records)
    assert clusters == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/runtime/test_clustering.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement FailureClusterer**

Create `forge/runtime/clustering.py`:

```python
# forge/runtime/clustering.py
from __future__ import annotations
import json
from dataclasses import dataclass, field
from forge.runtime.replay import EpisodeRecord


@dataclass
class FailureCluster:
    check_name: str
    count: int
    episode_ids: list[str] = field(default_factory=list)


class FailureClusterer:
    def cluster(self, episodes: list[EpisodeRecord]) -> list[FailureCluster]:
        buckets: dict[str, list[str]] = {}
        for record in episodes:
            if record.episode.passed:
                continue
            check_name = self._first_failed_check(record)
            if check_name is None:
                continue
            buckets.setdefault(check_name, []).append(record.episode.id)

        clusters = [
            FailureCluster(
                check_name=name,
                count=len(ids),
                episode_ids=ids[:5],
            )
            for name, ids in buckets.items()
        ]
        clusters.sort(key=lambda c: c.count, reverse=True)
        return clusters[:5]

    def _first_failed_check(self, record: EpisodeRecord) -> str | None:
        for step in record.steps:
            results = json.loads(step.verifier_results)
            for vr in results:
                for check in vr.get("checks", []):
                    if not check.get("passed", True):
                        return check.get("name")
        return None
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/runtime/test_clustering.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run full suite**

```bash
pytest --tb=short -q
```

- [ ] **Step 6: Commit**

```bash
git add forge/runtime/clustering.py tests/runtime/test_clustering.py
git commit -m "feat: add FailureClusterer"
```

---

## Task 6: EpisodeService

**Files:**
- Create: `backend/app/services/episode_service.py`

No separate test file — the service is exercised through the API tests in Task 8. Write a few direct unit tests in `tests/backend/test_episode_api.py`.

- [ ] **Step 1: Write failing service unit tests**

Append these to `tests/backend/test_episode_api.py`:

```python
# --- append to tests/backend/test_episode_api.py ---

def test_create_episode_inserts_row():
    from backend.app.services import episode_service
    db = make_memory_db()
    ep = episode_service.create_episode(
        episode_id="ep_aabbccdd",
        env_name="my_env",
        task_name="my_task",
        seed=42,
        agent_id="random",
        db=db,
    )
    assert ep.id == "ep_aabbccdd"
    assert ep.status == "running"
    fetched = db.get(__import__("backend.app.models", fromlist=["Episode"]).Episode, "ep_aabbccdd")
    assert fetched is not None
    db.close()


def test_get_episode_returns_none_for_unknown_id():
    from backend.app.services import episode_service
    db = make_memory_db()
    result = episode_service.get_episode("nonexistent", db)
    assert result is None
    db.close()


def test_get_stats_returns_zero_stats_for_empty_env():
    from backend.app.services import episode_service
    db = make_memory_db()
    stats = episode_service.get_stats("empty_env", db)
    assert stats["pass_rate"] == 0.0
    assert stats["avg_reward"] == 0.0
    assert stats["policy_violation_count"] == 0
    assert stats["top_failures"] == []
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/backend/test_episode_api.py::test_create_episode_inserts_row tests/backend/test_episode_api.py::test_get_episode_returns_none_for_unknown_id tests/backend/test_episode_api.py::test_get_stats_returns_zero_stats_for_empty_env -v
```

Expected: `ImportError: cannot import name 'episode_service'`.

- [ ] **Step 3: Implement EpisodeService**

Create `backend/app/services/episode_service.py`:

```python
# backend/app/services/episode_service.py
from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from backend.app.models import Episode, EpisodeStep
from forge.runtime.replay import EpisodeRecord, ReplayService
from forge.runtime.clustering import FailureClusterer


def create_episode(
    episode_id: str,
    env_name: str,
    task_name: str,
    seed: int,
    agent_id: str,
    db: Session,
    jsonl_path: str | None = None,
) -> Episode:
    ep = Episode(
        id=episode_id,
        env_name=env_name,
        task_name=task_name,
        seed=seed,
        agent_id=agent_id,
        status="running",
        total_steps=0,
        total_reward=0.0,
        passed=False,
        started_at=datetime.now(timezone.utc),
        jsonl_path=jsonl_path,
    )
    db.add(ep)
    db.commit()
    return ep


def get_episode(episode_id: str, db: Session) -> Episode | None:
    return db.get(Episode, episode_id)


def get_episode_steps(episode_id: str, db: Session) -> list[EpisodeStep]:
    return (
        db.query(EpisodeStep)
        .filter_by(episode_id=episode_id)
        .order_by(EpisodeStep.step_index)
        .all()
    )


def list_episodes(env_name: str, db: Session, limit: int = 20) -> list[Episode]:
    return (
        db.query(Episode)
        .filter_by(env_name=env_name)
        .order_by(Episode.started_at.desc())
        .limit(limit)
        .all()
    )


def get_stats(env_name: str, db: Session) -> dict:
    episodes = (
        db.query(Episode)
        .filter_by(env_name=env_name, status="completed")
        .order_by(Episode.started_at.desc())
        .limit(100)
        .all()
    )
    n = len(episodes)
    if n == 0:
        return {
            "pass_rate": 0.0,
            "avg_reward": 0.0,
            "avg_steps": 0.0,
            "policy_violation_count": 0,
            "top_failures": [],
        }

    pass_rate = sum(1 for ep in episodes if ep.passed) / n
    avg_reward = sum(ep.total_reward for ep in episodes) / n
    avg_steps = sum(ep.total_steps for ep in episodes) / n

    episode_ids = [ep.id for ep in episodes]
    violation_steps = (
        db.query(EpisodeStep)
        .filter(
            EpisodeStep.episode_id.in_(episode_ids),
            EpisodeStep.events.contains("policy_violation"),
        )
        .all()
    )
    policy_violation_count = len({s.episode_id for s in violation_steps})

    failed_episodes = [ep for ep in episodes if not ep.passed]
    replay = ReplayService()
    records = [replay.load_episode(ep.id, db) for ep in failed_episodes]
    clusters = FailureClusterer().cluster(records)

    return {
        "pass_rate": round(pass_rate, 4),
        "avg_reward": round(avg_reward, 4),
        "avg_steps": round(avg_steps, 4),
        "policy_violation_count": policy_violation_count,
        "top_failures": [
            {
                "check_name": c.check_name,
                "count": c.count,
                "episode_ids": c.episode_ids,
            }
            for c in clusters
        ],
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/backend/test_episode_api.py::test_create_episode_inserts_row tests/backend/test_episode_api.py::test_get_episode_returns_none_for_unknown_id tests/backend/test_episode_api.py::test_get_stats_returns_zero_stats_for_empty_env -v
```

Expected: 3 passed.

- [ ] **Step 5: Run full suite**

```bash
pytest --tb=short -q
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/episode_service.py tests/backend/test_episode_api.py
git commit -m "feat: add EpisodeService with CRUD and stats aggregation"
```

---

## Task 7: RunnerService

**Files:**
- Create: `backend/app/services/runner_service.py`

`RunnerService` dynamically loads a compiled env from `generated_envs/{env_name}/gym_wrapper.py`, wires in `TelemetryClient`, and runs the episode in an asyncio background task. It exposes `episode_queues` and `episode_tasks` dicts for use by the WebSocket endpoint.

- [ ] **Step 1: Create RunnerService**

Create `backend/app/services/runner_service.py`:

```python
# backend/app/services/runner_service.py
from __future__ import annotations
import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from forge.runtime.policy import RandomPolicy
from forge.runtime.telemetry import TelemetryClient
from backend.app.services import episode_service
from backend.app.database import get_session_factory

# episode_id → asyncio.Queue of step event dicts
episode_queues: dict[str, asyncio.Queue] = {}

# episode_id → asyncio.Task
episode_tasks: dict[str, asyncio.Task] = {}


def _load_env(env_name: str, telemetry: TelemetryClient):
    """Dynamically import the generated gym_wrapper and build the ForgeEnv."""
    envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
    parent = str(envs_root.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(f"generated_envs.{env_name}.gym_wrapper")
    build_fn = getattr(module, f"build_{env_name}_env")
    env = build_fn()
    env._telemetry = telemetry
    return env


async def start_episode(
    env_name: str,
    task_name: str,
    seed: int,
    agent_id: str,
) -> str:
    """Create Episode row, initialise queue, spawn background task, return episode_id."""
    episode_id = f"ep_{seed:08x}"

    # Create Episode row synchronously so it's immediately queryable
    SessionFactory = get_session_factory()
    db = SessionFactory()
    try:
        episode_service.create_episode(
            episode_id=episode_id,
            env_name=env_name,
            task_name=task_name,
            seed=seed,
            agent_id=agent_id,
            db=db,
        )
    finally:
        db.close()

    episode_queues[episode_id] = asyncio.Queue()
    task = asyncio.create_task(
        _run_episode(episode_id, env_name, task_name, seed, agent_id)
    )
    episode_tasks[episode_id] = task
    return episode_id


async def _run_episode(
    episode_id: str,
    env_name: str,
    task_name: str,
    seed: int,
    agent_id: str,
) -> None:
    queue = episode_queues.get(episode_id)
    SessionFactory = get_session_factory()
    db = SessionFactory()
    try:
        envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
        jsonl_dir = envs_root / env_name / "episodes"
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = jsonl_dir / f"{episode_id}.jsonl"

        telemetry = TelemetryClient(
            episode_id=episode_id,
            db_session=db,
            jsonl_path=jsonl_path,
        )
        env = _load_env(env_name, telemetry)
        policy = RandomPolicy(env.action_types)

        obs, info = env.reset(seed=seed)
        terminated = truncated = False

        while not (terminated or truncated):
            action = policy.act(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)

            event = {
                "type": "step",
                "step_index": env._step_count - 1,
                "action": action,
                "reward": reward,
                "diff": step_info.get("reward_breakdown", {}),
                "verifier_results": step_info.get("verifier_results", []),
                "events": step_info.get("events", []),
                "terminated": terminated,
            }
            if queue:
                await queue.put(event)
            await asyncio.sleep(0)

        complete_event = {
            "type": "complete",
            "total_reward": env._total_reward,
            "passed": terminated,
            "total_steps": env._step_count,
        }
        if queue:
            await queue.put(complete_event)
    except Exception as exc:
        if queue:
            await queue.put({"type": "error", "message": str(exc)})
    finally:
        db.close()
        episode_tasks.pop(episode_id, None)
```

- [ ] **Step 2: Verify RunnerService imports cleanly**

```bash
cd /path/to/worktree
python -c "from backend.app.services import runner_service; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Run full suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass (no regressions).

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/runner_service.py
git commit -m "feat: add RunnerService — asyncio background task runner for ForgeEnv episodes"
```

---

## Task 8: Episodes REST API + envs.py Updates + main.py

**Files:**
- Create: `backend/app/api/episodes.py`
- Modify: `backend/app/api/envs.py` (add `/stats` and `/compiler-input` endpoints)
- Modify: `backend/app/main.py` (include episodes router)
- Modify: `tests/backend/test_episode_api.py` (append API tests)

**REST endpoints:**
- `POST /api/episodes/` — start episode, returns `{"episode_id": "..."}`
- `GET /api/episodes/{episode_id}` — full episode with all steps
- `GET /api/episodes/?env_name=X` — list episodes for env
- `GET /api/episodes/{episode_id}/steps/{step_n}/branch` — action sequence 0..N-1
- `GET /api/envs/{env_name}/stats` — pass rate + failure clusters
- `GET /api/envs/{env_name}/compiler-input` — CompilerInput JSON for the env

- [ ] **Step 1: Write failing REST tests**

Append to `tests/backend/test_episode_api.py`:

```python
# --- append to tests/backend/test_episode_api.py ---
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    return TestClient(app)


@pytest.fixture
def api_client_with_episode(tmp_path, monkeypatch):
    """Client with one completed episode pre-inserted."""
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    from backend.app.models import Episode, EpisodeStep
    from sqlalchemy.orm import Session
    SessionFactory = database.get_session_factory()
    db = SessionFactory()
    ep = Episode(
        id="ep_0000002a",
        env_name="test_env",
        task_name="test_task",
        seed=42,
        agent_id="random_policy",
        status="completed",
        total_steps=2,
        total_reward=0.8,
        passed=True,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        completed_at=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
    )
    db.add(ep)
    for i in range(2):
        db.add(EpisodeStep(
            episode_id="ep_0000002a",
            step_index=i,
            action=f'{{"type": "action_{i}"}}',
            reward=0.4,
            verifier_results="[]",
            diff="{}",
            events="[]",
            state_hash_before="abc",
            state_hash_after="def",
            terminated=(i == 1),
            truncated=False,
        ))
    db.commit()
    db.close()
    return TestClient(app)


def test_post_episodes_returns_episode_id(api_client, monkeypatch):
    async def fake_start_episode(env_name, task_name, seed, agent_id):
        return f"ep_{seed:08x}"
    import backend.app.services.runner_service as rs
    monkeypatch.setattr(rs, "start_episode", fake_start_episode)
    resp = api_client.post("/api/episodes/", json={
        "env_name": "test_env",
        "task_name": "test_task",
        "seed": 1,
        "agent_id": "random_policy",
    })
    assert resp.status_code == 200
    assert resp.json()["episode_id"] == "ep_00000001"


def test_get_episode_returns_full_record(api_client_with_episode):
    resp = api_client_with_episode.get("/api/episodes/ep_0000002a")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "ep_0000002a"
    assert data["status"] == "completed"
    assert len(data["steps"]) == 2


def test_get_episode_returns_404_for_unknown(api_client):
    resp = api_client.get("/api/episodes/ep_unknown")
    assert resp.status_code == 404


def test_list_episodes_filters_by_env_name(api_client_with_episode):
    resp = api_client_with_episode.get("/api/episodes/?env_name=test_env")
    assert resp.status_code == 200
    ids = [ep["id"] for ep in resp.json()]
    assert "ep_0000002a" in ids


def test_branch_returns_action_sequence(api_client_with_episode):
    resp = api_client_with_episode.get("/api/episodes/ep_0000002a/steps/1/branch")
    assert resp.status_code == 200
    actions = resp.json()["actions"]
    assert len(actions) == 1
    assert actions[0] == {"type": "action_0"}


def test_get_env_stats_returns_pass_rate(api_client_with_episode):
    resp = api_client_with_episode.get("/api/envs/test_env/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "pass_rate" in data
    assert data["pass_rate"] == 1.0
    assert "top_failures" in data


def test_get_compiler_input_returns_json(api_client, tmp_path, monkeypatch):
    # Insert a completed compile job for "test_env"
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    from backend.app.models import CompileJob
    from forge.extraction.schemas import CompilerInput
    SessionFactory = database.get_session_factory()
    db = SessionFactory()
    ci = CompilerInput(project_name="test_env", domain="test", entities=[], actions=[], tasks=[])
    job = CompileJob(
        id="job_001",
        project_name="test_env",
        status="complete",
        prompt="test",
        compiler_input_json=ci.model_dump_json(),
    )
    db.add(job)
    db.commit()
    db.close()
    resp = api_client.get("/api/envs/test_env/compiler-input")
    assert resp.status_code == 200
    assert resp.json()["project_name"] == "test_env"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/backend/test_episode_api.py::test_post_episodes_returns_episode_id tests/backend/test_episode_api.py::test_get_episode_returns_full_record -v
```

Expected: `404` or routing errors.

- [ ] **Step 3: Create episodes.py REST router**

Create `backend/app/api/episodes.py`:

```python
# backend/app/api/episodes.py
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import episode_service, runner_service
from forge.runtime.replay import ReplayService

router = APIRouter(prefix="/api/episodes")


class StartEpisodeRequest(BaseModel):
    env_name: str
    task_name: str
    seed: int
    agent_id: str = "random_policy"


class StepOut(BaseModel):
    step_index: int
    action: str
    reward: float
    verifier_results: str
    diff: str
    events: str
    state_hash_before: str
    state_hash_after: str
    terminated: bool
    truncated: bool


class EpisodeOut(BaseModel):
    id: str
    env_name: str
    task_name: str
    seed: int
    agent_id: str
    status: str
    total_steps: int
    total_reward: float
    passed: bool
    steps: list[StepOut]


@router.post("/")
async def start_episode(req: StartEpisodeRequest):
    episode_id = await runner_service.start_episode(
        env_name=req.env_name,
        task_name=req.task_name,
        seed=req.seed,
        agent_id=req.agent_id,
    )
    return {"episode_id": episode_id}


@router.get("/{episode_id}", response_model=EpisodeOut)
def get_episode(episode_id: str, db: Session = Depends(get_db)):
    ep = episode_service.get_episode(episode_id, db)
    if not ep:
        raise HTTPException(status_code=404, detail="Episode not found")
    steps = episode_service.get_episode_steps(episode_id, db)
    return EpisodeOut(
        id=ep.id,
        env_name=ep.env_name,
        task_name=ep.task_name,
        seed=ep.seed,
        agent_id=ep.agent_id,
        status=ep.status,
        total_steps=ep.total_steps,
        total_reward=ep.total_reward,
        passed=ep.passed,
        steps=[
            StepOut(
                step_index=s.step_index,
                action=s.action,
                reward=s.reward,
                verifier_results=s.verifier_results,
                diff=s.diff,
                events=s.events,
                state_hash_before=s.state_hash_before,
                state_hash_after=s.state_hash_after,
                terminated=s.terminated,
                truncated=s.truncated,
            )
            for s in steps
        ],
    )


@router.get("/")
def list_episodes(env_name: str, db: Session = Depends(get_db)):
    episodes = episode_service.list_episodes(env_name, db)
    return [
        {
            "id": ep.id,
            "env_name": ep.env_name,
            "task_name": ep.task_name,
            "status": ep.status,
            "passed": ep.passed,
            "total_reward": ep.total_reward,
            "total_steps": ep.total_steps,
            "started_at": ep.started_at.isoformat() if ep.started_at else None,
        }
        for ep in episodes
    ]


@router.get("/{episode_id}/steps/{step_n}/branch")
def branch(episode_id: str, step_n: int, db: Session = Depends(get_db)):
    ep = episode_service.get_episode(episode_id, db)
    if not ep:
        raise HTTPException(status_code=404, detail="Episode not found")
    actions = ReplayService().branch_from(episode_id, step_n, db)
    return {"actions": actions}
```

- [ ] **Step 4: Add stats and compiler-input endpoints to envs.py**

Open `backend/app/api/envs.py` and append the following two endpoints (after the existing `update_config` endpoint):

```python
# append to backend/app/api/envs.py

from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import episode_service
from backend.app.models import CompileJob
from forge.extraction.schemas import CompilerInput


@router.get("/{env_name}/stats")
def get_env_stats(env_name: str, db: Session = Depends(get_db)):
    _validate_env_name(env_name)
    return episode_service.get_stats(env_name, db)


@router.get("/{env_name}/compiler-input")
def get_compiler_input(env_name: str, db: Session = Depends(get_db)):
    _validate_env_name(env_name)
    job = (
        db.query(CompileJob)
        .filter_by(project_name=env_name)
        .order_by(CompileJob.created_at.desc())
        .first()
    )
    if not job or not job.compiler_input_json:
        raise HTTPException(status_code=404, detail=f"No compiler input found for '{env_name}'")
    return CompilerInput.model_validate_json(job.compiler_input_json)
```

**Important:** The `envs.py` file must have `from fastapi import APIRouter, HTTPException` already. You need to add the new imports (`Session`, `get_db`, `episode_service`, `CompileJob`, `CompilerInput`) carefully — add them at the top with the other imports, not duplicating existing ones.

The full `backend/app/api/envs.py` after modification:

```python
# backend/app/api/envs.py
from __future__ import annotations
import os
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.models import CompileJob
from backend.app.services import episode_service
from forge.extraction.schemas import CompilerInput

router = APIRouter(prefix="/api/envs")


def _envs_root() -> Path:
    return Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))


def _validate_env_name(env_name: str) -> None:
    if ".." in env_name or "/" in env_name or "\\" in env_name:
        raise HTTPException(status_code=400, detail="Invalid environment name")


class ConfigPayload(BaseModel):
    yaml: str


@router.get("/", response_model=list[str])
def list_envs() -> list[str]:
    root = _envs_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


@router.get("/{env_name}/config", response_model=ConfigPayload)
def get_config(env_name: str) -> ConfigPayload:
    _validate_env_name(env_name)
    config_path = _envs_root() / env_name / "custom" / "config.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found for '{env_name}'")
    return ConfigPayload(yaml=config_path.read_text())


@router.put("/{env_name}/config", response_model=ConfigPayload)
def update_config(env_name: str, payload: ConfigPayload) -> ConfigPayload:
    _validate_env_name(env_name)
    custom_dir = _envs_root() / env_name / "custom"
    if not custom_dir.exists():
        raise HTTPException(status_code=404, detail=f"Environment '{env_name}' not found")
    config_path = custom_dir / "config.yaml"
    config_path.write_text(payload.yaml)
    return ConfigPayload(yaml=payload.yaml)


@router.get("/{env_name}/stats")
def get_env_stats(env_name: str, db: Session = Depends(get_db)):
    _validate_env_name(env_name)
    return episode_service.get_stats(env_name, db)


@router.get("/{env_name}/compiler-input")
def get_compiler_input(env_name: str, db: Session = Depends(get_db)):
    _validate_env_name(env_name)
    job = (
        db.query(CompileJob)
        .filter_by(project_name=env_name)
        .order_by(CompileJob.created_at.desc())
        .first()
    )
    if not job or not job.compiler_input_json:
        raise HTTPException(status_code=404, detail=f"No compiler input found for '{env_name}'")
    return CompilerInput.model_validate_json(job.compiler_input_json)
```

- [ ] **Step 5: Update main.py to include episodes router**

Replace `backend/app/main.py` with:

```python
# backend/app/main.py
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.compile import router as compile_router
from backend.app.api.envs import router as envs_router
from backend.app.api.episodes import router as episodes_router
from backend.app.database import init_db

app = FastAPI(title="Forge API", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(compile_router)
app.include_router(envs_router)
app.include_router(episodes_router)


@app.on_event("startup")
def startup():
    init_db()
```

- [ ] **Step 6: Run all new REST tests**

```bash
pytest tests/backend/test_episode_api.py -v
```

Expected: all tests pass (including the 3 unit tests from Task 6 and the new REST tests).

- [ ] **Step 7: Run full suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add backend/app/api/episodes.py backend/app/api/envs.py backend/app/main.py tests/backend/test_episode_api.py
git commit -m "feat: add episodes REST API, env stats, compiler-input endpoints"
```

---

## Task 9: WebSocket Stream Endpoint

**Files:**
- Modify: `backend/app/api/episodes.py` (append WebSocket endpoint)
- Modify: `tests/backend/test_episode_api.py` (append WebSocket tests)

The WebSocket endpoint `WS /api/episodes/{episode_id}/stream`:
- If the episode is completed, replay stored steps immediately then close.
- If the episode is running, drain events from the queue and forward to client.
- If the client sends `{"type":"fork","step":N}`, cancel the current task, get action sequence, start a new episode that fast-forwards to step N, and keep streaming.

- [ ] **Step 1: Write failing WebSocket test**

Append to `tests/backend/test_episode_api.py`:

```python
# --- append to tests/backend/test_episode_api.py ---

import asyncio


def test_websocket_stream_replays_completed_episode(api_client_with_episode):
    with api_client_with_episode.websocket_connect("/api/episodes/ep_0000002a/stream") as ws:
        events = []
        while True:
            msg = ws.receive_json()
            events.append(msg)
            if msg["type"] in ("complete", "error"):
                break
    step_events = [e for e in events if e["type"] == "step"]
    complete_events = [e for e in events if e["type"] == "complete"]
    assert len(step_events) == 2
    assert len(complete_events) == 1
    assert complete_events[0]["passed"] is True


def test_websocket_stream_running_episode(api_client, monkeypatch):
    from backend.app.services import runner_service

    # Pre-populate queue with synthetic events
    episode_id = "ep_00000099"
    runner_service.episode_queues[episode_id] = asyncio.Queue()

    # Pre-create episode row
    from backend.app import database
    SessionFactory = database.get_session_factory()
    db = SessionFactory()
    from backend.app.models import Episode
    ep = Episode(
        id=episode_id,
        env_name="test_env",
        task_name="test_task",
        seed=0x99,
        agent_id="random_policy",
        status="running",
        total_steps=0,
        total_reward=0.0,
        passed=False,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    db.add(ep)
    db.commit()
    db.close()

    # Put events into the queue directly (simulates background runner)
    runner_service.episode_queues[episode_id].put_nowait({
        "type": "step", "step_index": 0, "action": {"type": "a"},
        "reward": 0.5, "diff": {}, "verifier_results": [], "events": [], "terminated": False,
    })
    runner_service.episode_queues[episode_id].put_nowait({
        "type": "complete", "total_reward": 0.5, "passed": False, "total_steps": 1,
    })

    with api_client.websocket_connect(f"/api/episodes/{episode_id}/stream") as ws:
        msg1 = ws.receive_json()
        msg2 = ws.receive_json()

    assert msg1["type"] == "step"
    assert msg2["type"] == "complete"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/backend/test_episode_api.py::test_websocket_stream_replays_completed_episode tests/backend/test_episode_api.py::test_websocket_stream_running_episode -v
```

Expected: routing errors or `404` (no WebSocket route yet).

- [ ] **Step 3: Add WebSocket endpoint to episodes.py**

Append the following to `backend/app/api/episodes.py` (add imports at the top of the file as needed):

Add to the imports at top of file:
```python
import asyncio
import json as _json
from fastapi import WebSocket, WebSocketDisconnect
```

Add to the bottom of the file:

```python
@router.websocket("/{episode_id}/stream")
async def stream_episode(
    websocket: WebSocket, episode_id: str, db: Session = Depends(get_db)
):
    await websocket.accept()

    ep = episode_service.get_episode(episode_id, db)
    if not ep:
        await websocket.close(code=1008)
        return

    # Completed episode: replay stored steps then close
    if ep.status == "completed":
        steps = episode_service.get_episode_steps(episode_id, db)
        for step in steps:
            await websocket.send_json({
                "type": "step",
                "step_index": step.step_index,
                "action": _json.loads(step.action),
                "reward": step.reward,
                "diff": _json.loads(step.diff),
                "verifier_results": _json.loads(step.verifier_results),
                "events": _json.loads(step.events),
                "terminated": step.terminated,
            })
        await websocket.send_json({
            "type": "complete",
            "total_reward": ep.total_reward,
            "passed": ep.passed,
            "total_steps": ep.total_steps,
        })
        await websocket.close()
        return

    # Running episode: drain from queue
    queue = runner_service.episode_queues.get(episode_id)
    if queue is None:
        await websocket.close(code=1011)
        return

    try:
        while True:
            # Poll queue with short timeout to allow receiving fork messages
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                # Check for client messages (fork)
                try:
                    msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.0)
                    if msg.get("type") == "fork":
                        step_n = msg["step"]
                        old_task = runner_service.episode_tasks.get(episode_id)
                        if old_task and not old_task.done():
                            old_task.cancel()
                        actions = ReplayService().branch_from(episode_id, step_n, db)
                        new_ep = ep
                        new_episode_id = await runner_service.start_episode(
                            env_name=new_ep.env_name,
                            task_name=new_ep.task_name,
                            seed=new_ep.seed,
                            agent_id=new_ep.agent_id,
                        )
                        queue = runner_service.episode_queues[new_episode_id]
                        episode_id = new_episode_id
                except (asyncio.TimeoutError, Exception):
                    pass
                continue

            await websocket.send_json(event)
            if event.get("type") in ("complete", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        runner_service.episode_queues.pop(episode_id, None)
        await websocket.close()
```

**Note on imports:** The full import block at the top of `backend/app/api/episodes.py` should be:

```python
from __future__ import annotations
import asyncio
import json as _json
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import episode_service, runner_service
from forge.runtime.replay import ReplayService
```

- [ ] **Step 4: Run WebSocket tests**

```bash
pytest tests/backend/test_episode_api.py::test_websocket_stream_replays_completed_episode tests/backend/test_episode_api.py::test_websocket_stream_running_episode -v
```

Expected: 2 passed.

- [ ] **Step 5: Run full test suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/episodes.py tests/backend/test_episode_api.py
git commit -m "feat: add WebSocket stream endpoint for live episode events"
```

---

## Task 10: RewardBreakdown + EpisodeTimeline Components

**Files:**
- Create: `frontend/components/RewardBreakdown.tsx`
- Create: `frontend/components/EpisodeTimeline.tsx`

These are pure UI components — no server/network calls. `RewardBreakdown` renders a named-component reward table. `EpisodeTimeline` renders a scrollable step list with selection and a "Branch from here" button.

Before starting, read the Next.js 16 docs to confirm any relevant API changes:
```bash
ls frontend/node_modules/next/dist/docs/ 2>/dev/null | head -20
```

- [ ] **Step 1: Create RewardBreakdown component**

Create `frontend/components/RewardBreakdown.tsx`:

```tsx
"use client";

interface RewardComponent {
  name: string;
  value: number;
}

interface RewardBreakdownProps {
  components: RewardComponent[];
  total: number;
}

export default function RewardBreakdown({ components, total }: RewardBreakdownProps) {
  return (
    <div className="space-y-1">
      {components.map((c) => (
        <div key={c.name} className="flex justify-between text-sm font-mono">
          <span className="text-muted-foreground">{c.name}:</span>
          <span className={c.value >= 0 ? "text-green-400" : "text-red-400"}>
            {c.value >= 0 ? "+" : ""}
            {c.value.toFixed(3)}
          </span>
        </div>
      ))}
      <div className="border-t pt-1 flex justify-between text-sm font-mono font-semibold">
        <span className="text-muted-foreground">total:</span>
        <span className={total >= 0 ? "text-green-400" : "text-red-400"}>
          {total >= 0 ? "+" : ""}
          {total.toFixed(3)}
        </span>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create EpisodeTimeline component**

Create `frontend/components/EpisodeTimeline.tsx`:

```tsx
"use client";

interface StepSummary {
  step_index: number;
  action: string;       // JSON string
  reward: number;
  terminated: boolean;
  truncated: boolean;
}

interface EpisodeTimelineProps {
  steps: StepSummary[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  onBranch: (index: number) => void;
}

export default function EpisodeTimeline({
  steps,
  selectedIndex,
  onSelect,
  onBranch,
}: EpisodeTimelineProps) {
  return (
    <div className="flex flex-col gap-1 h-full overflow-auto">
      <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
        Steps
      </div>
      {steps.map((step) => {
        const action = (() => {
          try {
            return JSON.parse(step.action);
          } catch {
            return { type: "?" };
          }
        })();
        const isSelected = step.step_index === selectedIndex;
        const hasFailed = step.terminated || step.truncated;

        return (
          <button
            key={step.step_index}
            onClick={() => onSelect(step.step_index)}
            className={`text-left rounded px-2 py-1.5 text-xs transition-colors ${
              isSelected
                ? "bg-blue-950 border border-blue-400"
                : "bg-muted hover:bg-muted/80"
            }`}
          >
            <div className={isSelected ? "text-blue-400" : hasFailed ? "text-red-400" : "text-muted-foreground"}>
              Step {step.step_index}
            </div>
            <div className="text-foreground/70 truncate">{action.type ?? "—"}</div>
          </button>
        );
      })}
      {selectedIndex >= 0 && (
        <button
          onClick={() => onBranch(selectedIndex)}
          className="mt-2 w-full bg-blue-700 hover:bg-blue-600 text-white rounded px-2 py-1 text-xs font-medium"
        >
          ⑂ Branch from here
        </button>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Verify TypeScript compiles cleanly**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -20
```

Expected: no errors (or only pre-existing errors unrelated to the new files).

- [ ] **Step 4: Commit**

```bash
git add frontend/components/RewardBreakdown.tsx frontend/components/EpisodeTimeline.tsx
git commit -m "feat: add RewardBreakdown and EpisodeTimeline components"
```

---

## Task 11: Episode Replay Page

**Files:**
- Create: `frontend/app/environments/[env_name]/replay/[episode_id]/page.tsx`

This is a client-side interactive page that fetches an episode, lets the user select a step, and shows action + state diff + reward breakdown.

**Before writing this file**, check the Next.js 16 docs for client components with dynamic routes:
```bash
cat frontend/node_modules/next/dist/docs/app-router.md 2>/dev/null | head -80 || echo "check docs manually"
```

Also check the existing config page at `frontend/app/environments/[env_name]/config/page.tsx` to match the `params: Promise<{...}>` pattern.

- [ ] **Step 1: Create the page**

Create `frontend/app/environments/[env_name]/replay/[episode_id]/page.tsx`:

```tsx
"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import EpisodeTimeline from "@/components/EpisodeTimeline";
import RewardBreakdown from "@/components/RewardBreakdown";

interface StepData {
  step_index: number;
  action: string;
  reward: number;
  verifier_results: string;
  diff: string;
  events: string;
  state_hash_before: string;
  state_hash_after: string;
  terminated: boolean;
  truncated: boolean;
}

interface EpisodeData {
  id: string;
  env_name: string;
  task_name: string;
  seed: number;
  status: string;
  total_reward: number;
  passed: boolean;
  steps: StepData[];
}

export default function EpisodeReplayPage() {
  const params = useParams();
  const envName = params.env_name as string;
  const episodeId = params.episode_id as string;

  const [episode, setEpisode] = useState<EpisodeData | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`http://localhost:8000/api/episodes/${episodeId}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setEpisode)
      .catch((e) => setError(e.message));
  }, [episodeId]);

  function handleBranch(stepIndex: number) {
    window.location.href = `/environments/${envName}/graph?episode_id=${episodeId}&fork_step=${stepIndex}`;
  }

  if (error) {
    return (
      <div className="text-center py-20 text-destructive">
        Failed to load episode: {error}
      </div>
    );
  }

  if (!episode) {
    return <div className="text-center py-20 text-muted-foreground">Loading...</div>;
  }

  const selectedStep = episode.steps[selectedIndex];
  const action = selectedStep ? (() => { try { return JSON.parse(selectedStep.action); } catch { return {}; } })() : null;
  const diff = selectedStep ? (() => { try { return JSON.parse(selectedStep.diff); } catch { return {}; } })() : null;
  const vrResults = selectedStep ? (() => { try { return JSON.parse(selectedStep.verifier_results); } catch { return []; } })() : [];

  const rewardComponents = vrResults.flatMap((vr: { checks?: Array<{name: string; score: number}> }) =>
    (vr.checks ?? []).map((c: { name: string; score: number }) => ({ name: c.name, value: c.score }))
  );

  return (
    <div>
      <div className="mb-4">
        <h1 className="text-xl font-bold">Episode Replay</h1>
        <p className="text-sm text-muted-foreground">
          {episode.id} · {episode.task_name} · seed {episode.seed}
          {" · "}
          <span className={episode.passed ? "text-green-400" : "text-red-400"}>
            {episode.passed ? "✓ passed" : "✗ failed"}
          </span>
        </p>
      </div>

      <div className="grid grid-cols-[200px_1fr] gap-4 h-[calc(100vh-200px)]">
        {/* Left: Timeline */}
        <EpisodeTimeline
          steps={episode.steps}
          selectedIndex={selectedIndex}
          onSelect={setSelectedIndex}
          onBranch={handleBranch}
        />

        {/* Right: Step detail */}
        <div className="flex flex-col gap-3 overflow-auto">
          {selectedStep && (
            <>
              <div className="rounded-md border bg-card p-3">
                <div className="text-xs font-semibold text-muted-foreground uppercase mb-2">
                  Action
                </div>
                <pre className="text-xs font-mono text-yellow-400 whitespace-pre-wrap">
                  {JSON.stringify(action, null, 2)}
                </pre>
              </div>

              <div className="rounded-md border bg-card p-3">
                <div className="text-xs font-semibold text-muted-foreground uppercase mb-2">
                  State Diff
                </div>
                <div className="font-mono text-xs space-y-0.5">
                  {Object.entries(diff?.added ?? {}).map(([k, v]) => (
                    <div key={k} className="text-green-400">
                      + {k}: {JSON.stringify(v)}
                    </div>
                  ))}
                  {Object.entries(diff?.changed ?? {}).map(([k, v]) => {
                    const cv = v as { before: unknown; after: unknown };
                    return (
                      <div key={k} className="text-red-400">
                        ~ {k}: {JSON.stringify(cv.before)} → {JSON.stringify(cv.after)}
                      </div>
                    );
                  })}
                  {Object.entries(diff?.removed ?? {}).map(([k, v]) => (
                    <div key={k} className="text-red-400">
                      - {k}: {JSON.stringify(v)}
                    </div>
                  ))}
                  {Object.keys({...(diff?.added ?? {}), ...(diff?.changed ?? {}), ...(diff?.removed ?? {})}).length === 0 && (
                    <span className="text-muted-foreground">no state changes</span>
                  )}
                </div>
              </div>

              <div className="rounded-md border bg-card p-3">
                <div className="text-xs font-semibold text-muted-foreground uppercase mb-2">
                  Reward Breakdown — r = {selectedStep.reward.toFixed(3)}
                </div>
                <RewardBreakdown components={rewardComponents} total={selectedStep.reward} />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles cleanly**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -30
```

Expected: no new errors.

- [ ] **Step 3: Commit**

```bash
git add "frontend/app/environments/[env_name]/replay/"
git commit -m "feat: add Episode Replay page"
```

---

## Task 12: Dashboard Page

**Files:**
- Create: `frontend/app/dashboard/page.tsx`

The Dashboard fetches `GET /api/envs/{env_name}/stats` and renders stat cards, a failure mode bar chart, and a recent episodes list. The env_name comes from a URL query parameter.

- [ ] **Step 1: Create the dashboard page**

Create `frontend/app/dashboard/page.tsx`:

```tsx
import Link from "next/link";

interface FailureCluster {
  check_name: string;
  count: number;
  episode_ids: string[];
}

interface Stats {
  pass_rate: number;
  avg_reward: number;
  avg_steps: number;
  policy_violation_count: number;
  top_failures: FailureCluster[];
}

interface EpisodeSummary {
  id: string;
  env_name: string;
  task_name: string;
  status: string;
  passed: boolean;
  total_reward: number;
  total_steps: number;
  started_at: string | null;
}

async function getStats(envName: string): Promise<Stats | null> {
  try {
    const res = await fetch(`http://localhost:8000/api/envs/${envName}/stats`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function getRecentEpisodes(envName: string): Promise<EpisodeSummary[]> {
  try {
    const res = await fetch(
      `http://localhost:8000/api/episodes/?env_name=${envName}`,
      { cache: "no-store" }
    );
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export default async function DashboardPage({
  searchParams,
}: {
  searchParams: Promise<{ env_name?: string }>;
}) {
  const { env_name } = await searchParams;
  const envName = env_name ?? "";

  if (!envName) {
    return (
      <div className="text-center py-20">
        <p className="text-muted-foreground">
          Provide <code className="text-xs bg-muted px-1 rounded">?env_name=your_env</code> in the URL.
        </p>
      </div>
    );
  }

  const [stats, episodes] = await Promise.all([
    getStats(envName),
    getRecentEpisodes(envName),
  ]);

  const maxFailureCount = Math.max(1, ...(stats?.top_failures.map((f) => f.count) ?? [1]));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <p className="text-sm text-muted-foreground mt-1">{envName}</p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-4 gap-4">
        <div className="rounded-lg border bg-card p-4">
          <div className="text-xs uppercase text-muted-foreground tracking-wide">Pass Rate</div>
          <div className={`text-3xl font-bold mt-1 ${(stats?.pass_rate ?? 0) >= 0.7 ? "text-green-400" : "text-foreground"}`}>
            {stats ? `${Math.round(stats.pass_rate * 100)}%` : "—"}
          </div>
          <div className="text-xs text-muted-foreground mt-1">last 100 episodes</div>
        </div>

        <div className="rounded-lg border bg-card p-4">
          <div className="text-xs uppercase text-muted-foreground tracking-wide">Avg Reward</div>
          <div className="text-3xl font-bold mt-1">
            {stats ? stats.avg_reward.toFixed(2) : "—"}
          </div>
          <div className="text-xs text-muted-foreground mt-1">last 100 episodes</div>
        </div>

        <div className="rounded-lg border bg-card p-4">
          <div className="text-xs uppercase text-muted-foreground tracking-wide">Avg Steps</div>
          <div className="text-3xl font-bold mt-1">
            {stats ? stats.avg_steps.toFixed(1) : "—"}
          </div>
          <div className="text-xs text-muted-foreground mt-1">last 100 episodes</div>
        </div>

        <div className="rounded-lg border bg-card p-4">
          <div className="text-xs uppercase text-muted-foreground tracking-wide">Policy Violations</div>
          <div className={`text-3xl font-bold mt-1 ${(stats?.policy_violation_count ?? 0) > 0 ? "text-red-400" : "text-foreground"}`}>
            {stats ? stats.policy_violation_count : "—"}
          </div>
          <div className="text-xs text-muted-foreground mt-1">last 100 episodes</div>
        </div>
      </div>

      {/* Bottom panels */}
      <div className="grid grid-cols-2 gap-4">
        {/* Top Failure Modes */}
        <div className="rounded-lg border bg-card p-4">
          <div className="text-sm font-semibold mb-3">Top Failure Modes</div>
          {stats && stats.top_failures.length > 0 ? (
            <div className="space-y-2">
              {stats.top_failures.map((f) => (
                <div key={f.check_name} className="flex items-center justify-between gap-3">
                  <span className="text-xs text-foreground truncate flex-1">{f.check_name}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-24 h-1.5 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-red-400 rounded-full"
                        style={{ width: `${(f.count / maxFailureCount) * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-muted-foreground w-8 text-right">{f.count}×</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">No failures recorded.</p>
          )}
        </div>

        {/* Recent Episodes */}
        <div className="rounded-lg border bg-card p-4">
          <div className="text-sm font-semibold mb-3">Recent Episodes</div>
          {episodes.length > 0 ? (
            <div className="space-y-1.5">
              {episodes.slice(0, 8).map((ep) => (
                <Link
                  key={ep.id}
                  href={`/environments/${envName}/replay/${ep.id}`}
                  className="flex items-center justify-between text-xs hover:bg-muted/50 rounded px-1 py-0.5 transition-colors"
                >
                  <span className="text-muted-foreground font-mono">{ep.id}</span>
                  <span className={ep.passed ? "text-green-400" : "text-red-400"}>
                    {ep.passed ? "✓ passed" : "✗ failed"}
                  </span>
                  <span className="text-muted-foreground">r={ep.total_reward.toFixed(2)}</span>
                  <span className="text-muted-foreground">{ep.total_steps} steps</span>
                </Link>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">No episodes yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles cleanly**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -30
```

- [ ] **Step 3: Commit**

```bash
git add frontend/app/dashboard/
git commit -m "feat: add Dashboard page with stat cards, failure modes, recent episodes"
```

---

## Task 13: EnvironmentGraph Component + Graph Page

**Files:**
- Modify: `frontend/package.json` (add `@xyflow/react`)
- Create: `frontend/components/EnvironmentGraph.tsx`
- Create: `frontend/app/environments/[env_name]/graph/page.tsx`

Before writing code, read the `@xyflow/react` docs for the correct import paths and API in the version that ships with the installed package. Check whether it's already in the lockfile:
```bash
grep xyflow frontend/package-lock.json | head -5
```
If not present, install it first.

- [ ] **Step 1: Install @xyflow/react**

```bash
cd frontend && npm install @xyflow/react
```

Verify the package appears in `package.json` dependencies after installation.

- [ ] **Step 2: Create EnvironmentGraph component**

Check how to import React Flow in the installed version:
```bash
ls frontend/node_modules/@xyflow/react/dist/ 2>/dev/null | head -10
```

Create `frontend/components/EnvironmentGraph.tsx`:

```tsx
"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import ReactFlow, {
  Background,
  Controls,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

interface CompilerInput {
  project_name: string;
  domain: string;
  entities: Array<{ name: string; fields: Array<{ name: string }> }>;
  actions: Array<{ name: string }>;
  tasks: Array<{ name: string; success_conditions: unknown[] }>;
  policies: Array<{ name: string }>;
}

interface StepEvent {
  type: "step" | "complete" | "error";
  step_index?: number;
  action?: { type: string };
  diff?: { changed?: Record<string, { before: unknown; after: unknown }> };
  verifier_results?: Array<{ verifier_id: string; passed: boolean; checks: Array<{ name: string; passed: boolean }> }>;
  events?: Array<{ type: string; [key: string]: unknown }>;
  total_reward?: number;
  passed?: boolean;
  total_steps?: number;
}

function buildInitialGraph(ci: CompilerInput): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  // Entity nodes (left column)
  ci.entities.forEach((entity, i) => {
    nodes.push({
      id: `entity_${entity.name}`,
      type: "default",
      position: { x: 50, y: i * 120 + 50 },
      data: {
        label: entity.name,
        fieldValues: {} as Record<string, unknown>,
      },
      style: {
        border: "1px solid #38bdf8",
        background: "#0f2744",
        color: "#f8fafc",
        borderRadius: 8,
        padding: "8px 12px",
        minWidth: 120,
      },
    });
  });

  // Action nodes (center column)
  ci.actions.forEach((action, i) => {
    nodes.push({
      id: `action_${action.name}`,
      type: "default",
      position: { x: 280, y: i * 80 + 50 },
      data: { label: action.name, active: false },
      style: {
        border: "1px solid #334155",
        background: "#1e293b",
        color: "#94a3b8",
        borderRadius: 6,
        padding: "6px 10px",
        minWidth: 120,
      },
    });
  });

  // Task nodes (right column, top)
  ci.tasks.forEach((task, i) => {
    nodes.push({
      id: `task_${task.name}`,
      type: "default",
      position: { x: 500, y: i * 100 + 50 },
      data: {
        label: task.name,
        checksPassed: 0,
        checksTotal: (task.success_conditions as unknown[]).length,
      },
      style: {
        border: "1px solid #34d399",
        background: "#0d2318",
        color: "#34d399",
        borderRadius: 8,
        padding: "8px 12px",
        minWidth: 140,
      },
    });
  });

  // Policy nodes (right column, below tasks)
  ci.policies.forEach((policy, i) => {
    nodes.push({
      id: `policy_${policy.name}`,
      type: "default",
      position: { x: 500, y: ci.tasks.length * 100 + i * 80 + 50 },
      data: { label: policy.name, violated: false },
      style: {
        border: "1px solid #334155",
        background: "#1e293b",
        color: "#94a3b8",
        borderRadius: 8,
        padding: "8px 12px",
        minWidth: 140,
      },
    });
  });

  // Edges: entity → action (all-to-all for simplicity)
  ci.entities.forEach((entity) => {
    ci.actions.forEach((action) => {
      edges.push({
        id: `e_${entity.name}_${action.name}`,
        source: `entity_${entity.name}`,
        target: `action_${action.name}`,
        style: { stroke: "#334155" },
      });
    });
  });

  // Edges: action → task
  ci.actions.forEach((action) => {
    ci.tasks.forEach((task) => {
      edges.push({
        id: `e_${action.name}_${task.name}`,
        source: `action_${action.name}`,
        target: `task_${task.name}`,
        style: { stroke: "#334155" },
      });
    });
  });

  return { nodes, edges };
}

interface EnvironmentGraphProps {
  envName: string;
  episodeId?: string;
  compilerInput: CompilerInput | null;
}

export default function EnvironmentGraph({
  envName,
  episodeId,
  compilerInput,
}: EnvironmentGraphProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [stepLabel, setStepLabel] = useState<string>("");
  const wsRef = useRef<WebSocket | null>(null);

  // Build initial graph from compiler input
  useEffect(() => {
    if (!compilerInput) return;
    const { nodes: n, edges: e } = buildInitialGraph(compilerInput);
    setNodes(n);
    setEdges(e);
  }, [compilerInput, setNodes, setEdges]);

  // WebSocket connection for live updates
  useEffect(() => {
    if (!episodeId) return;
    const ws = new WebSocket(`ws://localhost:8000/api/episodes/${episodeId}/stream`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data: StepEvent = JSON.parse(event.data);

      if (data.type === "step") {
        const stepIndex = data.step_index ?? 0;
        const actionType = data.action?.type ?? "";
        setStepLabel(`step ${stepIndex}`);

        // Highlight fired action node (amber for 1.5s)
        setNodes((nds) =>
          nds.map((n) => {
            if (n.id === `action_${actionType}`) {
              return {
                ...n,
                data: { ...n.data, active: true },
                style: {
                  ...n.style,
                  border: "2px solid #f59e0b",
                  background: "#1c1400",
                  color: "#fbbf24",
                },
              };
            }
            // Reset other action nodes
            if (n.id.startsWith("action_") && n.id !== `action_${actionType}`) {
              return {
                ...n,
                data: { ...n.data, active: false },
                style: {
                  border: "1px solid #334155",
                  background: "#1e293b",
                  color: "#94a3b8",
                  borderRadius: 6,
                  padding: "6px 10px",
                  minWidth: 120,
                },
              };
            }
            return n;
          })
        );

        // Update entity field values from diff
        const changed = data.diff?.changed ?? {};
        setNodes((nds) =>
          nds.map((n) => {
            if (!n.id.startsWith("entity_")) return n;
            const entityName = n.id.replace("entity_", "");
            const updates: Record<string, unknown> = { ...(n.data.fieldValues as Record<string, unknown>) };
            Object.entries(changed).forEach(([key, val]) => {
              if (key.startsWith(`${entityName}.`)) {
                const field = key.split(".").slice(2).join(".");
                updates[field] = (val as { after: unknown }).after;
              }
            });
            return { ...n, data: { ...n.data, fieldValues: updates } };
          })
        );

        // Check for policy violations
        const events = data.events ?? [];
        const violatedPolicies = events
          .filter((e) => e.type === "policy_violation")
          .map((e) => e.policy_id as string);
        if (violatedPolicies.length > 0) {
          setNodes((nds) =>
            nds.map((n) => {
              if (!n.id.startsWith("policy_")) return n;
              const policyId = n.id.replace("policy_", "");
              if (violatedPolicies.includes(policyId)) {
                return {
                  ...n,
                  data: { ...n.data, violated: true },
                  style: {
                    border: "1px solid #f87171",
                    background: "#1f0d0d",
                    color: "#f87171",
                    borderRadius: 8,
                    padding: "8px 12px",
                    minWidth: 140,
                  },
                };
              }
              return n;
            })
          );
        }

        // Update task check counters
        const vrResults = data.verifier_results ?? [];
        setTimeout(() => {
          // Reset active action highlight after 1.5s
          setNodes((nds) =>
            nds.map((n) => {
              if (n.id === `action_${actionType}`) {
                return {
                  ...n,
                  data: { ...n.data, active: false },
                  style: {
                    border: "1px solid #334155",
                    background: "#1e293b",
                    color: "#94a3b8",
                    borderRadius: 6,
                    padding: "6px 10px",
                    minWidth: 120,
                  },
                };
              }
              return n;
            })
          );
        }, 1500);
      }

      if (data.type === "complete") {
        setStepLabel(`done — ${data.total_steps} steps, r=${data.total_reward?.toFixed(2)}`);
        ws.close();
      }
    };

    return () => {
      ws.close();
    };
  }, [episodeId, setNodes]);

  if (!compilerInput) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
        No compiler input available for this environment.
      </div>
    );
  }

  return (
    <div className="relative w-full h-[600px] rounded-lg border bg-card overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
      >
        <Background color="#1e293b" />
        <Controls />
      </ReactFlow>
      {stepLabel && (
        <div className="absolute bottom-2 right-2 text-xs text-muted-foreground bg-background/80 px-2 py-1 rounded">
          React Flow · {episodeId ? `live via WebSocket · ${stepLabel}` : "static"}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Create the graph page**

Create `frontend/app/environments/[env_name]/graph/page.tsx`:

```tsx
import EnvironmentGraph from "@/components/EnvironmentGraph";

async function getCompilerInput(envName: string) {
  try {
    const res = await fetch(
      `http://localhost:8000/api/envs/${envName}/compiler-input`,
      { cache: "no-store" }
    );
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export default async function GraphPage({
  params,
  searchParams,
}: {
  params: Promise<{ env_name: string }>;
  searchParams: Promise<{ episode_id?: string }>;
}) {
  const { env_name } = await params;
  const { episode_id } = await searchParams;

  const compilerInput = await getCompilerInput(env_name);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold">Environment Graph</h1>
        <p className="text-sm text-muted-foreground">
          {env_name}
          {episode_id && ` · ${episode_id} (live)`}
        </p>
      </div>
      <EnvironmentGraph
        envName={env_name}
        episodeId={episode_id}
        compilerInput={compilerInput}
      />
    </div>
  );
}
```

- [ ] **Step 4: Verify TypeScript compiles cleanly**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -30
```

Expected: no new errors (React Flow types should be available).

- [ ] **Step 5: Run full Python test suite**

```bash
cd .. && pytest --tb=short -q
```

Expected: all 200+ tests pass.

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/components/EnvironmentGraph.tsx "frontend/app/environments/[env_name]/graph/"
git commit -m "feat: add EnvironmentGraph component and graph page with React Flow + WebSocket"
```

---

## Final Verification

After all 13 tasks are complete:

```bash
# Run full Python test suite
pytest --tb=short -q

# Check TypeScript
cd frontend && npx tsc --noEmit

# Quick import check of new modules
python -c "
from forge.runtime.telemetry import TelemetryClient
from forge.runtime.policy import RandomPolicy
from forge.runtime.replay import ReplayService, EpisodeRecord
from forge.runtime.clustering import FailureClusterer, FailureCluster
from backend.app.services.episode_service import get_stats
from backend.app.services.runner_service import start_episode, episode_queues
print('All imports OK')
"
```
