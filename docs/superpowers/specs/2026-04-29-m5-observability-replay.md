# M5: Observability & Replay Design

**Goal:** Add episode trace storage, branch replay, failure clustering, a live WebSocket episode runner, and three frontend pages (Dashboard, Episode Replay, Environment Graph).

**Scope:** Backend Python + FastAPI + SQLAlchemy + frontend Next.js. No new infrastructure — uses existing SQLite DB and asyncio.

---

## 1. Architecture Overview

M5 adds a `TelemetryClient` that writes structured step spans to SQLite + JSONL after each `ForgeEnv.step()`. A `ReplayService` loads stored episodes and produces action sequences for fast-forward branching. A `FailureClusterer` groups episodes by the first failed `CheckResult`. A FastAPI WebSocket endpoint spawns an asyncio background task that runs `ForgeEnv` and streams step events to connected clients in real time.

**Key decisions:**
- **Storage:** SQLite primary (`Episode` + `EpisodeStep` tables) + JSONL on disk as backup
- **Telemetry:** Lightweight custom `TelemetryClient` — OTel concepts (episode trace, step spans) without the OTel SDK
- **Branch replay:** Fast-forward — stored action sequence replayed from step 0 to N-1, then diverges. No state snapshots needed.
- **Real-time channel:** WebSocket (bidirectional). Client receives step events; can send `{"type":"fork","step":N}` to branch.
- **Episode runner:** FastAPI asyncio background task. `ForgeEnv` runs inside the backend process. Replaced by Celery workers in M6.

---

## 2. File Map

```
forge/
  runtime/
    telemetry.py                  CREATE — TelemetryClient (SQLite + JSONL writes)
    replay.py                     CREATE — ReplayService (load episode, branch_from)
    clustering.py                 CREATE — FailureClusterer (group by first failed check)
    env.py                        MODIFY — inject TelemetryClient, call record_step + complete_episode
backend/
  app/
    models.py                     MODIFY — add Episode + EpisodeStep SQLAlchemy models
    api/
      episodes.py                 CREATE — REST endpoints + WebSocket endpoint
    services/
      episode_service.py          CREATE — CRUD, stats aggregation, failure clustering
      runner_service.py           CREATE — asyncio background task that runs ForgeEnv
    main.py                       MODIFY — include episodes router
    api/
      envs.py                     MODIFY — add /api/envs/{env_name}/stats and /api/envs/{env_name}/compiler-input endpoints
forge/
  runtime/
    policy.py                     CREATE — RandomPolicy stub (acts randomly from env action space)
frontend/
  app/
    dashboard/
      page.tsx                    CREATE — Dashboard page
    environments/
      [env_name]/
        replay/
          [episode_id]/
            page.tsx              CREATE — Episode Replay page
        graph/
          page.tsx                CREATE — Environment Graph (live WebSocket)
  components/
    EpisodeTimeline.tsx           CREATE — step-by-step action timeline with branch button
    EnvironmentGraph.tsx          CREATE — React Flow live graph, WebSocket-fed
    RewardBreakdown.tsx           CREATE — reward component breakdown display
tests/
  runtime/
    test_telemetry.py             CREATE
    test_replay.py                CREATE
    test_clustering.py            CREATE
  backend/
    test_episode_api.py           CREATE
```

---

## 3. Episode Storage

### 3.1 SQLAlchemy Models (`backend/app/models.py`)

```python
class Episode(Base):
    __tablename__ = "episodes"
    id: Mapped[str]           # episode_id, e.g. "ep_abc123"
    env_name: Mapped[str]
    task_name: Mapped[str]
    seed: Mapped[int]
    agent_id: Mapped[str]     # e.g. "random_policy", "claude-sonnet-4-6"
    status: Mapped[str]       # "running" | "completed" | "failed"
    total_steps: Mapped[int]
    total_reward: Mapped[float]
    passed: Mapped[bool]
    started_at: Mapped[datetime]
    completed_at: Mapped[datetime | None]
    jsonl_path: Mapped[str | None]  # absolute path to backup JSONL file

class EpisodeStep(Base):
    __tablename__ = "episode_steps"
    id: Mapped[int]           # auto-increment
    episode_id: Mapped[str]   # FK → episodes.id
    step_index: Mapped[int]
    action: Mapped[str]       # JSON
    reward: Mapped[float]
    verifier_results: Mapped[str]   # JSON
    diff: Mapped[str]               # JSON
    events: Mapped[str]             # JSON
    state_hash_before: Mapped[str]
    state_hash_after: Mapped[str]
    terminated: Mapped[bool]
    truncated: Mapped[bool]
```

### 3.2 JSONL Backup

Written to `generated_envs/<env_name>/episodes/<episode_id>.jsonl`. Each line is a `StepSnapshot.model_dump_json()`. Created by `TelemetryClient` in parallel with SQLite writes.

---

## 4. TelemetryClient

**`forge/runtime/telemetry.py`**

```python
class TelemetryClient:
    def __init__(
        self,
        episode_id: str,
        env_name: str,
        task_name: str,
        seed: int,
        agent_id: str,
        db_session,              # SQLAlchemy Session
        jsonl_path: Path | None = None,
    ) -> None: ...

    def record_step(self, snapshot: StepSnapshot) -> None:
        # Writes EpisodeStep row to SQLite
        # Appends snapshot.model_dump_json() + "\n" to JSONL file

    def complete_episode(self, total_reward: float, passed: bool) -> None:
        # Updates Episode row: status="completed", total_reward, passed, completed_at
```

**`ForgeEnv` changes (`forge/runtime/env.py`):**
- `__init__` accepts optional `telemetry: TelemetryClient | None = None`
- `step()` calls `self._telemetry.record_step(snapshot)` after building the step snapshot
- Episode completion (terminated/truncated) calls `self._telemetry.complete_episode(...)`

---

## 5. ReplayService

**`forge/runtime/replay.py`**

```python
@dataclass
class EpisodeRecord:
    episode: Episode
    steps: list[EpisodeStep]   # ordered by step_index

class ReplayService:
    def load_episode(self, episode_id: str, db) -> EpisodeRecord:
        # Fetches Episode + all EpisodeSteps ordered by step_index

    def branch_from(self, episode_id: str, step_n: int, db) -> list[dict]:
        # Returns [step.action (parsed JSON) for step in steps if step.step_index < step_n]
        # Caller fast-forwards a fresh ForgeEnv using these actions then diverges
```

---

## 6. FailureClusterer

**`forge/runtime/clustering.py`**

```python
@dataclass
class FailureCluster:
    check_name: str            # name of first failed CheckResult
    count: int
    episode_ids: list[str]     # up to 5 representative IDs

class FailureClusterer:
    def cluster(self, episodes: list[EpisodeRecord]) -> list[FailureCluster]:
        # For each failed episode, find the first EpisodeStep where
        # any verifier_result check has passed=False
        # Group by that check's name, sort by count descending, return top 5
```

---

## 7. Backend API

**`backend/app/api/episodes.py`** — `router = APIRouter(prefix="/api/episodes")`

| Method | Path | Description |
|---|---|---|
| `POST` | `/` | Start episode — creates `Episode` row, spawns background task, returns `{"episode_id": "..."}` |
| `GET` | `/{episode_id}` | Full episode with all steps |
| `GET` | `/?env_name=X` | List episodes for env, ordered by `started_at` desc |
| `GET` | `/{episode_id}/steps/{step_n}/branch` | Returns action sequence for steps 0..N-1 |
| `WS` | `/{episode_id}/stream` | WebSocket — pushes `StepEvent` JSON per step; accepts `{"type":"fork","step":N}` |

**`GET /api/envs/{env_name}/stats`** (added to existing `envs.py`):
Returns `{pass_rate, avg_reward, avg_steps, policy_violation_count, top_failures: FailureCluster[]}` computed over last 100 completed episodes for `env_name`.

### 7.1 WebSocket Protocol

**Server → Client** (after each step):
```json
{
  "type": "step",
  "step_index": 3,
  "action": {"type": "offer_refund", "ticket_id": "t_42"},
  "reward": -0.39,
  "diff": {"changed": {"ticket.status": {"before": "open", "after": "pending"}}},
  "verifier_results": [...],
  "events": [...],
  "terminated": false
}
```

**Server → Client** (on completion):
```json
{"type": "complete", "total_reward": 0.31, "passed": false, "total_steps": 22}
```

**Client → Server** (to fork):
```json
{"type": "fork", "step": 5}
```
Server stops current episode, loads action sequence to step 5, fast-forwards a new `ForgeEnv`, and begins streaming from step 5.

**Connecting to a completed episode:** If a client connects to `WS /{episode_id}/stream` for an episode whose status is already `"completed"` or `"failed"`, the server immediately replays all stored steps as `step` events in order, then sends the `complete` event, then closes the connection.

### 7.2 RunnerService (`backend/app/services/runner_service.py`)

```python
async def start_episode(
    env_name: str,
    task_name: str,
    seed: int,
    agent_id: str,
    websocket_queues: dict[str, asyncio.Queue],  # episode_id → queue
    db,
) -> str:  # returns episode_id
```

- Creates `Episode` row, `TelemetryClient`, builds `ForgeEnv`
- Runs `env.reset()` then loops `env.step(action)` until terminated/truncated
- After each step: puts step event onto `websocket_queues[episode_id]`
- Uses a `RandomPolicy` (or scripted policy) as default agent; M6 adds LLM adapters

---

## 8. Frontend Pages

### 8.1 Dashboard (`/dashboard`)

Fetches `/api/envs/{env_name}/stats`. Renders:
- 4 stat cards: pass rate (green if ≥ 70%), avg reward, avg steps, policy violations (red if > 0)
- **Top Failure Modes**: horizontal bar chart (inline CSS bars) of top-5 `FailureCluster` by count
- **Recent Episodes**: table with episode ID, pass/fail badge, reward, steps — each row links to replay page

### 8.2 Episode Replay (`/environments/[env_name]/replay/[episode_id]`)

Fetches `GET /api/episodes/{episode_id}`. Renders:
- **Left panel — Step Timeline**: scrollable list of steps. Each shows step index, action type, pass/fail icon. Selected step is highlighted. "⑂ Branch from here" button calls `GET .../steps/{n}/branch` then opens the Environment Graph page with the forked episode.
- **Right panel — Step Detail**: three sub-panels for the selected step:
  - **Action**: action type + params as formatted JSON
  - **State Diff**: added keys in green, changed/removed in red, flat key-path format
  - **Reward Breakdown**: named components with values (positive green, negative red), total reward

### 8.3 Environment Graph (`/environments/[env_name]/graph`)

Connects to `WS /api/episodes/{episode_id}/stream` when an `episode_id` query param is present; otherwise shows the static compiled graph.

React Flow nodes:
- **Entity nodes** (blue border) — show entity name + live field values from latest step diff
- **Action nodes** (grey) — one per action; highlighted amber + ⚡ when that action just fired
- **Task node** (green border) — shows task name + `N/M checks ✓` counter
- **Policy nodes** (red border on violation) — show policy id; turns red with ⚠ when violated

On WebSocket `step` event: updates entity node field values from `diff`, highlights the fired action node for 1.5s, updates task check counter from `verifier_results`, flags any policy violation.

**`EnvironmentGraph.tsx`** — React Flow graph component. Derives initial node/edge layout from the `/api/envs/{env_name}/compiler-input` endpoint (added to `envs.py` — returns stored `CompilerInput` JSON). Updates node data on WebSocket events.

**`EpisodeTimeline.tsx`** — step list with selection state, branch button, pass/fail icons.

**`RewardBreakdown.tsx`** — renders 5 named reward components as a small table with colored values.

---

## 9. Testing Strategy

- `test_telemetry.py` — `record_step` writes correct SQLite row + JSONL line; `complete_episode` updates status; `ForgeEnv(telemetry=None)` is a no-op (existing tests unaffected)
- `test_replay.py` — `load_episode` returns steps in order; `branch_from(episode_id, 3)` returns exactly 3 actions
- `test_clustering.py` — 5 episodes with known first-failure check names → correct cluster counts and ordering
- `test_episode_api.py` — `POST /api/episodes` returns episode_id; `GET /api/episodes/{id}` returns full record; WebSocket receives step events; `/api/envs/{env}/stats` returns correct pass_rate
- All existing 200 tests must continue to pass (`TelemetryClient` defaults to `None` in `ForgeEnv`)

---

## 10. What Is Not in M5

- LLM agent adapters (M6 — RandomPolicy only in M5)
- Parallel rollouts / Celery workers (M6)
- PolicyEngine DSL enforcement (M7)
- RBAC observation filtering (M7)
- Preference pair / GRPO export (M6)
