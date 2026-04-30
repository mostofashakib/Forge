# Forge

**Convert enterprise workflow specs into reinforcement learning environments.**

Forge takes a natural-language description of a business process and produces a fully wired Gymnasium-compatible environment — complete with state management, action validation, reward computation, verifiers, agent adapters, and a web UI for replay and analysis.

---

## Features

### Runtime Kernel
- **Gymnasium-compatible `ForgeEnv`** — drop-in `reset()` / `step()` loop with full 5-tuple returns
- **Deterministic replay** — same seed + action sequence always produces an identical trajectory hash
- **StateStore + TrajectoryStore** — immutable state snapshots, per-episode trajectory recording
- **ActionValidator** — rejects unknown action types before they reach the transition layer
- **TransitionEngine** — register named transition functions; composable per environment
- **Clock** — logical time advancement per step for temporally-sensitive verifiers

### LLM Extraction & Compiler
- **Multi-step extraction pipeline** — entity extractor → action inferencer → task generator → policy parser → `CompilerInput`
- **Jinja2 compiler** — `CompilerInput` → runnable Python environment package
- **Package Builder** — writes generated env to `generated_envs/<name>/` with gym wrapper, state factory, and transition stubs
- **Validation runner** — smoke-tests generated environments immediately after compilation
- **Compiler Review UI** — inspect extracted entities, actions, tasks, and generated code before committing

### Customization Layer
- **Override hooks** — drop Python files into `custom/transitions.py` or `custom/verifiers.py` to replace generated behavior
- **Config-based reward & observation** — YAML config controls reward weights and observation field inclusion without recompiling
- **`forge` CLI** — `forge compile`, `forge run`, `forge replay`, `forge validate` commands
- **Override validation** — recompile checks that custom overrides still match the generated interface

### Verifier & Reward Engine
Six built-in verifier types:
| Verifier | Description |
|---|---|
| `ExactStateVerifier` | Checks specific state field values match expected |
| `EventVerifier` | Confirms required events appeared in the trajectory |
| `TemporalVerifier` | Validates event ordering and timing constraints |
| `NegativeVerifier` | Asserts forbidden events did not occur |
| `PolicyVerifier` | Evaluates Python expressions against state |
| `SemanticVerifier` | LLM-based semantic correctness check with embedding cache |

- **Decomposed `RewardBreakdown`** — per-component scores summed to a total; full breakdown returned in step info
- **`RewardEngine`** — register named reward functions; default fallback support

### Observability & Replay
- **`TelemetryClient`** — records every step snapshot and episode completion to SQLite
- **Episode replay** — re-run any recorded episode step-by-step from the stored trajectory
- **Branch replay** — fork from any step index, try alternate action sequences
- **Failure clustering** — groups failed episodes by trajectory diff similarity
- **Dashboard** — episode list with reward summaries and pass/fail status
- **Episode Replay UI** — step-through viewer with state diff and event log
- **Environment Graph** — visual entity/action relationship map

### Parallel Rollouts & Training Export
- **Celery workers** — parallel episode execution across configurable worker pool
- **Five agent adapters** — `random`, `scripted:<path>`, `anthropic:<model>`, `openai:<model>`, `vllm:<model>`
- **Rollout Launcher UI** — configure agent, task, episode count, and seed from the browser

Six export formats for training:
| Format | Contents |
|---|---|
| `trajectories.jsonl` | Full step-by-step trajectory for every episode |
| `rewards.jsonl` | Per-episode reward breakdown |
| `verifier_results.jsonl` | Per-step verifier pass/fail and scores |
| `sft_pairs.jsonl` | (state, action) pairs from passed episodes only |
| `preference_pairs.jsonl` | Best/worst episode pairs grouped by (task, seed bucket) |
| `grpo_rollouts.parquet` | GRPO-ready rollout table via Pandas |

### Security & Policy
- **PolicyEngine DSL** — Python expressions evaluated in a sandboxed context (`eval` with empty `__builtins__`, state-only scope); violations block transitions and return 0.0 reward
- **Network isolation** — AST-based static scanner blocks `requests`, `httpx`, `urllib`, `socket`, `aiohttp` imports in generated environments; bypassed by `FORGE_DEV_NETWORK=true`
- **PII redaction** — regex-based redactor strips emails, phone numbers, and SSNs from `CompilerInput` before code generation
- **RBAC observation filtering** — `ObservationFilter` removes or restricts state fields per role using `can_see` / `cannot_see` rules; applied transparently in `reset()` and `step()`
- **Audit log** — every policy violation is persisted to `AuditLog` with episode ID, step index, rule ID, severity, and timestamp
- **Policy Violation Viewer** — filterable table of violations by environment, episode, and severity (high / medium / low)

---

## Architecture

```
Natural language spec
        │
        ▼
  Extraction Pipeline
  (entity → action → task → policy → CompilerInput)
        │
        ▼
    Jinja2 Compiler  ──→  generated_envs/<name>/
        │
        ▼
      ForgeEnv (Gymnasium)
  ┌─────────────────────────────────┐
  │  reset()                        │
  │    InitialStateFactory          │
  │    ObservationFilter            │
  │                                 │
  │  step(action)                   │
  │    ActionValidator              │
  │    PolicyEngine  ──→ AuditLog   │
  │    TransitionEngine             │
  │    VerifierEngine               │
  │    RewardEngine                 │
  │    TelemetryClient              │
  │    ObservationFilter            │
  └─────────────────────────────────┘
        │
        ▼
  Celery Workers  ──→  Export (jsonl / parquet)
```

---

## Project Structure

```
forge/
  runtime/        # Gymnasium env, state, trajectory, verifiers, agents
  extraction/     # LLM pipeline, PII redactor, schemas
  compiler/       # Jinja2 compiler, package builder
  customization/  # Override hooks, config loader
  cli/            # forge CLI commands
  templates/      # Jinja2 env templates
backend/
  app/
    api/          # FastAPI routers (compile, envs, episodes, rollouts, exports, audit)
    services/     # RunnerService, RolloutService, ExportService
    worker/       # Celery tasks
    models.py     # SQLAlchemy models (Episode, RolloutJob, ExportJob, AuditLog)
frontend/         # Next.js 16 app
  app/
    dashboard/    # Episode list
    environments/ # Compiler Review UI + Environment Graph
    rollouts/     # Rollout Launcher
    violations/   # Policy Violation Viewer
  components/     # ViolationTable, RolloutLauncher, ExportPanel, ...
tests/
  runtime/        # Kernel, verifier, policy, RBAC, network isolation, PII tests
  backend/        # API integration tests
generated_envs/   # Output of the compiler (gitignored)
examples/         # gmail_env reference implementation
```

---

## Getting Started

**Prerequisites:** Python 3.11+, Node.js 18+

```bash
# Install Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Install frontend dependencies
npm --prefix frontend install
```

**Run locally:**

```bash
./run.sh        # starts backend (:8000) + frontend (:3000)
./kill.sh       # stops all Forge processes
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

**Run tests:**

```bash
pytest
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FORGE_GENERATED_ENVS_DIR` | `generated_envs` | Where compiled environments are written |
| `FORGE_DEV_NETWORK` | `false` | Set to `true` to bypass network isolation checks |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL used by the frontend |

---

## CLI

```bash
forge compile --spec spec.yaml       # Extract + compile an environment
forge run <env_name> --agent random  # Run an episode interactively
forge replay <episode_id>            # Replay a recorded episode
forge validate <env_name>            # Smoke-test a compiled environment
```
