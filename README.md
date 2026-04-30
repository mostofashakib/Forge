# Forge

**Spin up real-world application environments for reinforcement learning.**

Forge lets you create three kinds of sandboxed environments — a high-fidelity app generated from a natural-language description, a full Linux CLI, or a live Chromium browser — all running in Docker and observable from a web UI. Each environment is paired with an RL runtime (Gymnasium-compatible), policy enforcement, reward computation, and training data export.

---

## Environment Types

| Type | Infrastructure | Use case |
|---|---|---|
| **General Purpose** | LLM-generated FastAPI app + Docker | Simulate real-world business apps with full observability, policy rules, and reward functions |
| **CLI** | Ubuntu 22.04 in Docker | Run scripts, install packages, interact via an integrated terminal |
| **Browser** | Chromium + KasmVNC in Docker | Web automation, browser-based RL tasks, accessible via a web UI |

---

## Features

### Sandbox Lifecycle
- **Create environments from the browser** — pick an environment type, name it, set a TTL, and Forge handles the rest
- **Real-time build progress** — WebSocket-based progress stream shows agent completion, Docker build phase, and live worker logs
- **Start / stop / delete** — full container lifecycle management from the UI or API
- **10-environment cap** — enforced at the UI and API level; expired environments are cleaned up automatically by a scheduled Celery task
- **Tabbed sandbox hub** — App / Terminal / Observability tabs, each full-screen; Browser envs go straight to the VNC iframe, CLI envs go straight to the terminal

### General Purpose Environment Generation
- **Multi-agent orchestration** — five parallel LLM agents generate the app code, telemetry instrumentation, state bridge, policy DSL, and reward function
- **Extraction pipeline** — entity extractor → action inferencer → task generator → policy parser → `CompilerInput`
- **Jinja2 compiler** — `CompilerInput` → runnable Python package written to `generated_envs/<name>/`
- **Docker build & launch** — image built from the generated app, container started and port-mapped automatically
- **Reverse proxy** — Next.js API route proxies the live app UI at `/api/proxy/<env_name>/ui`

### Runtime Kernel
- **Gymnasium-compatible `ForgeEnv`** — drop-in `reset()` / `step()` loop with full 5-tuple returns
- **Deterministic replay** — same seed + action sequence always produces an identical trajectory hash
- **StateStore + TrajectoryStore** — immutable state snapshots, per-episode trajectory recording
- **ActionValidator** — rejects unknown action types before they reach the transition layer
- **TransitionEngine** — register named transition functions; composable per environment
- **Clock** — logical time advancement per step for temporally-sensitive verifiers

### Observability & Replay
- **`TelemetryClient`** — records every step snapshot and episode completion to SQLite
- **Live event feed** — real-time observability panel streamed from the running container
- **Episode replay** — re-run any recorded episode step-by-step from the stored trajectory
- **Branch replay** — fork from any step index, try alternate action sequences
- **Failure clustering** — groups failed episodes by trajectory diff similarity
- **Dashboard** — episode list with reward summaries and pass/fail status
- **Episode Replay UI** — step-through viewer with state diff and event log
- **Environment Graph** — visual entity/action relationship map

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
- **PolicyEngine DSL** — Python expressions evaluated in a sandboxed context; violations block transitions and return 0.0 reward
- **Network isolation** — AST-based static scanner blocks `requests`, `httpx`, `urllib`, `socket`, `aiohttp` imports in generated environments; bypassed by `FORGE_DEV_NETWORK=true`
- **PII redaction** — regex-based redactor strips emails, phone numbers, and SSNs from `CompilerInput` before code generation
- **RBAC observation filtering** — `ObservationFilter` removes or restricts state fields per role; applied transparently in `reset()` and `step()`
- **Audit log** — every policy violation is persisted with episode ID, step index, rule ID, severity, and timestamp
- **Policy Violation Viewer** — filterable table of violations by environment, episode, and severity

---

## Architecture

```
                   Browser         ┌─────────────┐
                  ┌────────────────│  Next.js UI │
                  │                └──────┬──────┘
                  │                       │ REST / WebSocket
                  ▼                       ▼
         Docker Container       FastAPI Backend (:8000)
         (Chromium+KasmVNC)           │
                                       ├── POST /api/sandbox/  →  Celery task
                   CLI                 │                              │
                  ┌───────────────     │                    Redis pub/sub (progress)
                  │                    │                              │
                  ▼                    ▼                              ▼
         Docker Container       SQLite (forge.db)          Celery Worker
         (Ubuntu 22.04)                                         │
                                                                ├── CLI:     run ubuntu:22.04
          General Purpose                                       ├── Browser: run linuxserver/chromium
         ┌──────────────────                                    └── General: LLM agents → Docker build
         │
         ▼
  LLM Orchestration (5 agents in parallel)
    App Code → Telemetry → State Bridge → Policy DSL → Reward Fn
         │
         ▼
  Docker Build & Run  →  Reverse Proxy  →  Sandbox Hub (App / Terminal / Observability)
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
  envgen/         # LLM orchestration agents, container runtime (Docker), artifact bus
  cli/            # forge CLI commands
  templates/      # Jinja2 env templates
backend/
  app/
    api/          # FastAPI routers (sandbox, compile, envs, episodes, rollouts, exports, audit)
    services/     # EnvOrchestrator, RunnerService, RolloutService, ExportService, ExtractionService
    worker/       # Celery tasks (build_sandbox, run_episode, run_rollout, cleanup_expired)
    models.py     # SQLAlchemy models (SandboxEnvironment, Episode, RolloutJob, ExportJob, AuditLog)
frontend/         # Next.js app
  app/
    environments/          # Sandbox list, new environment form, build progress, sandbox hub
      new/                 # Environment type selector + form
      [env_name]/
        progress/          # Real-time build progress (WebSocket + REST fallback)
        sandbox/           # Tabbed sandbox hub (App / Terminal / Observability)
        graph/             # Entity/action relationship map
        replay/            # Episode step-through viewer
        config/            # Environment config editor
    dashboard/             # Episode list with reward summaries
    rollouts/              # Rollout Launcher
    violations/            # Policy Violation Viewer
    compiler-review/       # Extracted entities and generated code inspector
    api/proxy/             # Next.js reverse proxy to live app containers
  components/              # SandboxTerminal, SandboxEventFeed, ViolationTable, RolloutLauncher, ...
tests/
  runtime/        # Kernel, verifier, policy, RBAC, network isolation, PII tests
  backend/        # API integration tests, E2E sandbox creation tests
generated_envs/   # Output of the compiler (gitignored)
examples/         # gmail_env reference implementation
```

---

## Getting Started

**Prerequisites:** Python 3.11+, Node.js 18+, Docker, Redis

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
./run.sh        # starts Redis, Celery worker, backend (:8000), and frontend (:3000)
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
| `FORGE_DEV_NETWORK` | `false` | Set to `true` to bypass network isolation checks in generated envs |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL used by Celery and the build progress pub/sub channel |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL used by the frontend |

---

## CLI

```bash
forge compile --spec spec.yaml       # Extract + compile an environment
forge run <env_name> --agent random  # Run an episode interactively
forge replay <episode_id>            # Replay a recorded episode
forge validate <env_name>            # Smoke-test a compiled environment
```
