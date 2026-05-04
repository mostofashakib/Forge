# Forge

**Spin up real-world application environments for reinforcement learning.**

Forge lets you create sandboxed environments from a natural-language description, a premade template, a full Linux CLI, or a live Chromium browser — all running in Docker and observable from a web UI. Each environment is paired with an RL runtime (Gymnasium-compatible), policy enforcement, configurable reward scoring, a synthetic data engine, and training data export.

---

## Environment Types

| Type | Infrastructure | Use case |
|---|---|---|
| **CLI** | Ubuntu 22.04 in Docker | Shell scripting, package management, system administration |
| **Browser** | Chromium + KasmVNC in Docker | Web automation, form filling, navigation tasks |
| **Custom** | LLM-generated FastAPI app + Docker | Simulate any real-world business app with full observability, policy rules, and reward |
| **Premade** | Pre-built Docker image | Ready-to-use environments with pre-configured policy and reward — Gmail and Slack available |

---

## Features

### Environment Creation

- **4-option creation flow** — CLI, Browser, Custom, and Premade cards on the new-environment page
- **Premade environments** — Gmail (email client) and Slack (team messaging) ship with realistic apps, pre-configured policy rules, and reward criteria; more templates planned
- **Custom environment generator** — describe any real-world app in plain English; Forge runs five parallel LLM agents to scaffold the code, telemetry, state bridge, policy DSL, and reward function
- **Real-time build progress** — WebSocket-based progress stream shows agent completion, Docker build phase, and live worker logs
- **Start / stop / delete** — full container lifecycle management from the UI or API
- **Self-healing `/start` endpoint** — detects stale image tags, missing port bindings, and crash-looped containers; clears bad state and either auto-recovers or surfaces a clear error prompting a rebuild
- **10-environment cap** — enforced at UI and API level; expired environments cleaned up automatically by a scheduled Celery task

### Container Build & Resilience

- **LLM drift guardrails** — every generated file is post-processed before `docker build`:
  - `FROM` normalised to `python:3.12-slim` so all envs share one warm cache
  - `EXPOSE` and `--port` forced to port 8000 regardless of what the LLM picked
  - FastAPI + Forge baseline injected into `requirements.txt` to prevent `ModuleNotFoundError` at runtime
- **Registry resilience** — four-tier fallback when Docker Hub is flaking: canonical pull → AWS ECR mirror → GCR mirror → direct HTTPS via `httpx` with `docker load`
- **Worker boot pre-warm** — Celery `worker_ready` signal pulls base images in the background so user-triggered builds always find them in cache
- **Crash-loop detection** — `restart_policy=on-failure` with a 3-attempt cap; status flips to `error` when `RestartCount > 0`
- **Port-binding race protection** — `_wait_for_port_binding()` polls Docker until the host-port allocation appears before writing to the DB

### Agent Runs & Data Collection

- **Run agents inside any sandbox** — pick an agent (random / scripted / LLM-driven), set a step budget, and execute episodes against the live app
- **Five agent adapters** — `random`, `scripted:<path>`, `anthropic:<model>`, `openai:<model>`, `vllm:<model>`
- **Trajectory recording** — every step's state, action, and reward persisted to JSONL alongside DB rows tracking the run, episodes, and termination reasons
- **Cross-run episode selection** — pick episodes from multiple agent runs and export merged trajectories as a single training dataset
- **Per-environment dashboard** — pass rate, average reward, step efficiency, and termination-reason breakdown across all runs

### Synthetic Data Engine

Generate synthetic training epochs from a research goal without running live agents:

- **Goal suggestion** — LLM proposes novel research goals tailored to the environment's type, policy constraints, and reward criteria; de-duplicates against any goals already in the textarea
- **Difficulty scaling** — five tiers (Trivial → Expert) with concrete step-count ranges and complexity constraints injected into each generation prompt
- **Edge case injection** — select one or more structured edge case types to weave into every trajectory:

  | Type | Behaviour |
  |---|---|
  | `boundary_conditions` | Empty files, max-length inputs, threshold outputs |
  | `permission_errors` | Read-only paths or missing sudo — agent must adapt |
  | `missing_deps` | Commands or packages not yet installed |
  | `conflicting_state` | File already exists, port in use, service already running |
  | `recovery` | Mid-trajectory failure the agent must diagnose and fix |

- **Episode generator** — one LLM call per episode, returns a realistic bash command sequence; failed generations are skipped and the rest are returned
- **Replay manifest** — generated epochs are saved to `generated_envs/<env>/synthetic_replay.json`; active epochs replace live LLM inference in agent runs

### Reward Engine

- **Tiered reward** — `TieredRewardEngine` with configurable partial credit, completion bonuses, and efficiency scaling
- **Multi-method scoring** — select one or more scoring methods; when multiple are active, per-episode scores are averaged:

  | Method | Description |
  |---|---|
  | **LLM-as-judge** | Claude Haiku evaluates each trajectory against your requirements. Most flexible. |
  | **Sentence Embeddings** | Cosine similarity using `all-MiniLM-L6-v2`. Fast, no LLM calls. |
  | **ROUGE-L** | Longest common subsequence overlap between requirements and trajectory text. Deterministic. |
  | **BLEU** | N-gram precision overlap. Best for short, structured outputs. |

- **Reward configuration UI** — per-environment page to set requirements text and toggle scoring methods; persisted to `reward_config.json`
- **Re-run evaluation** — score any set of existing trajectories against updated requirements without re-running agents

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

### Observability & Replay

- **`TelemetryClient`** — records every step snapshot and episode completion to SQLite
- **Live event feed** — real-time observability panel streamed from the running container
- **Episode replay** — re-run any recorded episode step-by-step from the stored trajectory
- **Branch replay** — fork from any step index, try alternate action sequences
- **Failure clustering** — groups failed episodes by trajectory diff similarity
- **Environment graph** — visual entity/action relationship map
- **Container logs endpoint** — `GET /api/sandbox/{env}/logs` surfaces stdout/stderr, exit code, and restart count

### Dataset Export

Seven export formats, accessible from the per-environment **Export Dataset** page:

| Format | File | Use with |
|---|---|---|
| **SFT Pairs** | `sft_pairs.jsonl` | Messages-style pairs (user = task, assistant = command sequence) from passing episodes only. | TRL SFTTrainer, OpenAI fine-tuning, Axolotl |
| **Preference Pairs** | `preference_pairs.jsonl` | Chosen/rejected trajectory pairs ranked by total reward, grouped by (task, seed bucket). Full command sequences included in both sides. | TRL DPOTrainer, LlamaFactory |
| **RL Trajectories** | `grpo_rollouts.parquet` | Episode rollout table with `prompt`, `completion`, `total_reward`, and `per_step_rewards` per row. | TRL GRPOTrainer, veRL, OpenRLHF |
| **Failure Dataset** | `failure_dataset.jsonl` | Full step-by-step trajectories from failed episodes with per-step verifier diagnostics. | Failure analysis, contrastive training, red-teaming |
| **Raw Trajectories** | `trajectories.jsonl` | Full step-by-step data for all completed episodes. | Custom pipelines |
| **Rewards** | `rewards.jsonl` | Per-episode reward breakdowns with step-level components. | Analysis, custom reward models |
| **Verifier Results** | `verifier_results.jsonl` | Detailed verifier pass/fail and scores per step. | Debugging, custom reward models |

### Security & Policy

- **PolicyEngine DSL** — Python expressions evaluated in a sandboxed context; violations block transitions and return 0.0 reward
- **Network isolation** — AST-based static scanner blocks `requests`, `httpx`, `urllib`, `socket`, `aiohttp` imports in generated environments; bypassed by `FORGE_DEV_NETWORK=true`
- **PII redaction** — regex-based redactor strips emails, phone numbers, and SSNs from `CompilerInput` before code generation
- **RBAC observation filtering** — `ObservationFilter` removes or restricts state fields per role; applied transparently in `reset()` and `step()`
- **Audit log** — every policy violation persisted with episode ID, step index, rule ID, severity, and timestamp
- **Policy Violation Viewer** — filterable table of violations by environment, episode, and severity

### Runtime Kernel

- **Gymnasium-compatible `ForgeEnv`** — drop-in `reset()` / `step()` loop with full 5-tuple returns
- **Deterministic replay** — same seed + action sequence always produces an identical trajectory hash
- **StateStore + TrajectoryStore** — immutable state snapshots, per-episode trajectory recording
- **ActionValidator** — rejects unknown action types before they reach the transition layer
- **TransitionEngine** — register named transition functions; composable per environment
- **Clock** — logical time advancement per step for temporally-sensitive verifiers
- **Parallel Celery rollouts** — configurable worker pool for batch episode execution

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
          Premade / Custom                                      ├── Browser: run linuxserver/chromium
         ┌──────────────────                                    ├── Premade: run pre-built image
         │                                                      └── Custom:  LLM agents → Docker build
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
  │    RewardEngine (multi-method)  │
  │    TelemetryClient              │
  │    ObservationFilter            │
  └─────────────────────────────────┘
         │
         ▼
  Celery Workers  ──→  Export (7 formats: SFT / DPO / RL / Failure / Raw)
```

---

## Project Structure

```
forge/
  runtime/             # Gymnasium env, state, trajectory, verifiers, agents
  extraction/          # LLM pipeline, PII redactor, schemas
  compiler/            # Jinja2 compiler, package builder
  customization/       # Override hooks, config loader
  envgen/              # LLM orchestration, container runtime, episode/CLI/browser runners
    agents/            # AppGenerator, Telemetry, StateBridge, Policy, Reward, CLI/Browser/Container agents
    container.py       # Docker build, run, start/stop, normalisation, mirror fallback, pre-warm
    _image_pull_http.py# Direct HTTPS OCI registry pull (bypass for unstable dockerd transport)
    episode_runner.py  # Drives an agent through a general-env sandbox, records trajectory
    cli_runner.py      # Drives an agent through a CLI sandbox via docker exec
    browser_runner.py  # Drives an agent through a browser sandbox via Chrome DevTools Protocol
    tiered_reward.py   # TieredRewardEngine with partial credit and multi-method scoring
    ml_reward.py       # SentenceEmbeddingScorer (all-MiniLM-L6-v2), NGramScorer (ROUGE-L / BLEU)
  cli/                 # forge CLI commands
  templates/           # Jinja2 env templates
backend/
  app/
    api/               # FastAPI routers: sandbox, agent_runs, synthetic, evaluate, exports, audit, ...
    services/
      export_writers/  # sft_pairs, preference_pairs, grpo_rollouts, failure_dataset, trajectories, rewards, verifier_results
    worker/            # Celery tasks: build_sandbox, run_episode, run_rollout, run_agent_run, cleanup_expired
    models.py          # SQLAlchemy models: SandboxEnvironment, Episode, AgentRun, AgentEpisode, RolloutJob, ExportJob, AuditLog
frontend/
  app/
    environments/
      new/                    # 4-option landing page (CLI / Browser / Custom / Premade)
        custom/               # Custom environment creation form
        premade/              # Premade template selection (Gmail, Slack)
      [env_name]/
        progress/             # Real-time build progress (WebSocket + REST fallback)
        sandbox/              # Tabbed hub: App / Terminal / Observability
        agent/                # Agent runs management + cross-run episode selection
        dashboard/            # Pass rate, reward distribution, step efficiency
        policy/               # Policy requirements editor
        reward/               # Reward requirements + multi-method scoring selector
        evaluate/             # Re-run policy/reward evaluation on existing trajectories
        synthetic/            # Synthetic data: goal suggestion, difficulty, edge case injection
        export/               # Dataset export: SFT / DPO / RL / Failure / Raw formats
        violations/           # Policy audit log
        graph/                # Entity/action relationship map
        replay/               # Episode step-through viewer
        config/               # Environment config editor
    dashboard/                # Global episode list
    rollouts/                 # Rollout launcher
    violations/               # Global policy violation viewer
    compiler-review/          # Extracted entities and generated code inspector
    api/proxy/                # Next.js reverse proxy to live app containers
  components/                 # SandboxTerminal, SandboxEventFeed, ViolationTable, ExportPanel, ...
tests/
  runtime/        # Kernel, verifier, policy, RBAC, network isolation, PII tests
  backend/        # API integration tests, E2E sandbox + agent-runs creation tests
  envgen/         # ContainerRuntime, normalisation, pull/mirror/HTTPS fallback tests
docker/
  premade/
    gmail/        # Pre-built Gmail-like environment image
    slack/        # Pre-built Slack-like environment image
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
| `FORGE_DISABLE_PREWARM` | unset | Set to `1` to skip worker-startup base-image pre-warm (useful in tests / sandboxed CI) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL used by Celery and the build progress pub/sub channel |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL used by the frontend |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Override Celery broker (defaults to `REDIS_URL`) |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/0` | Override Celery result backend |

---

## CLI

```bash
forge compile --spec spec.yaml       # Extract + compile an environment
forge run <env_name> --agent random  # Run an episode interactively
forge replay <episode_id>            # Replay a recorded episode
forge validate <env_name>            # Smoke-test a compiled environment
```
