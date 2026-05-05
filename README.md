# Forge

**Sandbox environments for training AI agents on real-world apps.**

Forge lets you spin up isolated, observable app environments — Gmail-like email clients, Slack-like messaging, custom LLM-generated apps, raw Linux shells, or live browser sessions — and run RL agents inside them. Every action is logged, every state transition is verifiable, and every episode is exportable as a training dataset.

---

## What Forge Does

1. **Creates sandboxed app environments** — Docker containers running real apps (or realistic replicas) with full state access
2. **Runs agents inside them** — Random, scripted, or LLM-powered agents interact with the app via a clean API
3. **Records and rewards every step** — Policy enforcement, multi-method reward scoring, and trajectory logging at each step
4. **Exports training data** — SFT pairs, DPO preference pairs, GRPO rollouts, failure datasets, and more

---

## Environment Types

| Type | What runs | Good for |
|---|---|---|
| **Premade** | Pre-built Gmail or Slack replica | Ready-to-use evaluation; seeded with realistic emails, threads, and DMs |
| **Custom** | LLM-generated FastAPI app | Simulate any business app from a plain-English description |
| **CLI** | Ubuntu 22.04 shell | Shell scripting, sysadmin, package management tasks |
| **Browser** | Chromium + KasmVNC | Web automation, form filling, navigation |

---

## Premade Environments

Premade environments ship with realistic seed data that resembles real products. They're ready to evaluate agents immediately — no configuration needed.

### Gmail
- **34 emails** across Inbox, Sent, Drafts, Spam, and custom labels (Work, Personal, Finance, Newsletter)
- **19 contacts** with names and addresses
- **5 labels** with colour coding
- Send, receive, reply, archive, label, star, delete — all functional
- Auto-reply simulation: sending or replying triggers a realistic response from the recipient
- `POST /receive` endpoint lets evaluators inject new emails mid-episode
- Automatic baseline snapshot saved on first boot for reward drift detection

### Slack
- **7 channels**: `#general`, `#engineering`, `#product`, `#random`, `#design`, `#ops-infra`, `#announcements`
- **38 top-level messages** with multi-sentence, realistic content
- **88 thread replies** stored with correct parent references — clicking any thread shows full conversation
- **43 reactions** across messages
- **12 DMs** with realistic back-and-forth
- Per-channel auto-responders simulate realistic team activity when the agent posts
- Post, reply, react, DM, pin — all functional

---

## Core Features

### Environment Creation

- **4-option creation flow** — CLI, Browser, Custom, and Premade on the new-environment page
- **Custom environment generator** — describe any app in plain English; five parallel LLM agents scaffold code, telemetry, state bridge, policy DSL, and reward function
- **Real-time build progress** — WebSocket stream shows agent completion, Docker build phase, and live worker logs
- **Self-healing `/start`** — detects stale image tags, missing port bindings, and crash-looped containers; clears bad state and auto-recovers
- **10-environment cap** — enforced at UI and API level; expired environments cleaned up automatically

### Container Build & Resilience

- **LLM drift guardrails** — every generated file is post-processed before `docker build`: base image normalised, port forced to 8000, required packages injected
- **Registry fallback** — four-tier fallback when Docker Hub flakes: canonical pull → AWS ECR → GCR → direct HTTPS via `httpx`
- **Worker pre-warm** — Celery pulls base images on boot so user builds always hit cache
- **Crash-loop detection** — `restart_policy=on-failure` with 3-attempt cap; status flips to `error` automatically

### Agent Runs & Data Collection

- **Five agent adapters** — `random`, `scripted:<path>`, `anthropic:<model>`, `openai:<model>`, `vllm:<model>`
- **Trajectory recording** — every step's state, action, and reward persisted to JSONL and DB
- **Cross-run episode selection** — pick episodes from multiple runs, export as a single merged dataset
- **Per-environment dashboard** — pass rate, average reward, step efficiency, termination-reason breakdown

### Observability & Replay

- **Live event feed** — real-time observability panel streamed from the running container
- **Episode replay** — re-run any recorded episode step-by-step from stored trajectory
- **Branch replay** — fork from any step index and try alternate action sequences
- **Failure clustering** — groups failed episodes by trajectory diff similarity
- **Environment graph** — visual entity/action relationship map

### Reward Engine

**Tiered reward** with configurable partial credit, completion bonuses, and efficiency scaling. Mix and match scoring methods per environment:

| Method | How it works |
|---|---|
| **LLM-as-judge** | Claude Haiku evaluates each trajectory against your requirements. Most flexible. |
| **Sentence Embeddings** | Cosine similarity via `all-MiniLM-L6-v2`. Fast, no LLM calls. |
| **ROUGE-L** | Longest common subsequence overlap. Deterministic. |
| **BLEU** | N-gram precision. Best for short, structured outputs. |

### Verifiers

Six built-in verifier types compose into a `RewardBreakdown` returned on every step:

| Verifier | Checks |
|---|---|
| `ExactStateVerifier` | Specific state field values |
| `EventVerifier` | Required events appeared in the trajectory |
| `TemporalVerifier` | Event ordering and timing constraints |
| `NegativeVerifier` | Forbidden events did not occur |
| `PolicyVerifier` | Python expressions against current state |
| `SemanticVerifier` | LLM-based semantic correctness (with embedding cache) |

### Security & Policy

- **PolicyEngine DSL** — Python expressions evaluated in a sandboxed context; violations block transitions and return 0.0 reward
- **Network isolation** — AST-based scanner blocks `requests`, `httpx`, `urllib`, `socket`, `aiohttp` in generated envs (bypass with `FORGE_DEV_NETWORK=true`)
- **PII redaction** — strips emails, phone numbers, and SSNs from LLM input before code generation
- **RBAC observation filtering** — removes or restricts state fields per role, applied in `reset()` and `step()`
- **Policy Violation Viewer** — filterable table of violations by environment, episode, and severity

### Synthetic Data Engine

Generate training data without running live agents:

- **Goal suggestion** — LLM proposes research goals tailored to the env's policy and reward; de-duplicates against existing goals
- **Difficulty scaling** — five tiers (Trivial → Expert) with concrete step-count ranges
- **Edge case injection** — inject `boundary_conditions`, `permission_errors`, `missing_deps`, `conflicting_state`, or `recovery` scenarios into generated trajectories
- **Replay manifest** — saved to `generated_envs/<env>/synthetic_replay.json`; active epochs replace live LLM inference in agent runs

### Dataset Export

Seven export formats from the per-environment **Export Dataset** page:

| Format | File | Use with |
|---|---|---|
| **SFT Pairs** | `sft_pairs.jsonl` | TRL SFTTrainer, OpenAI fine-tuning, Axolotl |
| **Preference Pairs** | `preference_pairs.jsonl` | TRL DPOTrainer, LlamaFactory |
| **RL Trajectories** | `grpo_rollouts.parquet` | TRL GRPOTrainer, veRL, OpenRLHF |
| **Failure Dataset** | `failure_dataset.jsonl` | Contrastive training, red-teaming |
| **Raw Trajectories** | `trajectories.jsonl` | Custom pipelines |
| **Rewards** | `rewards.jsonl` | Analysis, custom reward models |
| **Verifier Results** | `verifier_results.jsonl` | Debugging, custom reward models |

---

## Architecture

```
Browser / API Client
        │
        ▼
  Next.js Frontend (:3000)
        │  REST / WebSocket
        ▼
  FastAPI Backend (:8000)
        │
        ├── SQLite (forge.db)       — environments, runs, episodes, audit log
        │
        └── Celery Worker           — async tasks
              │
              ├── CLI:      docker run ubuntu:22.04
              ├── Browser:  docker run linuxserver/chromium
              ├── Premade:  docker run pre-built image (gmail / slack)
              └── Custom:   5 parallel LLM agents → docker build → docker run
                                │
                                ▼
                    Reverse Proxy → Sandbox Hub (App / Terminal / Observability)
                                │
                                ▼
                    ForgeEnv (Gymnasium-compatible)
                    ┌─────────────────────────────────┐
                    │  reset()                        │
                    │    InitialStateFactory          │
                    │    ObservationFilter (RBAC)     │
                    │                                 │
                    │  step(action)                   │
                    │    ActionValidator              │
                    │    PolicyEngine  ──→ AuditLog   │
                    │    TransitionEngine             │
                    │    VerifierEngine               │
                    │    RewardEngine (multi-method)  │
                    │    TelemetryClient              │
                    └─────────────────────────────────┘
                                │
                                ▼
                    Export (7 formats: SFT / DPO / RL / Failure / Raw)
```

---

## Project Structure

```
forge/
  runtime/             # Gymnasium env, state, trajectory, verifiers, agents
  extraction/          # LLM pipeline, PII redactor, schemas
  compiler/            # Jinja2 compiler, package builder
  envgen/              # LLM orchestration, container runtime, episode/CLI/browser runners
    agents/            # AppGenerator, Telemetry, StateBridge, Policy, Reward agents
    container.py       # Docker build, run, start/stop, normalisation, mirror fallback
    tiered_reward.py   # TieredRewardEngine with partial credit and multi-method scoring
    ml_reward.py       # SentenceEmbeddingScorer, NGramScorer (ROUGE-L / BLEU)
backend/
  app/
    api/               # FastAPI routers: sandbox, agent_runs, synthetic, evaluate, exports, audit
    services/
      export_writers/  # sft_pairs, preference_pairs, grpo_rollouts, failure_dataset, ...
    worker/            # Celery tasks: build_sandbox, run_episode, run_rollout, cleanup_expired
    models.py          # SQLAlchemy models: SandboxEnvironment, Episode, AgentRun, AuditLog, ...
frontend/
  app/
    environments/
      new/             # 4-option landing page (CLI / Browser / Custom / Premade)
      [env_name]/
        sandbox/       # Tabbed hub: App / Terminal / Observability
        agent/         # Agent runs + cross-run episode selection
        dashboard/     # Pass rate, reward distribution, step efficiency
        policy/        # Policy requirements editor
        reward/        # Reward requirements + scoring method selector
        synthetic/     # Synthetic data: goal suggestion, difficulty, edge cases
        export/        # Dataset export
        violations/    # Policy audit log
        replay/        # Episode step-through viewer
docker/
  premade/
    gmail/             # Gmail-like environment (seeded with 34 emails, 19 contacts)
    slack/             # Slack-like environment (seeded with 7 channels, 88 thread replies)
tests/
  runtime/             # Kernel, verifier, policy, RBAC, network isolation, PII tests
  backend/             # API integration tests, E2E sandbox + agent-runs tests
  envgen/              # ContainerRuntime, normalisation, pull/mirror/HTTPS fallback tests
```

---

## Getting Started

**Prerequisites:** Python 3.11+, Node.js 18+, Docker, Redis

```bash
# 1. Clone and install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
npm --prefix frontend install

# 2. Start everything
./run.sh        # Redis + Celery worker + backend (:8000) + frontend (:3000)

# 3. Stop everything
./kill.sh
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |

**Run tests:**
```bash
pytest
```

---

## Environment Variables

### Infrastructure

| Variable | Default | Description |
|---|---|---|
| `FORGE_GENERATED_ENVS_DIR` | `generated_envs` | Where compiled environments are written |
| `FORGE_DEV_NETWORK` | `false` | Set to `true` to bypass network isolation in generated envs |
| `FORGE_DISABLE_PREWARM` | unset | Set to `1` to skip base-image pre-warm on worker boot |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for Celery and build progress pub/sub |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL used by the frontend |

### LLM Provider

All LLM calls go through a single `get_client()` factory — swap providers or models without touching code.

| Variable | Default | Description |
|---|---|---|
| `FORGE_LLM_PROVIDER` | `anthropic` | LLM backend. Supported: `anthropic`, `ollama` |
| `FORGE_LLM_MODEL` | `claude-haiku-4-5-20251001` | Standard-tier model (faster, cheaper) |
| `FORGE_LLM_MODEL_CAPABLE` | `claude-sonnet-4-6` | Capable-tier model (code generation, complex reasoning) |
| `ANTHROPIC_API_KEY` | — | Required when `FORGE_LLM_PROVIDER=anthropic` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

**Run fully locally with Ollama:**
```bash
FORGE_LLM_PROVIDER=ollama FORGE_LLM_MODEL=gemma4:12b ./run.sh
```

---

## CLI

```bash
forge compile --spec spec.yaml       # Extract + compile an environment
forge run <env_name> --agent random  # Run an episode interactively
forge replay <episode_id>            # Replay a recorded episode
forge validate <env_name>            # Smoke-test a compiled environment
```
