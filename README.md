# Forge

**Sandbox environments for training AI agents on real-world apps.**

Forge lets you spin up isolated, observable app environments — Gmail-like email clients, Slack-like messaging, custom LLM-generated apps, raw Linux shells, or live browser sessions — and run RL agents inside them. Every action is logged, every state transition is verifiable, and every episode is exportable as a training dataset.

---

## What Forge Does

1. **Creates sandboxed app environments** — Docker containers running real apps (or realistic replicas) with full state access
2. **Runs agents inside them** — Random, scripted, or LLM-powered agents interact with the app via a clean API
3. **Records and rewards every step** — Policy enforcement, multi-method reward scoring, and trajectory logging at each step
4. **Exports training data** — SFT pairs, DPO preference pairs, GRPO rollouts, failure datasets, and more
5. **Benchmarks environment quality** — Runs a task suite across domains and scores coverage, reward density, dead-end rate, and action diversity

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
- **Compiler review** — inspect and edit LLM-generated compiler input before the build starts
- **Self-healing `/start`** — detects stale image tags, missing port bindings, and crash-looped containers; clears bad state and auto-recovers
- **10-environment cap** — enforced at UI and API level; expired environments cleaned up automatically

### Container Build & Resilience

- **LLM drift guardrails** — every generated file is post-processed before `docker build`: base image normalised, port forced to 8000, required packages injected
- **Registry fallback** — four-tier fallback when Docker Hub flakes: canonical pull → AWS ECR → GCR → direct HTTPS via `httpx`
- **Worker pre-warm** — Celery pulls base images on boot so user builds always hit cache
- **Crash-loop detection** — `restart_policy=on-failure` with 3-attempt cap; status flips to `error` automatically

### Agent Runs & Data Collection

- **Five agent adapters** — `random`, `scripted:<path>`, `anthropic:<model>`, `openai:<model>`, `vllm:<model>`
- **AgentContext** — per-episode agent memory with a compact deterministic digest for prompt injection, stuck-vs-context-limit diagnosis, and automatic pruning of error spam and revisited-state noise
- **Trajectory recording** — every step's state, action, and reward persisted to JSONL and DB
- **Cross-run episode selection** — pick episodes from multiple runs, export as a single merged dataset
- **Parallel rollouts** — launch batched episode rollouts across any compiled environment from the global Rollouts page; `ParallelRolloutRunner` runs the same task across many isolated env copies concurrently (one fresh instance per rollout, millisecond start/teardown) and classifies each outcome as success, failure, partial success, or edge case so a single batch yields diverse training scenarios
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

**Layered verification** — `LayeredVerifier` composes five layers into one verdict: final-state checks, invariant milestone checks (none skipped, correct order), trajectory checks (necessary tool calls made, no unnecessary ones), LLM-as-judge rubrics for creative tasks, and negative checks for unintended side effects.

**Reward-hacking audit** — `RewardHackingAuditor` is a separate audit agent that asks whether a passing verdict was *earned*: it flags passes with skipped milestones, suspiciously short episodes, redundant call patterns, and supports a pluggable LLM audit client. `RewardHackingAuditor.for_verifier(...)` inherits the milestone list straight from a `LayeredVerifier`.

### Interaction Contracts

Every environment declares which capabilities the agent has access to — `tool_use`, `computer_use`, `browser_use`, or any combination via `env.capabilities()`. Each capability validates actions against its schema *before* anything touches the environment, so a hallucinated tool or out-of-bounds click never executes:

| Capability | Interacts with | Schema enforces |
|---|---|---|
| **ToolUse** | API endpoints / functions of the environment | tool exists, required params present, param types match |
| **ComputerUse** | The VM / OS (Linux, macOS, Windows) | allowed primitives (`exec`, `screenshot`), non-empty commands |
| **BrowserUse** | The browser | allowed primitives (`click`, `type`, `press`, `navigate`, `scroll`), viewport bounds |

- Every `ForgeEnv` exposes ToolUse (`env.tool_use.execute(...)` is a schema-validated `step()`); attach the others with `EnvBuilder.with_computer_use(...)` / `.with_browser_use(...)`
- CLI environments grant ComputerUse (`os="linux"`); browser environments route every agent action through BrowserUse

### Determinism

Environments are verified deterministic at creation and launch — same seed and same trajectory always produce the same observations *and* the same score:

- **Launch-time determinism check** — two identically-seeded rollouts are hashed (observations + rewards + termination flags); a mismatch raises `DeterminismError` and aborts the launch. Runs in the backend env loader, `forge run` / `forge export`, and `EnvBuilder.build()` (skip with `FORGE_SKIP_DETERMINISM_CHECK=1`)
- **EnvBuilder + DeterminismConfig** — virtual clock, seeded RNG and UUIDs, canonical sorted-key JSON, float rejection (integers only), serialized transitions, network and filesystem guards inside the env, and fresh-universe startup (factory caches dropped every reset)
- **Replayable episodes** — every step records the tool call, emitted events, state diff, hashes, and reward; `replay_episode(env, seed, steps)` re-executes any recording and verifies every state hash and reward against it
- **Flake-free UI** — premade UIs ship a CSS no-motion override and browser sessions force `prefers-reduced-motion` + injected no-animation styles
- **SQLite as source of truth** — premade and generated apps persist state in SQLite; verification reads `/forge/state` (DB-backed), never the UI
- **Enforced separation of concerns** — architecture tests keep environment, agents, verifiers, and training code from importing across boundaries

### Security & Policy

- **PolicyEngine DSL** — Python expressions evaluated in a sandboxed context; violations block transitions and return 0.0 reward
- **Network isolation** — AST-based scanner blocks `requests`, `httpx`, `urllib`, `socket`, `aiohttp` in generated envs (bypass with `FORGE_DEV_NETWORK=true`)
- **PII redaction** — strips emails, phone numbers, and SSNs from LLM input before code generation
- **RBAC observation filtering** — removes or restricts state fields per role, applied in `reset()` and `step()`
- **Policy Violation Viewer** — global filterable table of violations by environment, episode, and severity

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

## Benchmark

Benchmark runs a fixed task suite across your environments, collects episodes, and scores each environment on four quality metrics. Results are accessible from the **Benchmark** section in the top nav.

### Task Suite

Two domains, 10 tasks total, difficulty 1–5:

| Domain | Tasks |
|---|---|
| `email` | Read & star, reply & label, bulk archive, multi-step reply+label, conditional filter & schedule send |
| `project_mgmt` | View & mark done, create & assign, filter & set deadline, find blocked & reassign, cross-project dependency |

The `--depth` / **Max difficulty** slider is a ceiling: only tasks with `difficulty ≤ depth` are included. Depth 1 runs easiest tasks only; depth 5 runs all tasks.

### Quality Metrics

| Metric | What it measures | Target |
|---|---|---|
| **State Coverage** | Fraction of state schema fields touched per step on average | ≥ 0.7 |
| **Reward Density** | Fraction of steps that produced a positive reward | ≥ 0.7 |
| **Dead-end Rate** | Fraction of episodes that terminated with no progress | ≤ 0.3 |
| **Action Diversity** | Unique endpoints / total endpoints called | ≥ 0.7 |

The report page colour-codes each value: green ≥ 0.7, amber 0.4–0.7, red < 0.4 (dead-end rate is inverted before thresholding).

### Web UI

| Page | What it does |
|---|---|
| **Run** | Configure domains, max difficulty (1–5), seeds per task, and output dir; launch with live log streaming and a progress bar |
| **Report** | Table of quality metrics per environment for the most recent completed run; CSV download |
| **Transfer** | Stub — fine-tune a base model on collected data (GPU node required) |
| **Eval** | Stub — evaluate a fine-tuned checkpoint zero-shot on WebArena / WorkArena (eval harness required) |

### CLI

```bash
forge benchmark run \
  --domains email,project_mgmt \
  --depth 5 \
  --seeds 5 \
  --output benchmark_results

forge benchmark report --output benchmark_results

forge benchmark transfer \
  --data benchmark_results/data \
  --base-model meta-llama/Llama-3.1-8B   # requires GPU + trl/transformers/datasets

forge benchmark eval \
  --checkpoint ./benchmark_results/forge_ft \
  --suite webArena                         # requires eval harness
```

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
        ├── SQLite (forge.db)       — environments, runs, episodes, benchmark_runs, audit log
        │
        └── Celery Worker           — async tasks
              │
              ├── CLI:       docker run ubuntu:22.04
              ├── Browser:   docker run linuxserver/chromium
              ├── Premade:   docker run pre-built image (gmail / slack)
              ├── Custom:    5 parallel LLM agents → docker build → docker run
              │                       │
              │                       ▼
              │           Reverse Proxy → Sandbox Hub (App / Terminal / Observability)
              │                       │
              │                       ▼
              │           ForgeEnv (Gymnasium-compatible)
              │           ┌─────────────────────────────────┐
              │           │  reset()                        │
              │           │    InitialStateFactory          │
              │           │    ObservationFilter (RBAC)     │
              │           │                                 │
              │           │  step(action)                   │
              │           │    ActionValidator              │
              │           │    PolicyEngine  ──→ AuditLog   │
              │           │    TransitionEngine             │
              │           │    VerifierEngine               │
              │           │    RewardEngine (multi-method)  │
              │           │    TelemetryClient              │
              │           └─────────────────────────────────┘
              │                       │
              │                       ▼
              │           Export (7 formats: SFT / DPO / RL / Failure / Raw)
              │
              └── Benchmark: TaskSuite → DataCollector → EnvQualityMetrics
                                                │
                                                ▼
                                         BenchmarkReport → report.json / CSV
```

---

## Project Structure

```
forge/
  runtime/             # Gymnasium env, state, trajectory, verifiers, agents
  extraction/          # LLM pipeline, PII redactor, schemas
  compiler/
    generators/        # Jinja2 compiler, package builder
  envgen/              # LLM orchestration, container runtime, episode/CLI/browser runners
    agents/            # AppGenerator, Telemetry, StateBridge, Policy, Reward agents
    telemetry/         # Telemetry client and collectors
    container.py       # Docker build, run, start/stop, normalisation, mirror fallback
    tiered_reward.py   # TieredRewardEngine with partial credit and multi-method scoring
    ml_reward.py       # SentenceEmbeddingScorer, NGramScorer (ROUGE-L / BLEU)
  benchmark/
    task_suite.py      # Task registry: 2 domains, 10 tasks, difficulty 1–5
    data_collector.py  # Episode collection loop
    env_quality.py     # EnvQualityMetrics: coverage, reward density, dead-end rate, diversity
    report.py          # BenchmarkReport: paper-ready figures and summary tables
    transfer_pipeline.py  # Fine-tune stub (GPU required)
    _fine_tune.py      # fine_tune_model() entry point
    _eval.py           # evaluate_on_suite() entry point (eval harness required)
  customization/       # Environment customisation helpers
  schema/              # StateSchemaManifest and related schemas
  cli/
    main.py            # forge CLI: compile, validate, run, replay, diagnose, benchmark *
backend/
  app/
    api/               # FastAPI routers: sandbox, agent_runs, synthetic, evaluate, exports,
    │                  #   audit, rollouts, detect, compile, benchmark
    services/
      export_writers/  # sft_pairs, preference_pairs, grpo_rollouts, failure_dataset, ...
    worker/            # Celery tasks: build_sandbox, run_episode, run_rollout,
    │                  #   run_benchmark_task, cleanup_expired
    models.py          # SQLAlchemy models: SandboxEnvironment, Episode, AgentRun,
                       #   AuditLog, BenchmarkRun, ...
frontend/
  app/
    dashboard/         # Cross-environment stats: pass rate, reward, failure clusters
    rollouts/          # Global parallel episode rollout launcher
    violations/        # Global policy audit log (filterable by env / episode / severity)
    compiler-review/
      [job_id]/        # Inspect and edit LLM compiler output before build
    benchmark/
      run/             # Launch benchmark: domain/depth/seed config + live log + progress bar
      report/          # Quality metrics table with colour coding + CSV download
      transfer/        # Stub: fine-tune on collected data (GPU required)
      eval/            # Stub: zero-shot eval on WebArena / WorkArena (harness required)
    environments/
      new/             # 4-option landing page (CLI / Browser / Custom / Premade)
      [env_name]/
        sandbox/       # Tabbed hub: App / Terminal / Observability
        progress/      # Real-time build progress (5 LLM agent steps + Docker build)
        agent/         # Agent runs + cross-run episode selection
        dashboard/     # Pass rate, reward distribution, step efficiency
        config/        # Environment config editor
        policy/        # Policy requirements editor
        reward/        # Reward requirements + scoring method selector
        evaluate/      # Policy and reward evaluation viewer
        synthetic/     # Synthetic data: goal suggestion, difficulty, edge cases
        export/        # Dataset export
        violations/    # Per-environment policy audit log
        replay/        # Episode step-through viewer
        graph/         # Visual entity/action relationship map
docker/
  premade/
    gmail/             # Gmail-like environment (seeded with 34 emails, 19 contacts)
    slack/             # Slack-like environment (seeded with 7 channels, 88 thread replies)
tests/
  runtime/             # Kernel, verifier, policy, RBAC, network isolation, PII tests
  backend/             # API integration tests, E2E sandbox + agent-runs + benchmark tests
  envgen/              # ContainerRuntime, normalisation, pull/mirror/HTTPS fallback tests
  benchmark/           # Task suite and quality metric tests
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
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for Celery and build/benchmark progress pub/sub |
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
forge compile --input spec.json      # Extract + compile an environment
forge run <env_name> --agent random  # Run an episode interactively
forge replay <episode_id>            # Replay a recorded episode (ep_* or cep_*)
forge validate <env_name>            # Smoke-test a compiled environment
forge diagnose <env_name>            # Analyse episode quality across all runs

forge benchmark run \
  --domains email,project_mgmt \
  --depth 5 \                        # max difficulty (1=easy only, 5=all tasks)
  --seeds 5                          # episodes per task
forge benchmark report               # generate summary tables from collected results
forge benchmark transfer \
  --data benchmark_results/data \
  --base-model meta-llama/Llama-3.1-8B   # GPU + trl/transformers/datasets required
forge benchmark eval \
  --checkpoint ./benchmark_results/forge_ft \
  --suite webArena                        # eval harness required
```
