# Forge

**Sandbox environments for training AI agents on real-world apps.**

Forge lets you spin up isolated, observable app environments — Gmail-like email clients, Slack-like messaging, custom LLM-generated apps, raw Linux shells, or live browser sessions — and run RL agents inside them. Every action is logged, every state transition is verifiable, and every episode is exportable as a training dataset.

---

## What Forge Does

1. **Creates sandboxed app environments** — Docker containers running real apps (or realistic replicas) with full state access
2. **Runs agents inside them** — Random, scripted, or LLM-powered agents interact with the app via a clean API
3. **Records and rewards every step** — Policy enforcement, multi-method reward scoring, and a unified per-run trace (LLM calls, actions, state changes, verifier decisions) written durably so any run is replayable — even one that crashed mid-episode
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
- **Custom environment generator** — describe any app in plain English; optionally enable the user researcher with an original product name and URL, then a prompt planner creates a dependency-aware task graph for dedicated backend, UI, telemetry, state-bridge, policy, reward, correctness, and review agents
- **Agent-to-Agent context protocol** — specialists exchange typed task and artifact messages while scoped channels expose only each task's declared inputs
- **Reviewer quality gate** — static checks and semantic review verify syntax, required APIs, UI action coverage, RL artifacts, code quality, and the original user requirements before files are written
- **Determinism correctness specialist** — a dedicated gate audits generated code for wall-clock access, unseeded randomness, and nondeterministic identifiers, requiring a counter-based virtual clock (`forge_now()`) and sequential IDs (`_next_id()`) before artifacts are written; after the container boots, a runtime validator proves `/forge/reset` restores the exact initial universe (rows, IDs, counters, database included) and that snapshot/restore round-trips, hard-failing the build on any drift
- **Real-time build progress** — WebSocket stream shows agent completion, Docker build phase, and live worker logs
- **Compiler review** — inspect and edit LLM-generated compiler input before the build starts
- **Self-healing `/start`** — detects stale image tags, missing port bindings, and crash-looped containers; clears bad state and auto-recovers
- **10-environment cap** — enforced at UI and API level; expired environments cleaned up automatically

### Container Build & Resilience

- **LLM drift guardrails** — every generated file is post-processed before `docker build`: base image normalised, port forced to 8000, required packages injected
- **Registry fallback** — four-tier fallback when Docker Hub flakes: canonical pull → AWS ECR → GCR → direct HTTPS via `httpx`
- **Worker pre-warm** — Celery pulls base images on boot so user builds always hit cache
- **Crash-loop detection** — `restart_policy=on-failure` with 3-attempt cap; status flips to `error` automatically

### Custom Generation Pipeline

Custom environment generation separates planning, implementation, assembly, and review:

```text
User prompt + compiler input + optional original product research
          │
          ▼
   UserResearchAgent? ─→ backend / UI / RL / review briefs
          │                 (role-pruned + size-bounded)
          ▼
   PromptPlannerAgent ──→ typed todo DAG + acceptance criteria
          │
          ├── BackendBuilderAgent ─┐
          ├── UIBuilderAgent ──────┴─→ AppAssemblyAgent → TelemetryAgent → StateBridgeAgent
          ├── ScenarioBuilderAgent    (realistic seeded scenarios via seed_state)
          ├── PolicyAgent
          └── RewardAgent
                    │
                    ├─→ EnvironmentCorrectnessAgent   determinism audit gate
                    └─→ ReviewerAgent                 static + semantic checks
                    │
             approved artifacts → docker build/run
                    │
                    ▼
            CorrectnessValidator (post-boot)
      reset fidelity + snapshot/restore round-trip
```

When enabled for a custom environment, `UserResearchAgent` reads the extracted application spec, the required original product name and URL, optional reference URLs, and a small web search when references are not provided. It synthesizes the target product's workflows, functionality, UI states, data, rules, RL observations, and edge cases. Raw pages are discarded inside the research task; backend, UI, RL, and review specialists receive only their relevant sections under a hard character budget. When disabled, the planner omits the research task and downstream specialists run with the application spec alone.

`TaskExecutor` runs independent tasks concurrently and waits on declared dependencies. Each task receives a scoped artifact channel. The A2A protocol records assignment, completion, failure, review, and artifact-availability messages with correlation IDs, without copying large generated files into message payloads. The reviewer blocks artifact writes when generated code or requirement coverage fails. A dedicated correctness specialist runs alongside it and blocks writes when generated code is nondeterministic — wall-clock reads, unseeded randomness, nondeterministic IDs, or a `/forge/reset` that fails to re-initialize the virtual clock and ID counters — while exempting telemetry event-envelope timestamps that never reach `/forge/state`. Once the container is built and running, `CorrectnessValidator` exercises the live endpoints to prove reset fidelity (two resets and a mutate→reset both return the byte-identical pristine baseline) and a snapshot→mutate→restore round-trip before the environment is accepted; any drift hard-fails the build. Local follow-up work, including reviewer-driven automatic repairs, is tracked in the git-ignored `TASKS.md`.

Generation prompts are grouped behind prompt catalog classes, and the shared LLM client appends an explicit Pydantic output contract to every structured call. `EnvGenConfig` centralizes model token budgets, research limits, context budgets, and reviewer excerpt sizes; each value can be changed through its `FORGE_ENVGEN_*`, `FORGE_RESEARCH_*`, `FORGE_SPECIALIST_*`, or `FORGE_REVIEW_*` environment variable. `GenerationErrorHandler` normalizes specialist failures and retains task/agent error records for orchestration and A2A diagnostics.

Agent execution and data collection are separate layers. Runtime agents choose actions, and `ForgeEnv` emits immutable snapshots through the storage-agnostic `TelemetrySink` protocol. Backend `EpisodeDataCollector` owns SQLite and JSONL persistence; runtime code does not import backend models or database libraries.

### Agent Runs & Data Collection

- **Five agent adapters** — `random`, `scripted:<path>`, `anthropic:<model>`, `openai:<model>`, `vllm:<model>`
- **AgentContext** — per-episode agent memory with a compact deterministic digest for prompt injection, stuck-vs-context-limit diagnosis, and automatic pruning of error spam and revisited-state noise
- **Trajectory recording** — every step's state, action, and reward persisted to JSONL and DB
- **Cross-run episode selection** — pick episodes from multiple runs, export as a single merged dataset
- **Parallel rollouts** — launch batched episode rollouts across any compiled environment from the global Rollouts page; `ParallelRolloutRunner` runs the same task across many isolated env copies concurrently (one fresh instance per rollout, millisecond start/teardown) and classifies each outcome as success, failure, partial success, or edge case so a single batch yields diverse training scenarios
- **Per-environment dashboard** — pass rate, average reward, step efficiency, termination-reason breakdown

### Observability & Replay

- **Live event feed** — real-time observability panel streamed from the running container
- **Unified per-run trace** — `AgentRunLogger` records the agent's LLM layer (prompt, chosen tool call, response) alongside every action, result, and state change as one ordered, step-correlated trace; persisted per run even when the run aborts mid-flight
- **Per-run loss analysis** — `LossAnalyzer` classifies why a run failed into a fixed seven-mode taxonomy (instruction-following, hallucination, tool-sequencing, early-stopping, context-loss, reward-hacking, surface-overfitting) from the run trace and verifier result, emits a per-run report with evidence and confidence, and aggregates modes across runs. A clean, correct run yields no failure modes
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

**Per-environment verifier composition** — `VerifierComposer` builds a configured `LayeredVerifier` for each task from its declared success/failure conditions and scenario ground truth, mapping them onto the five tiers (the LLM judge stays off by default). It then scores an episode's result into a `RewardBreakdown` under either mode: **binary** (full credit only when every tier passes) or **partial** (weighted per-tier mean, so a partially-correct trajectory earns graded credit). A right answer reached by an unauthorized side effect or the wrong tool order still fails.

**Reward-hacking audit** — `RewardHackingAuditor` is a separate audit agent that asks whether a passing verdict was *earned*: it flags passes with skipped milestones, suspiciously short episodes, redundant call patterns, and supports a pluggable LLM audit client. `RewardHackingAuditor.for_verifier(...)` inherits the milestone list straight from a `LayeredVerifier`.

### Interaction Contracts

Every environment declares which capabilities the agent has access to — the actions an agent can take, across every interaction modality — via `env.capabilities()`. Each capability validates actions against its schema *before* anything touches the environment, so a hallucinated tool, unknown endpoint, or out-of-bounds click never executes. Not every environment needs every modality; each advertises only the ones it attaches:

| Capability | Interacts with | Action shape | Schema enforces |
|---|---|---|---|
| **ToolUse** | API endpoints / functions of the environment | `{"type", …}` | tool exists, required params present, param types match |
| **MCPUse** | MCP server tools | `{"tool", "arguments"}` | tool exists, required arguments present and typed |
| **RESTUse** | HTTP endpoints | `{"method", "path", "input"}` | method+path is a declared endpoint, required params present |
| **ORPCUse** | Typed RPC procedures | `{"procedure", "input"}` | procedure exists, required input present and typed |
| **ComputerUse** | The VM / OS (Linux, macOS, Windows) | `{"action_type", …}` | allowed primitives (`exec`, `screenshot`), non-empty commands |
| **BrowserUse** | The browser | `{"action_type", …}` | allowed primitives (`click`, `type`, `press`, `navigate`, `scroll`), viewport bounds |

- Every `ForgeEnv` exposes ToolUse (`env.tool_use.execute(...)` is a schema-validated `step()`); attach the others with `EnvBuilder.with_mcp_use(...)` / `.with_rest_use(...)` / `.with_orpc_use(...)` / `.with_computer_use(...)` / `.with_browser_use(...)`
- `env.capability_surface()` returns `{modality: [ToolSpec]}` — the full set of actions the agent can take, grouped by modality, so every attached interface is discoverable through one tool surface
- CLI environments grant ComputerUse (`os="linux"`); browser environments route every agent action through BrowserUse

### Determinism

Environments are verified deterministic at creation and launch — same seed and same trajectory always produce the same observations *and* the same score:

- **Launch-time determinism check (mandatory)** — two identically-seeded rollouts are hashed (observations + rewards + termination flags); a mismatch raises `DeterminismError` and aborts the launch. Runs in the backend env loader, `forge run` / `forge export`, and `EnvBuilder.build()`. The check is non-bypassable — a non-reproducible environment fails to load
- **EnvBuilder + DeterminismConfig** — virtual clock, seeded RNG and UUIDs, canonical sorted-key JSON, float rejection (integers only), serialized transitions, network and filesystem guards inside the env, and fresh-universe startup (factory caches dropped every reset)
- **Seed control** — the seed threads end to end (`reset(seed)` → `POST /forge/reset {"seed": …}` → `STATE.seed_state(seed)`), so the same seed reproduces the same starting universe and a different seed produces a different-but-reproducible one; an unseeded reset restores the fixed baseline
- **Generated-app determinism contract** — custom LLM-generated apps must use a counter-based virtual clock (`forge_now()`) and sequential IDs (`_next_id()`) in place of wall-clock timestamps and random UUIDs, and build the universe from a `random.Random(seed)`; a static correctness specialist audits this before artifacts are written, and a post-boot `CorrectnessValidator` proves `/forge/reset` restores a byte-identical initial universe (rows, IDs, counters, DB included), that snapshot/restore round-trips, and that the same seed reproduces identical state while distinct seeds diverge — hard-failing the build on any violation
- **Replayable episodes** — every step records the tool call, emitted events, state diff, hashes, and reward; `replay_episode(env, seed, steps)` re-executes any recording and verifies every state hash and reward against it. Container/CLI/browser trajectories are written incrementally (each step flushed as it happens), so a run that crashes mid-episode still leaves a durable, replayable partial trace
- **Flake-free UI** — premade UIs ship a CSS no-motion override and browser sessions force `prefers-reduced-motion` + injected no-animation styles
- **SQLite as source of truth** — premade and generated apps persist state in SQLite; verification reads `/forge/state` (DB-backed), never the UI
- **Enforced separation of concerns** — architecture tests keep environment, agents, verifiers, and training code from importing across boundaries
- **Test-scenario-diversity gate** — a static analyzer (`tests/architecture/diversity_audit.py`) parses every test module and fails the suite if one asserts only the happy path; each behavior must pair its happy case with a negative case (invalid input / error path) and a false-positive guard (a look-valid input that must be rejected, detected via `pytest.raises`, a differential/exclusion assertion, or a rejection-named test)

### Security & Policy

- **PolicyEngine DSL** — policy and verifier expressions use a restricted AST evaluator instead of `eval`; violations block transitions and return 0.0 reward
- **Network and process isolation** — AST-based scanning blocks network modules, subprocess access, shell execution, and dynamic imports in generated envs (bypass with `FORGE_DEV_NETWORK=true`)
- **Generated-code validation** — compiler checks run in an isolated subprocess with time and output limits; generated paths are confined to the configured environment root
- **Credential-safe logging** — bearer values and URL query strings are redacted before HTTP, Docker pull, or worker errors reach logs; signed CDN URLs are never emitted intact
- **Local-only default** — `run.sh` binds the backend to `127.0.0.1` unless `FORGE_HOST` is explicitly changed
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

### Policy Training

Closing the RL loop, `forge train` turns Forge's *own* graded experience into a policy update — distinct from the benchmark, which zero-shot *evaluates* a base model. It consumes the exports above and produces a loadable checkpoint:

- **GRPO** over `grpo_rollouts.parquet` — rewards are mapped to group-relative advantages `(r − mean) / (std + eps)` across rollouts that share a prompt
- **DPO** over `preference_pairs.jsonl` — chosen/rejected labels are kept only where the chosen trajectory was graded strictly higher

The reward→signal mapping is a deterministic function of the grades already assigned, and a graded set with **no relative signal** (all rollouts scored the same, or every preference pair a tie) raises `NoTrainingSignalError` and writes no checkpoint — the training backend is never invoked. The heavy backend gates on `trl` + `transformers` and expects a GPU node. A finished run writes a `policy_checkpoint.json` manifest that runtime agents load via `forge.training.checkpoint.load_policy_agent`, so the same policy can collect → grade → export → train → reload.

```bash
forge train \
  --data <export_dir> \             # dir holding grpo_rollouts.parquet / preference_pairs.jsonl
  --base-model Qwen/Qwen2.5-3B \
  --output policy_checkpoint \
  --objective grpo                  # grpo | dpo
```

---

## Benchmark

Benchmark runs your selected environments against **their own compiled tasks**, collects episodes, and scores each environment on four quality metrics. Results are accessible from the **Benchmark** section in the top nav.

### Task Suite

Each benchmarked environment is run against the tasks it was compiled with — `CompiledTaskProvider` resolves them from the environment's compiler input (its `TaskTemplate`s) and maps them onto the benchmark's task shape. There is no fixed built-in suite; an environment with no compiled tasks is skipped rather than falling back to a curated one. Grading is unchanged — each generated environment is scored inside the container episode runner by its own reward function and verifiers.

The `--depth` / **Max difficulty** slider is a ceiling: only tasks with `difficulty ≤ depth` are included (difficulty is derived from how much a task asserts). Depth 1 runs the simplest tasks only; depth 5 runs all of them.

### Quality Metrics

| Metric | What it measures | Target |
|---|---|---|
| **State Coverage** | Fraction of state schema fields touched per step on average | ≥ 0.7 |
| **Reward Density** | Fraction of steps that produced a positive reward | ≥ 0.7 |
| **Dead-end Rate** | Fraction of episodes that terminated with no progress | ≤ 0.3 |
| **Action Diversity** | Unique endpoints / total endpoints called | ≥ 0.7 |

The report page colour-codes each value: green ≥ 0.7, amber 0.4–0.7, red < 0.4 (dead-end rate is inverted before thresholding).

### Web UI

The responsive Next.js control surface uses an industrial foundry visual system with active navigation, environment inventory telemetry, live status indicators, and accessible reduced-motion behavior.

| Page | What it does |
|---|---|
| **Run** | Select which active environments to benchmark, max difficulty (1–5), seeds per task, and output dir; launch with live log streaming and a progress bar. A snackbar prompts you if no environment is available or selected |
| **Report** | Table of quality metrics per environment for the most recent completed run; CSV download |
| **Transfer** | Stub — fine-tune a base model on collected data (GPU node required) |
| **Eval** | Stub — evaluate a fine-tuned checkpoint zero-shot on WebArena / WorkArena (eval harness required) |

### CLI

```bash
forge benchmark run \
  --domains my_env,other_env \      # comma-separated generated environment names
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
              ├── Custom:    planner → scoped specialist DAG → reviewer → docker build/run
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
    interaction.py     # Capability contracts: tool / MCP / REST / oRPC / computer / browser use
    verifier_composer.py  # Per-task LayeredVerifier composition + binary/partial scoring
    agent_logger.py    # Unified per-run trace (LLM calls + actions + state changes)
    loss_analysis.py   # Per-run failure-mode taxonomy + cross-run aggregation
    reward_hacking.py  # RewardHackingAuditor
    clustering.py      # FailureClusterer
  extraction/          # LLM pipeline, PII redactor, schemas
  compiler/
    generators/        # Jinja2 compiler, package builder
  envgen/              # LLM orchestration, container runtime, episode/CLI/browser runners
    agents/            # Backend, UI, assembly, telemetry, state, scenario, policy, reward, correctness, reviewer
    planning.py        # Typed task plans and dependency validation
    executor.py        # Dependency-aware specialist task execution
    a2a.py             # Typed Agent-to-Agent messages and scoped context permissions
    correctness_validator.py  # Post-boot reset-fidelity + snapshot/restore validation
    telemetry/         # Telemetry client and collectors
    container.py       # Docker build, run, start/stop, normalisation, mirror fallback
    tiered_reward.py   # TieredRewardEngine with partial credit and multi-method scoring
    ml_reward.py       # SentenceEmbeddingScorer, NGramScorer (ROUGE-L / BLEU)
  benchmark/
    task_suite.py      # Benchmark Task shape (resolved from each env's compiled tasks)
    compiled_tasks.py  # CompiledTaskProvider: an env's TaskTemplates → benchmark tasks
    data_collector.py  # Episode collection loop
    env_quality.py     # EnvQualityMetrics: coverage, reward density, dead-end rate, diversity
    report.py          # BenchmarkReport: paper-ready figures and summary tables
    transfer_pipeline.py  # Fine-tune stub (GPU required)
    _fine_tune.py      # fine_tune_model() entry point
    _eval.py           # evaluate_on_suite() entry point (eval harness required)
  training/            # Close the RL loop: train a policy from graded rollouts
    dataset.py         # Load grpo_rollouts.parquet / preference_pairs.jsonl exports
    reward_mapping.py  # Reward → GRPO advantage / DPO label (deterministic, no-signal guard)
    trainer.py         # PolicyTrainer: prepare signal → backend → PolicyCheckpoint
    checkpoint.py      # PolicyCheckpoint manifest + load_policy_agent()
    _backends.py       # GRPO/DPO backends gated on trl + transformers (GPU node)
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
        progress/      # Real-time planner, specialist, review, and Docker build progress
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
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
npm --prefix frontend install

# 2. Start everything
./run.sh        # Redis + Celery worker + backend (:8000) + frontend (:3000)

# 3. Stop everything
./kill.sh
```

The development runner configures Redis with a TCP backlog of 128 to match the default macOS kernel limit and avoid local startup warnings.

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |

**Run tests:**
```bash
uv run pytest
```

---

## Environment Variables

### Infrastructure

| Variable | Default | Description |
|---|---|---|
| `FORGE_GENERATED_ENVS_DIR` | `generated_envs` | Where compiled environments are written |
| `FORGE_DB_URL` | `sqlite:///./forge.db` | Backend database URL |
| `FORGE_SANDBOX_LIMIT` | `10` | Maximum active sandbox environments |
| `FORGE_HOST` | `127.0.0.1` | Backend bind host used by `run.sh` |
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

forge train \
  --data <export_dir> \              # dir with grpo_rollouts.parquet / preference_pairs.jsonl
  --base-model Qwen/Qwen2.5-3B \
  --output policy_checkpoint \
  --objective grpo                   # grpo | dpo — train a policy from graded rollouts

forge benchmark run \
  --domains my_env,other_env \       # comma-separated generated environment names
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
