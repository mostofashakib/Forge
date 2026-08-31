# Forge

**Sandbox environments for training AI agents on real-world apps.**

Forge lets you spin up isolated, observable app environments — Gmail-like email clients, Slack-like messaging, custom LLM-generated apps, raw Linux shells, or live browser sessions — and run RL agents inside them. Every action is logged, every state transition is verifiable, and every episode is exportable as a training dataset.

---

## What Forge Does

1. **Creates sandboxed app environments** — Docker containers running real apps (or realistic replicas) with full state access
2. **Runs agents inside them** — Random, scripted, or LLM-powered agents interact with the app via a clean API
3. **Records every step and grades each episode once** — Policy enforcement and state changes are written durably as they happen; one post-rollout evaluation produces the authoritative verifier verdict and reward
4. **Exports training data** — SFT pairs, DPO preference pairs, GRPO rollouts, failure datasets, and more
5. **Trains a policy on its own experience** — `forge train` turns graded rollouts into a GRPO or DPO update and writes a checkpoint the runtime agents can load back
6. **Measures held-out generalization** — Trains on an explicit environment split, evaluates only unseen environments, and records reproducible per-seed outcomes

The loop closes: generate environments → run agents → grade → export → train → evaluate
on environments the policy never trained on → reload the checkpoint and collect again.

## Example RL Tasks

Forge includes deterministic reference tasks in [`example_tasks/`](example_tasks/).
These examples demonstrate Slack and task-management environments, scripted
solutions, layered verifiers, and Harbor-based evaluation workflows. See the
[example tasks guide](example_tasks/README.md) for setup and usage.

---

## Environment Types

| Type | What runs | Good for |
|---|---|---|
| **Premade** | Pre-built Gmail or Slack replica | Ready-to-use evaluation; seeded with realistic emails, threads, and DMs |
| **Custom** | LLM-generated FastAPI app | Simulate any business app from a plain-English description |
| **CLI** | Ubuntu 22.04 shell | Shell scripting, sysadmin, package management tasks |
| **Browser** | Chromium + KasmVNC | Web automation, form filling, navigation |

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

`Environment` composes the concerns that describe the world. `EpisodeController`
stays outside because a benchmark, trainer, or rollout worker drives an environment;
`Verifier` remains separate from `Rubric` because deciding whether an objective was
met is different from assigning its reward.

Both `ForgeEnv` and container-backed environments implement this facade. Their hot
paths call the injected collaborators: reset uses `InitialStateProvider`, actions run
through `ExecutionBackend`, state is owned by `StateManager`, observations pass
through `ObservationEncoder`, final rewards use `Rubric`, and every completed step consults
`TerminationPolicy`. The reserved `submit` control action ends an episode without
being sent to a domain backend. Container episode controllers use the same facade rather than
duplicating reset, state, action, or termination plumbing with direct HTTP calls.
At reset, `TaskSource` selects an explicit task id or deterministically distributes
seeded episodes across its task set. `make_agent(..., environment=env)` binds that
selected task, the environment prompt, and full tool schemas to OpenAI, Anthropic,
or vLLM adapters, so the model-facing context cannot drift from the environment.

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
- **Headless or UI environments** — custom environments are API-only by default; opt into **With UI** on the creation form to also generate a browsable single-page app. A headless build skips the UI specialist entirely — no `ui.html`, no `/ui` route, and no UI review gate — so it builds faster and exposes only the RL surface. The toggle applies to custom environments; CLI is always headless, and Browser and Premade always ship a UI.
- **Custom environment generator** — describe any app in plain English; choose headless or with-UI; optionally enable the user researcher with an original product name and URL, then a prompt planner creates a dependency-aware task graph for dedicated backend, UI, telemetry, state-bridge, policy, reward, correctness, and review agents
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
          ├── ScenarioBuilderAgent    (seeded scenarios + source-bound verifier milestones)
          ├── PolicyAgent
          └── RewardAgent
                    │
                    ├─→ EnvironmentCorrectnessAgent   determinism audit gate
                    └─→ ReviewerAgent                 static + semantic checks
                    │
                    ▼
              RepairLoop  ──→ routes each finding back to the owning
                    │         specialist as a correction task, up to
                    │         FORGE_ENVGEN_MAX_REPAIR_ROUNDS rounds
                    ▼
             approved artifacts → docker build/run
                    │
                    ├─→ CorrectnessValidator (post-boot)
                    │     reset fidelity + snapshot/restore round-trip
                    └─→ PostGenerationValidator (post-boot)
                          exercises declared actions against the live
                          container and scores state-manifest coverage
```

When enabled for a custom environment, `UserResearchAgent` reads the extracted application spec, the required original product name and URL, optional reference URLs, and a small web search when references are not provided. It synthesizes the target product's workflows, functionality, UI states, data, rules, RL observations, and edge cases. Full raw pages are discarded inside the research task, but each retained source carries a bounded verbatim passage checked against the fetched document. Backend, UI, RL, and review specialists receive only their relevant sections and these verified evidence excerpts under a hard character budget. When disabled, the planner omits the research task and downstream specialists run with the application spec alone.

`ScenarioBuilderAgent` uses that evidence to bind every verifier-facing required action, forbidden action, and expected answer to a source when documentation justifies the constraint. A milestone source records the fetched document title, URL, and exact passage; a citation counts only when all three match the pruned research context. If no supplied passage supports the constraint, the generator must leave `source` null instead of inventing a citation. Legacy scenario files containing bare strings remain readable and are classified as unattributed.

Every generated `custom/scenarios.json` includes an `attribution_report`. It reports total milestones, source-attributed and model-invented counts and fractions, and analyzes the two groups separately. Source-attributed milestones are documented ground truth. Unattributed or unverifiable milestones are retained as generator findings rather than silently promoted to ground truth.

```json
{
  "scenarios": [
    {
      "scenario_id": "archive_inbox_message",
      "required_actions": [
        {
          "value": "archive_message",
          "source": {
            "title": "Acme Mail guide",
            "url": "https://docs.example.test/mail",
            "passage": "Archived messages leave the inbox."
          }
        }
      ]
    }
  ],
  "attribution_report": {
    "total_milestones": 3,
    "source_attributed_count": 2,
    "model_invented_count": 1,
    "source_attributed_fraction": 0.6666666667,
    "model_invented_fraction": 0.3333333333
  }
}
```

`TaskExecutor` runs independent tasks concurrently and waits on declared dependencies. Each task receives a scoped artifact channel. The A2A protocol records assignment, completion, failure, review, and artifact-availability messages with correlation IDs, without copying large generated files into message payloads. The reviewer blocks artifact writes when generated code or requirement coverage fails. A dedicated correctness specialist runs alongside it and blocks writes when generated code is nondeterministic — wall-clock reads, unseeded randomness, nondeterministic IDs, or a `/forge/reset` that fails to re-initialize the virtual clock and ID counters — while exempting telemetry event-envelope timestamps that never reach `/forge/state`. When either gate rejects the artifacts, `RepairLoop` does not fail the build immediately: `FindingRouter` attributes each review issue to the specialist that owns the offending artifact, `RepairPlanner` turns the attributed issues into typed correction tasks, and the executor re-runs only those specialists before re-review. It runs at most `FORGE_ENVGEN_MAX_REPAIR_ROUNDS` (default `2`) rounds, de-duplicates findings by fingerprint so an unfixable issue is not retried forever, and raises `UnrepairableFinding` when a finding cannot be attributed to any agent.

Once the container is built and running, `CorrectnessValidator` exercises the live endpoints to prove reset fidelity (two resets and a mutate→reset both return the byte-identical pristine baseline) and a snapshot→mutate→restore round-trip before the environment is accepted; any drift hard-fails the build. `PostGenerationValidator` then boots the same container, calls `/forge/reset`, exercises the declared actions, and checks the observed state against the `StateSchemaManifest` — reporting missing fields and a coverage score so a generated app that never populates part of its declared state surfaces before agents run against it.

Generation prompts are grouped behind prompt catalog classes, and the shared LLM client appends an explicit Pydantic output contract to every structured call. `EnvGenConfig` centralizes model token budgets, research limits, context budgets, and reviewer excerpt sizes; each value can be changed through its `FORGE_ENVGEN_*`, `FORGE_RESEARCH_*`, `FORGE_SPECIALIST_*`, or `FORGE_REVIEW_*` environment variable. `GenerationErrorHandler` normalizes specialist failures and retains task/agent error records for orchestration and A2A diagnostics.

Agent execution and data collection are separate layers. Runtime agents choose actions, and `ForgeEnv` emits immutable snapshots through the storage-agnostic `TelemetrySink` protocol. Backend `EpisodeDataCollector` owns SQLite and JSONL persistence; runtime code does not import backend models or database libraries.

### Agent Runs & Data Collection

- **Five agent adapters** — `random`, `scripted:<path>`, `anthropic:<model>`, `openai:<model>`, `vllm:<model>`
- **AgentContext** — per-episode agent memory with a compact deterministic digest for prompt injection, stuck-vs-context-limit diagnosis, and automatic pruning of error spam and revisited-state noise
- **Trajectory recording** — every step's state, action, and reward persisted to JSONL and DB
- **Post-episode objective scoring** — container and browser runners call `ObjectiveScorer` once on the final state; CLI uses its final tiered grader. Cheap state-hash and loop monitors may stop stuck runs without exposing grader feedback to the agent
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
- **Cross-run anomaly detection** — `POST /api/sandbox/{env}/detect` loads recent completed episodes, compares trajectories across runs, and returns severity-tagged findings in five categories (`reward_hacking`, `distribution_drift`, `policy_gaming`, `anomalous_pattern`, `reward_collapse`), surfaced on the per-environment Violations page next to the rule-based audit log
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

Six built-in verifier types compose into the final `EpisodeEvaluation` and its auditable `RewardBreakdown`:

| Verifier | Checks |
|---|---|
| `ExactStateVerifier` | Specific state field values |
| `EventVerifier` | Required events appeared in the trajectory |
| `TemporalVerifier` | Event ordering and timing constraints |
| `NegativeVerifier` | Forbidden events did not occur |
| `PolicyVerifier` | Python expressions against current state |
| `SemanticVerifier` | LLM-based semantic correctness (with embedding cache) |

**Layered verification** — `LayeredVerifier` composes five layers into one verdict: final-state checks, invariant milestone checks (none skipped, correct order), trajectory checks (necessary tool calls made, no unnecessary ones), LLM-as-judge rubrics for creative tasks, and negative checks for unintended side effects.

**Per-environment verifier composition** — `VerifierComposer` builds a configured `LayeredVerifier` for each task from its declared success/failure conditions and scenario ground truth, mapping them onto the five tiers (the LLM judge stays off by default). Provenance metadata does not affect runtime matching: the composer unwraps each milestone's `value` before checking action presence, order, and forbidden calls, while also accepting legacy strings and serialized milestone dictionaries. It then scores an episode's result into a `RewardBreakdown` under either mode: **binary** (full credit only when every tier passes) or **partial** (weighted per-tier mean, so a partially-correct trajectory earns graded credit). A right answer reached by an unauthorized side effect or the wrong tool order still fails.

**Reward-hacking audit** — `RewardHackingAuditor` is a separate audit agent that asks whether a passing verdict was *earned*: it flags passes with skipped milestones, suspiciously short episodes, redundant call patterns, and supports a pluggable LLM audit client. `RewardHackingAuditor.for_verifier(...)` inherits the milestone list straight from a `LayeredVerifier`.

**Reward ablation presets** — set `reward_preset` in an experiment YAML; both
`VerifierComposer` and `TieredRewardEngine` resolve the same named contract:

| Preset | Verifier and scoring behavior |
|---|---|
| `full_layered_partial` | All five verifier layers, weighted partial credit, and reward-hacking audit penalties |
| `binary_final_state` | Final-state checks only; reward is exactly 0 or 1, with no efficiency or partial-credit adjustment |
| `judge_only` | LLM judge score only; structured state, trajectory, and negative checks are excluded |
| `full_no_auditor` | Same layered partial reward as the full preset, while the auditor remains post-hoc and does not alter pass/reward |

### Verification Independence

Forge's environments are authored by an LLM. If the same model family then *graded* the agent, the grade would not be independent evidence — the generator would be marking its own homework. Forge's answer is structural: **the verdict is computed, not asked for.**

| Grading path | Who issues the verdict | Contaminable |
|---|---|---|
| Final-state checks | Assertions over recorded state | No |
| Invariant / milestone checks | Order and presence over the trajectory | No |
| Trajectory checks | Necessary vs. unnecessary tool calls | No |
| Negative checks | Forbidden side effects over the trajectory | No |
| Reward-hacking audit | Milestone, length, and call-pattern rules | No — the optional LLM audit client is off by default |
| LLM judge (`judge` layer, semantic checks) | A model | **Yes** |
| Objective progress scoring (reward shaping only) | A model | **Yes** |

Five of the seven paths never consult a model, and a sixth only does so when explicitly given a client. `VerifierComposer` leaves the judge layer off unless a task explicitly declares a semantic check, and the `binary_final_state` preset removes it entirely — so the default reward path is fully structural and needs no independence guarantee at all. This is the reason the layered verifier is built the way it is, not an incidental design choice.

For the paths that *do* consult a model, independence is configurable and enforced:

- **Separate judge client** — `FORGE_JUDGE_MODEL` / `FORGE_JUDGE_PROVIDER` configure grading independently of generation. Every grading call site (`TieredRewardEngine`, `ObjectiveScorer`, trajectory anomaly detection) resolves through `get_judge_client()`; generation keeps using `get_client()`. Unset, grading falls back to the generation model — a workable local default, and an explicitly non-independent one.
- **Family-level comparison** — `model_family()` collapses tiers, so grading Sonnet-authored environments with Haiku does *not* count as separation. Swapping tiers changes cost, not provenance.
- **Enforced before the run** — experiments declare `require_grader_independence` (default `true`). When a run would issue LLM verdicts under a judge from the generating family, `forge benchmark eval` raises `GraderContaminationError` before spending a single episode, and names the variable that fixes it.
- **Recorded in the result** — every `result.json` carries a `grading` block naming the generating models, the judge, and whether the pair was independent. A structural run records `llm_graded: false`, which is the *positive* claim: this number came from computed checks, not from a model's opinion.

```json
"grading": {
  "generator_models": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
  "generator_families": ["claude"],
  "judge_model": "gpt-4o",
  "judge_family": "gpt",
  "llm_graded": true,
  "llm_verdicts": 412,
  "independent": true
}
```

**Declared before, counted after.** The reward preset describes which verifier layers are enabled; it does not describe what a grading path actually does. Each post-episode `ObjectiveScorer` call increments `EpisodeResult.llm_verdicts`, and the observed count lands in the record.

The two must agree. A run that declared itself structural and then issued model verdicts raises instead of writing the record — an under-declaring grading path is a bug, and a result file that understates model involvement is worse than no file, because it is what a reader trusts when they cannot re-run the experiment.

**Pass/fail is computed, not inferred from stopping.** Held-out episodes are graded by `structural_verdict`, which composes a `LayeredVerifier` from the environment's own compiled `success_conditions` and `failure_conditions` and runs it against the recorded final state and trajectory. Reaching the right final state by a forbidden route still fails. Termination reasons such as `submitted`, `dead_end`, and `max_steps` never stand in for verifier success.

When a task carries no compiled ground truth, the final objective verdict is recorded as LLM-derived rather than quietly presented as computed ground truth: absent structural ground truth is *unknown*, never a free pass.

### Verdict Quorum

A `Jury` can vote on top of the structural verdict. Members implement one small interface and may be LLM-backed or deterministic, so a statistical test and a Gemini judge are the same kind of thing to the jury.

| Setting | Meaning |
|---|---|
| `FORGE_QUORUM_MODELS` | Comma-separated `provider:model` list. Members must come from different families, and none may share a family with the generator. Empty means a single judge. |
| `agreement_threshold` | Unanimity among voting members by default; `0.67` accepts 2-1 on a three-member jury |
| `max_abstention_rate` | Ceiling on undecided episodes (default `0.2`) |

Majority decides, but a split below the threshold is **indeterminate**: the episode leaves the pass-rate denominator instead of being counted as a failure. An abstaining member — a provider outage is missing evidence, not evidence of failure — is excluded from the agreement denominator rather than counted as a vote against.

Because excluding episodes shrinks the denominator, `abstention_rate` is reported beside `heldout_pass_rate` in every `result.json`, and a run exceeding `max_abstention_rate` **fails and writes no record**. Past that point the surviving episodes are a biased sample of the ones the jury found easy, and publishing the number would be worse than failing.

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

Environments are deterministic by default — same seed and same trajectory produce the same observations *and* the same score:

- **Launch-time determinism check (default)** — two identically-seeded rollouts are hashed (observations + rewards + termination flags); a mismatch raises `DeterminismError` and aborts the launch. It runs in the backend env loader, `forge run` / `forge export`, and `EnvBuilder.build()`
- **EnvBuilder + DeterminismConfig** — virtual clock, seeded RNG and UUIDs, canonical sorted-key JSON, float rejection (integers only), serialized transitions, network and filesystem guards inside the env, and fresh-universe startup (factory caches dropped every reset)
- **Seed control** — the seed threads end to end (`reset(seed)` → `POST /forge/reset {"seed": …}` → `STATE.seed_state(seed)`), so the same seed reproduces the same starting universe and a different seed produces a different-but-reproducible one; an unseeded reset restores the fixed baseline
- **Generated-app determinism contract** — custom LLM-generated apps must use a counter-based virtual clock (`forge_now()`) and sequential IDs (`_next_id()`) in place of wall-clock timestamps and random UUIDs, and build the universe from a `random.Random(seed)`; a static correctness specialist audits this before artifacts are written, and a post-boot `CorrectnessValidator` proves `/forge/reset` restores a byte-identical initial universe (rows, IDs, counters, DB included), that snapshot/restore round-trips, and that the same seed reproduces identical state while distinct seeds diverge — hard-failing the build on any violation
- **Replayable episodes** — every step records the tool call, emitted events, state diff, hashes, and reward; `replay_episode(env, seed, steps)` re-executes any recording and verifies every state hash and reward against it. Container/CLI/browser trajectories are written incrementally (each step flushed as it happens), so a run that crashes mid-episode still leaves a durable, replayable partial trace
- **Flake-free UI** — premade UIs ship a CSS no-motion override and browser sessions force `prefers-reduced-motion` + injected no-animation styles
- **SQLite as source of truth** — premade and generated apps persist state in SQLite; verification reads `/forge/state` (DB-backed), never the UI
- **Enforced separation of concerns** — architecture tests keep environment, agents, verifiers, and training code from importing across boundaries
- **UI-determinism gate** — `tests/architecture/test_ui_determinism.py` asserts every premade UI ships the `forge-no-motion` kill switch, that the browser runner disables motion on every page it opens, and that verification reads the DB rather than the rendered UI
- **Test-scenario-diversity gate** — a static analyzer (`tests/architecture/diversity_audit.py`) parses every test module and fails the suite if one asserts only the happy path; each behavior must pair its happy case with a negative case (invalid input / error path) and a false-positive guard (a look-valid input that must be rejected, detected via `pytest.raises`, a differential/exclusion assertion, or a rejection-named test)

For the determinism ablation only, set `FORGE_DETERMINISM=off` before building,
starting, training, and evaluating environments. This skips the correctness and
launch gates, swaps the virtual clock for wall time, stops applying experiment
seeds to runtime RNGs, UUIDs, policies, and training libraries, and omits seeds
from container resets. Existing containers must be recreated so they receive the
flag. Omitting the variable (or setting it to `on`) preserves the default behavior.

### Security & Policy

- **PolicyEngine DSL** — policy and verifier expressions use a restricted AST evaluator instead of `eval`; violations block transitions and return 0.0 reward
- **Network and process isolation** — AST-based scanning blocks network modules, subprocess access, shell execution, and dynamic imports in generated envs (bypass with `FORGE_DEV_NETWORK=true`)
- **Generated-code validation** — compiler checks run in an isolated subprocess with time and output limits; generated paths are confined to the configured environment root
- **Credential-safe logging** — bearer values and URL query strings are redacted before HTTP, Docker pull, or worker errors reach logs; signed CDN URLs are never emitted intact
- **Local-only default** — `run.sh` binds the backend to `127.0.0.1` unless `FORGE_HOST` is explicitly changed
- **PII redaction** — strips emails, phone numbers, and SSNs from LLM input before code generation
- **RBAC observation filtering** — removes or restricts state fields per role, applied in `reset()` and `step()`
- **Policy Violation Viewer** — global filterable table of violations by environment, episode, and severity

### Environment Customization

A generated environment package can override compiled behaviour without editing generated code. Any `custom/*.py` file in the package is imported at load time and its decorated functions are registered by `CustomizationLoader`:

| Hook | Overrides |
|---|---|
| `@override_transition(action_name)` | The transition function for one action |
| `@verifier(task_name)` | The verifier for one task |
| `@reward(task_name)` | The reward function for one task |
| `@observation_transform(name)` | The observation shown to the agent |
| `@policy_rule(name)` | An additional policy rule evaluated on every step |

`EnvConfig` carries the per-environment knobs the UI's **Config** page edits — `RewardConfig` (base success, step penalty, policy-violation penalty, invalid-action penalty, reward clamps, semantic weight) and `ObservationConfig` (mode, actor role, visible/hidden entities for RBAC filtering). Overrides live in the package's `custom/` directory, which is included in the runnable source export.

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
| **RL Trajectories** | `grpo_rollouts.parquet` | Forge batch-GRPO, custom RL pipelines |
| **Failure Dataset** | `failure_dataset.jsonl` | Contrastive training, red-teaming |
| **Raw Trajectories** | `trajectories.jsonl` | Custom pipelines |
| **Rewards** | `rewards.jsonl` | Analysis, custom reward models |
| **Verifier Results** | `verifier_results.jsonl` | Debugging, custom reward models |

All environment-specific episode results convert to the shared
`forge.contracts.RolloutRecord`. Runtime collectors, Parquet export, and training
loaders therefore exchange one rollout shape while retaining richer controller
results internally.

**Runnable source export** — separately from the datasets, any generated environment can be downloaded as a self-contained zip (`GET /api/envs/{env_name}/download`, or the download action on the environment page). `build_source_bundle` packages the generated app, `container_env.py`, `reward_fn.py`, `state_schema.json`, and any `custom/` overrides alongside a README and a `docker-compose.yml`, while filtering runtime artifacts (`episodes/`, `__pycache__`, `.pyc`). The result runs with one command on a machine that has never seen Forge, so an environment can be published, reviewed, or archived independently of this platform.

### Policy Training

Closing the RL loop, `forge train` turns Forge's *own* graded experience into a policy update, then the benchmark evaluates that checkpoint on disjoint internal held-out environments. Training consumes the exports above and produces a loadable checkpoint:

- **Clipped batch-GRPO update** over `grpo_rollouts.parquet` — rewards become group-relative advantages `(r − mean) / (std + eps)`, behavior-policy token log-probabilities are frozen before the update, and training applies the clipped current/old policy ratio with a sampled KL penalty
- **TRL DPO** over `preference_pairs.jsonl` — chosen/rejected labels are kept only where the chosen trajectory was graded strictly higher and trained with `DPOTrainer`

The reward→signal mapping is a deterministic function of the grades already assigned, and a graded set with **no relative signal** (all rollouts scored the same, or every preference pair a tie) raises `NoTrainingSignalError` and writes no checkpoint — the training backend is never invoked. Install the optional GPU stack with `uv sync --extra training`. A finished run writes a `policy_checkpoint.json` manifest that runtime agents load via `forge.runtime.policy_loader.load_policy_agent`, so the same policy can collect → grade → export → train → reload.

For repeated policy improvement, `PolicyIterationLoop` supplies the missing feedback
edge: an injected collector exports graded experience, `PolicyTrainer` updates the
policy, the runtime loader binds the checkpoint back to its environment, and Forge
collects again with the updated agent. Multiple iterations continue training from
the preceding checkpoint rather than restarting from the original base model.

```bash
forge train \
  --data <export_dir> \             # merged graded exports with env_name metadata
  --experiment experiments/internal_heldout.yaml \
  --seed 0 \
  --output policy_checkpoint \
  --objective grpo                  # grpo | dpo
```

Experiment files declare `{train_envs, heldout_envs, reward_preset, base_model, seeds}`
and may set `determinism_repeats` (default `2`).
Training reads the base model and train split from the YAML and filters out all held-out
records. The checkpoint records the exact experiment, split, and seed.

```yaml
# experiments/internal_heldout.yaml
train_envs:
  - email_train
  - crm_train
heldout_envs:
  - calendar_heldout
reward_preset: full_layered_partial
base_model: Qwen/Qwen2.5-3B-Instruct
seeds: [0, 1, 2]
determinism_repeats: 2
require_grader_independence: true
```

The lists must be non-empty, unique, and disjoint. Run `forge train` once for each
declared seed. Export rows must contain `env_name`; older Forge exports are also
supported when their prompt contains an `Environment:` line.

---

## Benchmark

Benchmark runs your selected environments against **their own compiled tasks**, collects episodes, and scores each environment on four quality metrics. Results are accessible from the **Benchmark** section in the top nav.

### Task Suite

Each benchmarked environment is run against the tasks it was compiled with — `CompiledTaskProvider` resolves them from the environment's compiler input (its `TaskTemplate`s) and maps them onto the benchmark's task shape. There is no fixed built-in suite. Internal evaluation fails if a held-out environment has no compiled tasks, rather than silently weakening the denominator. Grading is unchanged — each generated environment is scored inside the container episode runner by its own reward function and verifiers.

For policy generalization, `forge benchmark eval` loads `policy_checkpoint.json` and
the same experiment YAML used for training. It refuses mismatched configs or leaked
train environments, runs only `heldout_envs`, and writes
`runs/<id>/result.json` with the config, seed, held-out pass rate, reward-hacking
rate, and reward variance.

The same evaluation is available in **Benchmark → Eval**. Choose **Forge native**,
enter the checkpoint and experiment paths, and start the run to stream worker output
and inspect the result metrics in the UI. **Harbor** is available as an optional
evaluation engine for local Harbor task directories; it remains outside Forge's base
dependencies. Enable it with `./example_tasks/run.sh setup`, then select the task,
agent, and model from the Eval page.

```bash
# 1. Train one declared seed using only train_envs.
forge train \
  --data exports \
  --experiment experiments/internal_heldout.yaml \
  --seed 0 \
  --output policy_checkpoint

# 2. Start every environment listed in heldout_envs, then evaluate the checkpoint.
forge benchmark eval \
  --checkpoint policy_checkpoint \
  --experiment experiments/internal_heldout.yaml \
  --runs-dir runs
```

The evaluator requires each held-out environment to be running and to have compiled
tasks. It rejects checkpoints missing experiment metadata, checkpoints produced from
a different config, undeclared seeds, and any train/held-out leakage.

### Policy Evaluation Metrics

| Metric | Definition |
|---|---|
| **Held-out pass rate** | Successful compiled-task episodes divided by all evaluated held-out episodes |
| **Reward-hacking rate** | Fraction of evaluated episodes flagged by `RewardHackingAuditor` |
| **Reward variance** | Mean within-task population variance across repeated trajectories with the identical environment, task, and seed |

Each seeded evaluation writes the paper-facing record below. The run ID is carried
from `policy_checkpoint.json`, linking the training checkpoint to its evaluation.

```text
runs/<run-id>/
├── eval/<environment>/<task>/seed_<seed>_repeat_<n>.jsonl
└── result.json
```

```json
{
  "config": {
    "train_envs": ["email_train", "crm_train"],
    "heldout_envs": ["calendar_heldout"],
    "reward_preset": "full_layered_partial",
    "base_model": "Qwen/Qwen2.5-3B-Instruct",
    "seeds": [0, 1, 2],
    "determinism_repeats": 2,
    "require_grader_independence": true
  },
  "seed": 0,
  "determinism": "on",
  "heldout_pass_rate": 0.72,
  "reward_hacking_rate": 0.03,
  "reward_variance": 0.08,
  "grading": {
    "generator_models": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
    "generator_families": ["claude"],
    "judge_model": "gpt-4o",
    "judge_family": "gpt",
    "llm_graded": true,
    "llm_verdicts": 412,
    "independent": true
  }
}
```

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
| **Transfer** | Reserved for a future external transfer benchmark |
| **Eval** | Evaluate a policy checkpoint on the declarative internal held-out split |

### CLI

```bash
forge benchmark run \
  --domains my_env,other_env \      # comma-separated generated environment names
  --depth 5 \
  --seeds 5 \
  --output benchmark_results

forge benchmark report --output benchmark_results

forge benchmark eval \
  --checkpoint ./policy_checkpoint \
  --experiment experiments/internal_heldout.yaml
```

---

## Status & Known Gaps

Everything documented above is implemented and covered by the test suite. Three
things are deliberately *not* — they are wired end to end but raise
`NotImplementedError` rather than silently degrading:

| Gap | Where | Why it matters |
|---|---|---|
| **External transfer benchmark** | `forge/benchmark/transfer_pipeline.py`, `_fine_tune.py` | Generalization is currently measured only on Forge's own held-out environments. Transfer to an independent harness (WebArena / WorkArena) is the claim that would separate "the split held" from "the training actually taught the policy something about real apps" |
| **SFT fine-tuning entry point** | `forge/benchmark/_fine_tune.py` | `forge train` supports GRPO and DPO; the SFT path that the transfer pipeline needs is still a stub |
| **Distributed parallel runs** | worker layer | `ParallelRolloutRunner` parallelizes within one host; multi-worker distribution with deterministic seed assignment, backpressure, and run-level aggregation is not built |

There are also **no published numbers**. The evaluation protocol, metrics, and
result-record format exist; no reference run has been executed and committed, so
the repository currently demonstrates capability rather than results. See
[TASKS.md](TASKS.md) for the ranked work queue.

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
              │           Environment facade (Gymnasium-compatible)
              │           ┌─────────────────────────────────┐
              │           │  reset()                        │
              │           │    InitialStateProvider         │
              │           │    StateManager                 │
              │           │    ObservationEncoder (RBAC)    │
              │           │                                 │
              │           │  step(action)                   │
              │           │    ActionValidator              │
              │           │    PolicyEngine  ──→ AuditLog   │
              │           │    ExecutionBackend             │
              │           │    Verifier → Rubric            │
              │           │    TerminationPolicy            │
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
  contracts/           # The interfaces every environment family implements
    types.py           # Shared shapes: Task, Action, Observation, ToolSpec, results
    environment.py     # Environment facade composing ten of the eleven contracts
    dataset.py         # TaskSource; initial_state.py, prompting.py, tools.py
    observation.py     # ObservationEncoder; backend.py, state.py, transport.py
    reward.py          # Verifier and Rubric; termination.py, episode.py
    rollout.py         # RolloutRecord: the shared collector → exporter → trainer record
  runtime/             # Gymnasium env, state, trajectory, verifiers, agents
    prompting.py       # ForgeAgentPromptTemplate: the default PromptTemplate
    tools.py           # Spec / Capability / OpenAPI ToolProviders
    tasks.py           # Seeded, reproducible task selection from a TaskSource
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
    artifact_bus.py    # Async artifact publish/await between dependent specialists
    repair.py          # RepairLoop: route review findings back to owning specialists
    correctness_validator.py  # Post-boot reset-fidelity + snapshot/restore validation
    post_generation_validator.py  # Post-boot action exercise + state-manifest coverage
    source_bundle.py   # Package a generated env as a standalone runnable zip
    objective.py       # ObjectiveScorer: LLM 0–1 progress score toward a stated goal
    error_handling.py  # GenerationErrorHandler: normalized specialist failures
    episode_runner.py  # Container episode loop; cli_runner.py / browser_runner.py
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
    transfer_pipeline.py  # Deferred external transfer-benchmark boundary
    _fine_tune.py      # fine_tune_model() entry point
    _eval.py           # checkpoint-backed internal held-out evaluation + result records
  experiments.py       # declarative experiment and per-run result contracts
  training/            # Close the RL loop: train a policy from graded rollouts
    dataset.py         # Load grpo_rollouts.parquet / preference_pairs.jsonl exports
    reward_mapping.py  # Reward → GRPO advantage / DPO label (deterministic, no-signal guard)
    trainer.py         # PolicyTrainer: prepare signal → backend → PolicyCheckpoint
    checkpoint.py      # Serializable PolicyCheckpoint manifest
    loop.py            # Collect → train → reload → recollect policy iteration
    _backends.py       # Offline GRPO and TRL DPO gradient updates (GPU node)
  customization/       # Per-env overrides: decorator hooks, EnvConfig, loader
  schema/              # StateSchemaManifest and related schemas
  settings.py          # Process-wide settings: determinism mode, seeds, paths, URLs
  reward_presets.py    # Canonical reward-ablation presets shared by every reward path
  grading_provenance.py  # Generator/grader independence: model families, enforcement, record
  logging_utils.py     # Credential-safe redaction for logs
  paths.py             # Confined-path helpers for generated-env file access
  cli/
    main.py            # forge CLI: compile, validate, run, replay, diagnose, benchmark *
backend/
  app/
    api/               # FastAPI routers: sandbox, envs, episodes, agent_runs, synthetic,
    │                  #   evaluate, exports, audit, rollouts, detect, compile, benchmark
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
      transfer/        # Deferred external transfer benchmark
      eval/            # Internal checkpoint evaluation on held-out environments
    environments/
      new/             # 4-option landing page (CLI / Browser / Custom / Premade)
        custom/        # Prompt form + optional user-research toggle
        premade/       # Gmail / Slack picker
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
  benchmark/           # Task suite, quality metric, and internal held-out eval tests
  training/            # Dataset loading, reward mapping, trainer, checkpoint tests
  cli/                 # forge CLI command tests (run determinism, train, replay)
  customization/       # Hook registry, loader, and config tests
  gmail_env/           # Premade Gmail determinism, transition, and verifier tests
  architecture/        # Separation-of-concerns, UI-determinism, and test-diversity gates
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
uv run pytest                                  # full suite (1,526 tests)
uv run pytest tests/architecture               # boundary, UI-determinism, and diversity gates only
uv run pytest tests/runtime tests/envgen       # kernel + generation pipeline
```

The full suite completes in about 12 seconds — LLM calls are mocked and no container is
built. `tests/architecture` doubles as a review gate: it fails the build on cross-layer
imports, animated premade UIs, and tests that assert only the happy path.

---

## Environment Variables

### Infrastructure

| Variable | Default | Description |
|---|---|---|
| `FORGE_GENERATED_ENVS_DIR` | `generated_envs` | Where compiled environments are written |
| `FORGE_DB_URL` | `sqlite:///./forge.db` | Backend database URL |
| `FORGE_SANDBOX_LIMIT` | `10` | Maximum active sandbox environments |
| `FORGE_HOST` | `127.0.0.1` | Backend bind host used by `run.sh` |
| `FORGE_DETERMINISM` | `on` | Set to `off` only for the determinism ablation; disables virtual time, runtime/training seeding, and determinism gates |
| `FORGE_DEV_NETWORK` | `false` | Set to `true` to bypass network isolation in generated envs |
| `FORGE_DISABLE_PREWARM` | unset | Set to `1` to skip base-image pre-warm on worker boot |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for Celery and build/benchmark progress pub/sub; `run.sh` replaces this with a runtime-generated authenticated URL |
| `FORGE_REDIS_PASSWORD` | generated at startup | Optional local Redis password override used by `run.sh`; must be at least 32 hexadecimal characters and is never logged |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL used by the frontend |

### Container Images & Resource Limits

| Variable | Default | Description |
|---|---|---|
| `FORGE_PYTHON_BASE_IMAGE` | `python:3.12-slim` | Base image every generated custom environment is built on |
| `FORGE_CLI_IMAGE` | `ubuntu:22.04` | Image used for CLI environments |
| `FORGE_BROWSER_IMAGE` | `lscr.io/linuxserver/chromium:latest` | Image used for browser environments |
| `FORGE_CONTAINER_MEMORY` | `1g` | Memory cap for generated custom containers |
| `FORGE_CLI_MEMORY` | `1g` | Memory cap for CLI containers |
| `FORGE_BROWSER_MEMORY` | `2g` | Memory cap for browser containers |
| `FORGE_CONTAINER_NANO_CPUS` | `1000000000` | CPU quota in nano-CPUs (1e9 = 1 core) |
| `FORGE_CONTAINER_PIDS` | `256` | PID limit per container |

### Generation Budgets

`EnvGenConfig` centralizes every model token budget, research limit, context budget, and reviewer excerpt size; each field maps to one environment variable. The full list with defaults lives in [.env.example](.env.example) — the ones most often tuned:

| Variable | Default | Description |
|---|---|---|
| `FORGE_ENVGEN_MAX_REPAIR_ROUNDS` | `2` | Reviewer-driven repair rounds before the build fails |
| `FORGE_ENVGEN_CAPABLE_TOKENS` | `8192` | Output budget for capable-tier generation calls |
| `FORGE_ENVGEN_TELEMETRY_TOKENS` | `32768` | Output budget for the telemetry specialist |
| `FORGE_RESEARCH_SEARCH_RESULTS` | `3` | Pages fetched by the user researcher when no reference URLs are given |
| `FORGE_RESEARCH_DOCUMENT_CHARS` | `20000` | Per-document character cap on fetched research |
| `FORGE_SPECIALIST_CONTEXT_CHARS` | `12000` | Hard character budget on the context handed to each specialist |
| `FORGE_REVIEW_FILE_CHARS` | `16000` | Per-file excerpt size the reviewer sees |

### LLM Provider

All LLM calls go through a single `get_client()` factory — swap providers or models without touching code.

| Variable | Default | Description |
|---|---|---|
| `FORGE_LLM_PROVIDER` | `anthropic` | LLM backend. Supported: `anthropic`, `ollama` |
| `FORGE_LLM_MODEL` | `claude-haiku-4-5-20251001` | Standard-tier model (faster, cheaper) |
| `FORGE_LLM_MODEL_CAPABLE` | `claude-sonnet-4-6` | Capable-tier model (code generation, complex reasoning) |
| `ANTHROPIC_API_KEY` | — | Required when `FORGE_LLM_PROVIDER=anthropic` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `FORGE_JUDGE_PROVIDER` | falls back to `FORGE_LLM_PROVIDER` | Provider used for LLM **grading** only |
| `FORGE_JUDGE_MODEL` | falls back to `FORGE_LLM_MODEL` | Model used for LLM **grading** only. Set this to a model outside the generating family to make LLM-graded runs independent — see [Verification Independence](#verification-independence) |

**Run fully locally with Ollama:**
```bash
FORGE_LLM_PROVIDER=ollama FORGE_LLM_MODEL=gemma4:12b ./run.sh
```

---

## CLI

```bash
forge compile --input spec.json --output generated_envs   # Extract + compile an environment
forge validate generated_envs/<env_name>                  # Tests + override validation on a package
forge run --env <env_name> --task <verifier_id> \
  --seed 42 --steps 10               # Run one episode with a seeded random policy
forge export --env <env_name> --seed 42 --out exports     # Run + write the trajectory as JSONL
forge replay <episode_id> [--json]   # Replay a recorded episode (ep_* gym or cep_* container)
forge diagnose <env_name> [--json]   # Analyse episode quality across all runs

forge train \
  --data <export_dir> \              # dir with grpo_rollouts.parquet / preference_pairs.jsonl
  --experiment experiments/internal_heldout.yaml \
  --seed 0 \
  --output policy_checkpoint \
  --objective grpo                   # grpo | dpo — train a policy from graded rollouts

forge benchmark run \
  --domains my_env,other_env \       # comma-separated generated environment names
  --depth 5 \                        # max difficulty (1=easy only, 5=all tasks)
  --seeds 5                          # episodes per task
forge benchmark report               # generate summary tables from collected results
forge benchmark eval \
  --checkpoint ./policy_checkpoint \
  --experiment experiments/internal_heldout.yaml
forge benchmark transfer             # deferred — raises NotImplementedError (see Roadmap)
```

`forge run` and `forge export` drive the environment with a seeded random policy
(`seeded_random_policy`), not an LLM adapter; LLM/scripted agents run through the
backend agent-run API and the **Agent** page. Both commands run the launch-time
determinism check before the first `reset()`.
