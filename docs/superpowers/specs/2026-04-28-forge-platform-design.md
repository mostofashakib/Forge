# Forge Platform Design

**Date:** 2026-04-28
**Status:** Approved
**Scope:** Full platform — all seven milestones

---

## 1. Product Summary

Forge is a self-hosted web platform that converts a natural language prompt (with optional attachments) into a complete, runnable Gymnasium-compatible RL environment. Developers can provide an OpenAPI spec, database schema, workflow policy, or sample state as attachments. The platform generates a correct scaffold covering 90% of the environment; developers customize the remaining 10% through stable extension points without touching generated files.

Target workflows: Gmail triage, Slack routing, Jira backlog management, Salesforce CRM, Zendesk support, internal dashboard operations.

---

## 2. Core Architecture

### 2.1 Layers

```
Web App (Next.js)
  prompt input · compiler review · episode replay · dashboard

Backend API (FastAPI)
  ingestion · project mgmt · compile jobs · rollout service · replay service

LLM Extraction Layer
  EntityExtractor · ActionInferencer · PolicyParser · TaskGenerator · PermissionInferencer
  → CompilerInput (validated JSON, shown to developer for approval)

Deterministic Compiler (Jinja2 templates, no LLM)
  StateModelGenerator · ActionSchemaGenerator · TransitionGenerator
  VerifierGenerator · RewardGenerator · TaskTemplateGenerator
  PolicyGenerator · GymWrapperGenerator
  → EnvPackage (Python files + YAML configs)

Runtime Kernel (Python / Gymnasium)
  ForgeEnv · RuntimeContext · StateStore · TransitionEngine
  VerifierEngine · RewardEngine · TrajectoryStore

Storage
  PostgreSQL · Redis · Docker volume (local filesystem)
```

### 2.2 Compiler Approach

**Approach B — Deterministic Compiler + LLM for Inference.**

The LLM does understanding only: extracting entities, inferring actions, parsing policy prose. All code and config generation uses deterministic Jinja2 templates. Same `CompilerInput` always produces identical output regardless of LLM provider.

The critical trust boundary is the **Compiler Review UI**: the developer sees and approves the `CompilerInput` JSON before any generation runs. This is where they correct the 10% the LLM got wrong before it propagates into generated code.

### 2.3 Hybrid Input

Prompt alone is sufficient for simple environments (LLM infers everything from text). Attachments (OpenAPI YAML, JSON schema, policy docs, sample state) augment or override specific sections of the extracted `CompilerInput`. Both paths produce the same `CompilerInput` structure; they are equals, not primary/secondary.

### 2.4 Deployment

Self-hosted via Docker Compose. Single organization, JWT auth, project-scoped permissions. Model: users → projects → environments. No cross-project visibility, no billing layer.

---

## 3. Repository Structure

```
forge/                          # top-level Python package
  runtime/
    env.py                      # ForgeEnv (gym.Env subclass)
    state.py                    # StateStore
    action.py                   # ActionValidator
    transition.py               # TransitionEngine base
    verifier.py                 # VerifierEngine + all verifier types
    reward.py                   # RewardEngine
    trajectory.py               # TrajectoryStore
    snapshot.py                 # StepSnapshot schema
    diff.py                     # StateDiff engine
    policy.py                   # PolicyEngine
    context.py                  # RuntimeContext (clock, RNG, ID gen)
  compiler/
    openapi_parser.py
    schema_analyzer.py
    action_schema_generator.py
    state_model_generator.py
    transition_generator.py
    verifier_generator.py
    reward_generator.py
    package_builder.py
    templates/                  # Jinja2 templates
  extraction/
    entity_extractor.py
    action_inferencer.py
    policy_parser.py
    task_generator.py
    permission_inferencer.py
    schemas.py                  # CompilerInput Pydantic model
  cli/
    main.py                     # forge init / compile / validate / run / replay / export
  workers/
    rollout_worker.py
    agent_adapters/
      openai_adapter.py
      anthropic_adapter.py
      vllm_adapter.py
      random_policy.py
      scripted_policy.py
backend/
  app/
    main.py
    api/
    services/
      ingestion_service.py
      compiler_service.py
      rollout_service.py
      replay_service.py
    db/
    models/
frontend/
  app/
    dashboard/
    environments/
    replay/
    compiler-review/
generated_envs/                 # compiler output, gitignored
examples/
  gmail_env/                    # hand-written reference implementation
  zendesk_env/
  jira_env/
tests/
  runtime/
  compiler/
  generated_envs/
docs/
```

---

## 4. Runtime Kernel

### 4.1 Core Classes

```python
class ForgeEnv(gym.Env):
    def __init__(
        self,
        env_spec: EnvironmentSpec,
        initial_state_factory: InitialStateFactory,
        transition_engine: TransitionEngine,
        verifier_engine: VerifierEngine,
        reward_engine: RewardEngine,
        telemetry: TelemetryClient,
    ): ...

    def reset(self, seed: int | None = None, options: dict | None = None) \
        -> tuple[dict, dict]: ...

    def step(self, action: dict) \
        -> tuple[dict, float, bool, bool, dict]: ...
```

### 4.2 Determinism Contract

`RuntimeContext` owns all non-determinism sources. Real `uuid4()`, `datetime.now()`, and `random.random()` are banned from the kernel. A linter check in the test suite enforces this.

```python
ctx = RuntimeContext(seed=42)
ctx.clock.now()                  # simulated timestamp, advances per step
ctx.rng.random()                 # seeded random, reproducible
ctx.id_generator.next("email")   # "email_0001", "email_0002", ...
```

Same seed + same action sequence = identical trajectory hash. This is the M1 acceptance criterion.

### 4.3 Step Snapshot

Every `step()` produces and persists:

```json
{
  "episode_id": "ep_abc",
  "step_index": 3,
  "state_hash_before": "sha256:...",
  "state_hash_after":  "sha256:...",
  "action": { "type": "reply_email", "params": {} },
  "events": [],
  "reward": 0.7,
  "verifier_results": [],
  "diff": { "added": {}, "changed": {}, "removed": {} }
}
```

### 4.4 State Diff

Flat key-path format for replay UI rendering:

```json
{
  "changed": {
    "emails.e_1.labels": { "before": ["inbox"], "after": ["inbox", "refund"] }
  },
  "added": { "emails.e_9": { "id": "e_9", "thread_id": "t_1" } },
  "removed": {}
}
```

### 4.5 Invalid Action Handling

Invalid actions return a structured error, do not mutate state, and do not terminate the episode unless `max_invalid_actions` is exceeded:

```json
{ "error": "INVALID_ACTION", "code": "ENTITY_NOT_FOUND", "detail": "thread_id t_99 not found" }
```

### 4.6 Gmail Reference Environment

`examples/gmail_env/` is the M1 acceptance test and the reference implementation for developers.

Entities: `user`, `email`, `thread`, `label`
Actions: `reply_email`, `send_email`, `archive_email`, `apply_label`, `mark_read`, `escalate_thread`
Tasks: reply to customer, label urgent request, archive newsletter, escalate billing complaint

---

## 5. LLM Extraction Layer

### 5.1 Pipeline

The LLM runs as a multi-step chain. Each step has a focused job and produces a validated Pydantic object. Failed schema validation retries with the error as feedback (max 3 attempts). No step writes Python.

```
Input (prompt + attachments)
  │
  ├─→ EntityExtractor       → List[EntityDef]
  ├─→ ActionInferencer      → List[ActionDef]       (uses EntityDefs)
  ├─→ PolicyParser          → List[PolicyRule]      (uses EntityDefs + ActionDefs)
  ├─→ TaskGenerator         → List[TaskTemplate]    (uses all above)
  └─→ PermissionInferencer  → List[PermissionRule]  (uses EntityDefs + ActionDefs)
        │
        ▼
    CompilerInput (validated JSON)
```

Each extractor uses structured output (tool use / JSON mode). The LLM is provider-agnostic: OpenAI, Anthropic, or local vLLM, configured per deployment.

### 5.2 CompilerInput Schema

```json
{
  "project_name": "zendesk_support_env",
  "domain": "support",
  "entities": [
    {
      "name": "ticket",
      "primary_key": "id",
      "fields": [
        { "name": "status", "type": "enum", "values": ["open", "pending", "solved", "closed"] },
        { "name": "priority", "type": "enum", "values": ["low", "normal", "high", "urgent"] }
      ]
    }
  ],
  "actions": [
    {
      "name": "add_ticket_comment",
      "params": [
        { "name": "ticket_id", "type": "string" },
        { "name": "body", "type": "string" },
        { "name": "public", "type": "boolean", "default": true }
      ],
      "mutates": ["ticket.comments"],
      "requires_permission": ["ticket.comment"]
    }
  ],
  "policies": [
    {
      "id": "no_close_without_response",
      "condition": "ticket.status == 'open'",
      "forbidden_actions": ["close_ticket_without_comment"]
    }
  ],
  "tasks": [
    {
      "name": "resolve_refund_ticket",
      "description": "Resolve a refund ticket according to policy.",
      "success_conditions": [
        "ticket.status == 'solved'",
        "ticket.tags contains 'refund'",
        "latest_comment contains apology_or_acknowledgment"
      ],
      "failure_conditions": [
        "refund_promised_without_policy_match"
      ]
    }
  ],
  "permissions": []
}
```

### 5.3 Compiler Review UI

Shown to the developer after LLM extraction, before any code generation. Editable. Approving triggers the deterministic compiler.

```
┌─────────────────────────────────────────┐
│ Compiler Review — zendesk_support_env   │
│                                         │
│ Entities (4)          ✓ approved        │
│  ticket  user  comment  attachment      │
│                                         │
│ Actions (6)           ✏ 1 needs review  │
│  add_comment  update_status  assign...  │
│                                         │
│ Policies (3)          ✓ approved        │
│ Tasks (2)             ✓ approved        │
│                                         │
│           [Edit]  [Approve & Generate]  │
└─────────────────────────────────────────┘
```

---

## 6. Deterministic Compiler

### 6.1 Generators

Each generator takes `CompilerInput` and emits files via Jinja2 templates. No LLM involved.

| Generator | Output |
|---|---|
| StateModelGenerator | `state_models.py` — Pydantic v2 models per entity |
| ActionSchemaGenerator | `action_models.py` — one typed class per action |
| TransitionGenerator | `transitions/<action>.py` — one file per action |
| VerifierGenerator | `verifiers/<task>.py` — one file per task |
| RewardGenerator | `rewards/<task>.py` — one file per task |
| TaskTemplateGenerator | `tasks/<task>.yaml` |
| PolicyGenerator | `policies/<policy>.yaml` |
| GymWrapperGenerator | `gym_wrapper.py` |

### 6.2 Generated Package Structure

```
generated_envs/zendesk_support_env/
  env.yaml
  README.md
  state_models.py
  action_models.py
  transitions/
    add_ticket_comment.py
    update_ticket_status.py
    assign_ticket.py
  verifiers/
    resolve_refund_ticket.py
  rewards/
    resolve_refund_ticket.py
  tasks/
    refund_ticket_v1.yaml
  policies/
    support_policy.yaml
  custom/                       # never overwritten by recompile
    transitions.py
    verifiers.py
    rewards.py
    observations.py
    policies.py
    config.yaml
  gym_wrapper.py
  tests/
    test_determinism.py
    test_transitions.py
    test_verifiers.py
    test_invalid_actions.py
```

### 6.3 Regeneration Safety

The compiler never overwrites anything in `custom/`. Generated stubs contain a header pointing to the override path:

```python
# Generated by Forge — override in custom/transitions.py
# @override_transition("add_ticket_comment")
```

If regeneration would break an existing custom override (renamed action, removed entity), the validation runner reports it before completing.

### 6.4 Validation Runner

Runs automatically after generation. Blocks download on failure.

- `test_determinism.py` — same seed produces same trajectory hash
- `test_transitions.py` — each action mutates state as expected
- `test_verifiers.py` — success/failure conditions fire correctly
- `test_invalid_actions.py` — invalid inputs rejected cleanly

---

## 7. Customization Layer

### 7.1 Override Types

Five decorator-based extension points, all in `custom/`, never touched by recompile:

```python
@override_transition("update_ticket_status")
def custom_update_ticket_status(state, action, ctx): ...

@verifier("refund_policy_compliance")
def verify_refund_policy(state, trajectory, task): ...

@reward("ticket_resolution_reward")
def reward_ticket_resolution(state, trajectory, verifier_result): ...

@observation_transform("support_agent_view")
def support_agent_view(state, actor): ...

@policy_rule("no_refund_without_order_id")
def no_refund_without_order_id(state, action, ctx): ...
```

### 7.2 Config-Based Customization

`custom/config.yaml` for parameters that don't need Python:

```yaml
reward:
  base_success: 1.0
  step_penalty: 0.01
  policy_violation_penalty: 1.0
  max_reward: 1.0
  min_reward: -1.0

observation:
  mode: role_based
  actor_role: support_agent
  visible_entities: [assigned_tickets, public_comments, policy_docs]
  hidden_entities: [billing_internal_notes, audit_logs]
```

### 7.3 CLI

```bash
forge init zendesk_support_env
forge compile --config forge.yaml
forge validate generated_envs/zendesk_support_env
forge run --env zendesk_support_env --task refund_ticket_v1 --seed 42
forge replay --episode ep_123
forge export --format parquet --out ./rollouts
```

---

## 8. Verifier & Reward Engine

### 8.1 Verifier Types

| Type | Evaluation |
|---|---|
| ExactStateVerifier | `ticket.status == "solved"` |
| EventVerifier | trajectory contains `event.ticket_comment_added` |
| TemporalVerifier | `ask_for_order_id` occurs before `offer_refund` |
| PolicyVerifier | no forbidden action occurred |
| SemanticVerifier | LLM/classifier judges free-text quality |
| NegativeVerifier | agent did NOT perform action X |

All six are composable. A task's verifier is a list of checks from any combination of types.

### 8.2 SemanticVerifier Modes

Three modes to prevent rollouts from being blocked by LLM latency:

- `live` — calls LLM judge per episode (production)
- `cached` — caches by `(rubric, text_hash)`, deterministic for repeated inputs
- `mock` — returns fixed score, enforced by `FORGE_ENV=test`

### 8.3 VerificationResult Schema

```json
{
  "verifier_id": "resolve_refund_ticket",
  "passed": false,
  "score": 0.4,
  "checks": [
    { "name": "ticket_solved", "passed": true, "score": 1.0 },
    { "name": "refund_tag_added", "passed": false, "score": 0.0 },
    { "name": "asked_for_order_id_first", "passed": false, "score": 0.0,
      "evidence": "offer_refund at step 2, ask_for_order_id at step 4" }
  ],
  "explanation": "Agent did not follow order verification policy."
}
```

### 8.4 Decomposed Reward

Five named components. Never a single opaque score.

```python
reward = (
    task_success_reward       # verifier.pass_rate × base_success
  + policy_compliance_reward  # 0.0 or -penalty per violation
  + semantic_quality_reward   # SemanticVerifier score × weight
  - action_cost               # step_penalty × step_count
  - invalid_action_penalty    # per invalid action attempt
)
# clamped to [-1.0, 1.0]
```

### 8.5 Adversarial Trajectory Test Suite

Generated per task during `forge validate`. Covers reward-hacking patterns:

| Adversarial trajectory | Caught by |
|---|---|
| Set `status=solved` without any reply | `EventVerifier: ticket_comment_added` |
| Write empty comment body | `SemanticVerifier: acknowledges_issue` |
| Add "refund" tag, never resolve | `ExactStateVerifier: ticket.status == solved` |
| `offer_refund` before `ask_for_order_id` | `TemporalVerifier: order_before_refund` |
| Call forbidden internal API | `PolicyVerifier: no_forbidden_action` |
| Ask for order ID, then close ticket | `NegativeVerifier: no_premature_close` |

---

## 9. Observability, Replay & Debugging

### 9.1 Trace Model

OpenTelemetry spans at four levels: `compile`, `episode`, `step`, `verify+reward`.

Episode trace:

```json
{
  "episode_id": "ep_123",
  "environment": "zendesk_support_env",
  "task_id": "resolve_refund_ticket",
  "seed": 42,
  "agent_id": "agent_claude_sonnet",
  "started_at": "sim_time_0",
  "steps": []
}
```

Step trace includes `observation_hash`, `action`, `action_valid`, `state_diff`, `events`, `verifier_results`, `reward_breakdown`, `latency_ms`, `errors`.

### 9.2 Branch Replay

Fork from step N by calling `reset(seed)` and fast-forwarding through steps 0..N-1 using the stored action sequence, then diverging. No snapshot rollback required.

### 9.3 Failure Clustering

Groups episodes by the first `CheckResult` with `passed=false`. Dashboard surfaces the five most common failure modes per task automatically.

### 9.4 Frontend Pages

| Page | Content |
|---|---|
| Dashboard | pass rate, average reward, common failures, policy violations, average steps |
| Episode Replay | task instructions, action timeline, state diff per step, reward breakdown, verifier results |
| Environment Graph | entities, actions, mutation paths, policies, task templates |
| Compiler Review | LLM-inferred entities/actions/policies/tasks, approval before generation |

---

## 10. Parallel Rollouts & Training Export

### 10.1 Rollout Workers

Celery workers. Each receives a `RolloutJob` and runs episodes sequentially within the worker. Parallelism from multiple workers, not threading.

```json
{
  "environment": "zendesk_support_env",
  "task_id": "resolve_refund_ticket",
  "agent_config": { "provider": "anthropic", "model": "claude-sonnet-4-6" },
  "num_episodes": 100,
  "seed_start": 1000
}
```

### 10.2 Agent Adapters

```python
class AgentAdapter(Protocol):
    def act(self, observation: dict, tools: list[ToolSpec]) -> dict: ...
```

Five implementations at launch: OpenAI tool-calling, Anthropic tool-calling, local vLLM, random policy, scripted policy (YAML action sequence — primary tool for baseline testing and adversarial generation).

### 10.3 Export Formats

```
exports/
  trajectories.jsonl
  rewards.jsonl
  verifier_results.jsonl
  sft_pairs.jsonl
  preference_pairs.jsonl
  grpo_rollouts.parquet
```

Preference pairs: highest-scoring and lowest-scoring episodes for the same task+seed.

---

## 11. Security & Policy Enforcement

### 11.1 Network Isolation

No outbound network calls from generated environments unless `FORGE_DEV_NETWORK=true`. Generated transition stubs containing `requests`, `httpx`, or `urllib` fail the validation runner.

### 11.2 Policy Engine

Sits between `ActionValidator` and `TransitionEngine`. Uses a simple internal DSL (Python expressions evaluated against a sandboxed state snapshot). OPA/Cedar deferred to post-M7.

```
ActionValidator → PermissionChecker → PolicyEngine → TransitionEngine
```

### 11.3 PII Redaction

Preprocessing step in `IngestionService` before uploaded content touches the state model generator. Configurable pipeline with `@pii_detector` hook for custom detectors.

### 11.4 Role-Based Observations

```yaml
roles:
  support_agent:
    can_see: [assigned_tickets, public_comments]
    cannot_see: [billing_internal_notes, audit_logs]
```

### 11.5 Audit Logs

Every policy violation:

```json
{
  "episode_id": "ep_123",
  "step": 4,
  "actor": "agent",
  "action": "offer_refund",
  "violation": "refund_without_order_id",
  "severity": "high"
}
```

### 11.6 Auth

JWT-based, project-scoped permissions. Users → Projects → Environments.

---

## 12. Milestone Map

| # | Milestone | Scope | Web UI |
|---|---|---|---|
| M1 | Runtime Kernel | Pure Python Gymnasium kernel, Gmail hand-written env, determinism tests | None |
| M2 | LLM Extraction + Compiler | Multi-step LLM chain → CompilerInput, Jinja2 template compiler, Package Builder, validation runner | Prompt input, file upload, Compiler Review UI |
| M3 | Customization Layer | Override hooks, config-based reward/observation, CLI, override validation on recompile | Environment config editor |
| M4 | Verifier & Reward Engine | All six verifier types, decomposed reward, SemanticVerifier with caching, adversarial test suite | None |
| M5 | Observability & Replay | Full OTel trace model, branch replay, failure clustering, episode replay UI | Dashboard, Episode Replay, Environment Graph |
| M6 | Parallel Rollouts & Export | Celery workers, five agent adapters, all export formats, preference pair generation | Rollout launcher, export UI |
| M7 | Security & Policy | Network isolation, PolicyEngine DSL, PII redaction, RBAC observations, audit logs | Policy violation viewer |

---

## 13. Tech Stack

| Layer | Technology |
|---|---|
| Backend API | Python 3.11+, FastAPI, Pydantic v2 |
| ORM / DB | SQLAlchemy, PostgreSQL |
| Queue / Cache | Redis, Celery |
| Code Generation | Jinja2 |
| RL Interface | Gymnasium |
| Observability | OpenTelemetry |
| Frontend | Next.js, TypeScript, Tailwind, shadcn/ui, React Flow, Monaco Editor, TanStack Query |
| LLM Interface | Provider-agnostic (OpenAI, Anthropic, vLLM) |
| Auth | JWT |
| Infra | Docker Compose (self-hosted) |
