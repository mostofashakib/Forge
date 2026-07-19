# Automatic reviewer repair loop — design

Status: approved for planning
Task: TASKS.md #1 "Automatic reviewer repair loop"

## Problem

The planned generation pipeline ends with two quality gates: `ReviewerAgent`
(`review_report`) and `EnvironmentCorrectnessAgent` (`correctness_report`).
Today, `enforce_generation_gates()` in
`backend/app/services/env_orchestrator.py` raises `GenerationReviewError` the
moment either report is not approved. A single fixable finding throws away the
entire generation. There is no mechanism to route a finding back to the
specialist that produced the offending artifact and try again.

We want a bounded, self-terminating repair loop: convert rejected findings into
typed correction tasks, route each to the responsible specialist, re-run that
specialist (correction-aware) plus its downstream and the reviewers, and repeat
until the generation is approved or the loop provably cannot make progress.

## Goals (from TASKS.md #1)

1. Convert rejected reviewer findings into typed correction tasks.
2. Map every correction task to the specialist responsible for its artifact.
3. Route only the finding, acceptance criteria, and relevant artifact context
   back to that specialist.
4. Re-run affected downstream tasks and the reviewer with a bounded retry count.
5. Preserve review and correction history in the A2A protocol.
6. Add tests for routing, retry limits, downstream invalidation, and unrepaired
   failures.

## Non-goals

- No change to the "explicitly supplied agents" branch of
  `EnvironmentOrchestrator.run` (the open-bus path used by extensions and older
  tests). The repair loop runs only in the planned-pipeline branch.
- No new persistence. History lives in the in-process `A2AProtocol` for the
  duration of a run, exactly like existing coordination messages.
- No repair of WARNING-severity findings. Those never block acceptance, so they
  must never spend an LLM round.

## Decisions

- **Correction-aware regeneration.** A specialist re-run receives the specific
  findings and its prior output so its LLM prompt can target the fix, rather
  than regenerating blind and hoping variance helps.
- **Circuit breaker to prevent churn.** Bounded retries alone are not enough;
  the loop also stops early when a round makes no progress, so we never keep
  paying for LLM calls that repeatedly fail to move the same finding.
- **Unmappable findings fail fast.** A finding whose artifact maps to no
  specialist (e.g. a semantic requirement-coverage finding with `artifact=None`)
  is unrepairable. If such a finding survives, the loop stops and raises
  `GenerationReviewError`.

## Architecture

### New module: `forge/envgen/repair.py`

Isolated from the orchestrator so it is unit-testable without the backend.

- `CorrectionTask` (pydantic `BaseModel`):
  - `finding: ReviewIssue` — the rejecting issue.
  - `target_agent_id: str` — the specialist that owns the artifact.
  - `artifact: str | None` — the offending artifact/file.
  - `acceptance_criteria: list[str]` — copied from the target task's criteria.
  - `source_report: str` — `"review"` or `"correctness"`.
  - `round: int` — 1-based repair round that produced this task.

- `FindingRouter`:
  - Built from the pipeline's agents. Produces an artifact→`agent_id` map from
    each agent's `produces`, then overlays a file-path resolution table for
    members of the composite `app_code` artifact:
    - `ui.html` → `ui_builder`
    - any other `app_code` file (`main.py`, `requirements.txt`, `Dockerfile`,
      etc.) → `backend_builder`
    - `instrumented:<path>` → `telemetry`
    - a bare named artifact (`state_bridge_code`, `reward_fn_code`,
      `policy_dsl`, `state_schema_manifest`) → its declared producer
  - `route(issue: ReviewIssue) -> str | None`. Returns `None` for
    `artifact is None` or any artifact with no owning specialist.
  - Rationale for the file table: `app_code` is *produced* by `app_assembler`
    (a pure combine step), but the fixable source lives in `backend_builder` /
    `ui_builder`. Routing `app_code` findings to the assembler would never fix
    them.

- `RepairPlanner`:
  - Given the base `GenerationPlan` and a set of target `agent_id`s, computes
    the **re-run set** = targets ∪ all transitive downstream tasks (following
    artifact dependencies) ∪ `{correctness_reviewer, reviewer}` (whichever are
    present). Returns a `GenerationPlan` sub-plan containing only those tasks, in
    dependency order.

- `RepairLoop`:
  - `async def run(plan, agents, ctx, bus, executor) -> None`
  - Owns rounds and the circuit breaker. Raises `GenerationReviewError` on any
    terminal non-approved stop.

### Circuit breaker / termination

Evaluated at the top of every round. The loop stops on the first true
condition:

0. **Missing report** — if `review_report` is absent from the bus, raise
   `RuntimeError("Reviewer did not publish a review report")`, preserving the
   current guard in `enforce_generation_gates`.
1. **Approved** — both gates approved (`review_report.approved` and, if present,
   `correctness_report.approved`). Success; return.
2. **Retry bound** — completed `max_repair_rounds` rounds
   (default 2, `FORGE_ENVGEN_MAX_REPAIR_ROUNDS`). Stop, raise.
3. **No progress** — the multiset of ERROR-finding fingerprints
   `(category, artifact, message)` did not strictly shrink versus the previous
   round. The specialist churned without fixing anything; open the breaker.
   Stop, raise.
4. **Unrepairable** — at least one surviving ERROR finding routes to `None`.
   Stop, raise.

Only ERROR-severity findings are collected for repair. WARNINGs are ignored by
the loop (they do not affect `approved`). `max_repair_rounds` counts repair
attempts; `0` reproduces today's immediate hard-fail behavior.

### Correction channel (agent side)

Generic and opt-in so the mechanism is one code path, not per-agent bespoke
wiring.

- The loop publishes `correction:{agent_id}` to the bus:
  `{ "findings": [ReviewIssue…], "acceptance_criteria": [...],
     "prior_output": <the agent's last produced artifact value(s)> }`.
- `render_correction_context(bus, agent_id) -> str | None` helper (in
  `forge/envgen/agents/base.py`) renders a prompt-ready block or `None`.
- Builder agents that own repairable artifacts append the rendered block to
  their LLM `user` prompt when present: `backend_builder`, `ui_builder`,
  `telemetry`, `state_bridge`, `policy`, `reward`. `app_assembler` is a pure
  combine step (no LLM) and simply re-runs.
- The per-agent correction artifact is added to that agent's readable scope in
  the re-run sub-plan (`context_keys`), so the least-privilege channel still
  permits the read.

### Downstream invalidation & re-run

- `ArtifactBus.invalidate(names: Iterable[str])`: remove each value and reset its
  `asyncio.Event` (clear + recreate) so any consumer that calls `wait_for`
  blocks until the artifact is republished.
- Executor change (`TaskExecutor.execute`): a task dependency that is **not**
  present in the current plan's task set is assumed already satisfied on the bus
  and is skipped rather than recreated. This lets `RepairLoop` execute a
  sub-plan whose external dependencies are already published. `run_task`'s
  per-call `running` dict already isolates one `execute` call from the next.
- Per round the loop: (a) publishes `correction:{agent_id}` for each target,
  (b) invalidates the outputs of every task in the re-run set (so the reviewers
  wait for fresh code), (c) `await executor.execute(sub_plan, agents, ctx, bus)`.

### A2A history

Extend `MessageKind` with `REVIEW_REJECTED`, `CORRECTION_ASSIGNED`,
`CORRECTION_COMPLETED`, `REPAIR_EXHAUSTED`. The loop sends these through
`bus.protocol` so `protocol.history` preserves the full review + correction
trail:

- `REVIEW_REJECTED` — one per round, payload lists the ERROR findings.
- `CORRECTION_ASSIGNED` — one per `CorrectionTask` (sender orchestrator,
  recipient target specialist), payload carries finding + acceptance criteria +
  round.
- `CORRECTION_COMPLETED` — after a re-run finishes for a target.
- `REPAIR_EXHAUSTED` — terminal, payload records the stop reason
  (`retry_bound` | `no_progress` | `unrepairable`).

### Orchestrator integration

`enforce_generation_gates()` is replaced by invoking `RepairLoop`. In
`EnvironmentOrchestrator.run` planned branch:

```
plan = PromptPlannerAgent().create_plan(ctx, agents)
await bus.publish("generation_plan", plan)
executor = TaskExecutor()
await executor.execute(plan, agents, ctx, bus)
await RepairLoop().run(plan, agents, ctx, bus, executor)   # raises if unrepaired
self._write_artifacts(env_name, bus)
```

On success the loop returns and artifacts are written as before. On any terminal
non-approved stop it raises `GenerationReviewError` with the final report, so all
existing callers and error handling are unchanged.

## Data flow (one repair round)

```
reports on bus
  -> collect ERROR findings (review + correctness)
  -> route each -> CorrectionTask   (any None -> unrepairable, stop+raise)
  -> breaker: progress vs previous round?  (no shrink -> stop+raise)
  -> publish correction:{agent} for each target
  -> invalidate outputs of re-run set
  -> executor.execute(sub-plan)  [targets -> assembler -> telemetry -> bridge -> reviewers]
  -> new reports -> next round
```

## Testing (goal 6)

Unit tests, LLM specialists replaced by deterministic stubs.

- **Routing:** `main.py`/`Dockerfile` → backend_builder, `ui.html` → ui_builder,
  `instrumented:main.py` → telemetry, `reward_fn_code` → reward,
  `state_bridge_code` → state_bridge, `policy_dsl` → policy; `artifact=None` →
  `None`.
- **Retry limit:** a stub reviewer that always rejects → exactly
  `max_repair_rounds` repair attempts, then `GenerationReviewError`; assert the
  target specialist ran `max_repair_rounds + 1` times total.
- **Downstream invalidation:** a finding on `backend_code`/`main.py` re-runs
  assembler, telemetry, state_bridge, and the reviewers, but not an unrelated
  branch (e.g. `policy`); assert via per-agent run counters.
- **No-progress breaker:** a stub that keeps emitting the identical finding →
  loop stops before `max_repair_rounds` and raises; assert round count.
- **Unrepairable:** a semantic finding with `artifact=None` → immediate
  `GenerationReviewError`, zero specialist re-runs.
- **History:** `protocol.history` contains `REVIEW_REJECTED`,
  `CORRECTION_ASSIGNED`, `CORRECTION_COMPLETED` in order, and `REPAIR_EXHAUSTED`
  on failure.
- **Happy path:** one repairable finding fixed in round 1 → both gates approved,
  loop returns, artifacts written.
- **Correction context:** `render_correction_context` returns the findings +
  acceptance criteria + prior output for a targeted agent and `None` otherwise;
  a correction-aware builder stub proves it consumed the block.

## Files touched

- `forge/envgen/repair.py` — new: `CorrectionTask`, `FindingRouter`,
  `RepairPlanner`, `RepairLoop`.
- `forge/envgen/a2a.py` — new `MessageKind` values.
- `forge/envgen/artifact_bus.py` — `invalidate()`.
- `forge/envgen/executor.py` — skip out-of-plan dependencies.
- `forge/envgen/agents/base.py` — `render_correction_context()` helper.
- `forge/envgen/agents/app_generator.py`, `telemetry.py`, `state_bridge.py`,
  `policy.py`, `reward.py` — fold correction context into prompts.
- `forge/envgen/config.py` — `max_repair_rounds` + `FORGE_ENVGEN_MAX_REPAIR_ROUNDS`.
- `backend/app/services/env_orchestrator.py` — invoke `RepairLoop` instead of
  `enforce_generation_gates`.
- `tests/envgen/test_repair_loop.py` — new test module.
