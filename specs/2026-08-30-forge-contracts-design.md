# Shared Contracts for Environment Generation

**Date:** 2026-08-30
**Status:** Approved for planning (phases 1-3 not started)

## Problem

Forge runs four unrelated families of environment:

- **In-process** — `ForgeEnv`, assembled by `forge/runtime/env_builder.py` from compiler output.
- **Container** — `ContainerEnvBase` (gymnasium over HTTP) and `ContainerEpisodeRunner`, wrapping an LLM-generated FastAPI app.
- **CLI** — `CliEpisodeRunner`, driving a shell over `docker exec`.
- **Browser** — `BrowserEpisodeRunner`, driving Chromium over Playwright.

Eleven concerns recur in all four. Today each family solves them independently, and only one — termination — is genuinely shared:

| Concern | `ForgeEnv` | Container | CLI | Browser |
|---|---|---|---|---|
| Tasks / dataset | `TaskTemplate`, `default_task` | `ScenarioSuite` + objective | objective string | objective string |
| Initial state | `InitialStateFactory.create()` | `POST /forge/reset` | container boot | page load |
| Prompt template | `AgentPrompt` | agent-specific | agent-specific | agent-specific |
| Tool definition | `ToolSpec` / `Capability` | `/openapi.json` discovery | `ComputerUse` | `BrowserUse` |
| Observation format | gym dict + `ObservationFilter` | `GET /forge/state` | `_state()` dict | page snapshot |
| Execution backend | `TransitionEngine` | Docker + HTTP | `docker exec` | Playwright |
| State management | `StateStore` | app-side SQLite | history list | browser session |
| Reward / rubric | `RewardEngine` | `compute_reward`, `ObjectiveScorer` | `TieredRewardEngine` | `ObjectiveScorer` |
| Termination | gym `terminated`/`truncated` | `TerminationMonitor` | `TerminationMonitor` | `TerminationMonitor` |
| Episode control | external caller | `ContainerEpisodeRunner` | `CliEpisodeRunner` | `BrowserEpisodeRunner` |
| Transport | in-process calls | httpx REST | subprocess | CDP |

Two consequences drive this work:

1. **Generation has no target.** The envgen specialists (`BackendBuilderAgent`, `StateBridgeAgent`, `RewardAgent`, `ScenarioBuilderAgent`) describe these concerns in prose inside their prompts. There is no importable contract for a generated environment to satisfy, and no mechanical way for `ReviewerAgent` to check that it did.
2. **The registries are unchecked.** `TransitionEngine.register`, `VerifierEngine.register`, and `RewardEngine.register` all accept a bare `Callable` with no verified signature. A generated handler with the wrong arity is accepted at build time and fails mid-episode, which is the single weakest seam in the runtime.

## Goals

- One importable vocabulary for the eleven concerns, in a package both families can depend on.
- Nominal binding: implementations inherit the contracts, so conformance is checked at class-definition and registration time rather than at first call.
- A documented extension surface for someone authoring an environment by hand.
- A concrete target the envgen specialists reference and the reviewer gate verifies.

## Non-goals

- Changing runtime behavior. Every existing test must pass unchanged except where a signature is deliberately tightened.
- Unifying the four families into one implementation. They stay distinct; they share a vocabulary, not an implementation.
- Adding new environment types or capabilities.

## Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Binding | Prescriptive ABCs, refactor now | Structural Protocols would not catch the unchecked-registry bug, which is the main defect motivating the work. |
| Shape | Eleven independent ABCs + composed facade | Families differ in which concerns apply. A shell has no tool schema; a pure-Python env has no transport. A monolith would force every family to stub concerns it does not have. |
| Scope of rebase | Full (approach C), including `ForgeEnv` internals and `forge/templates/` | The `Callable` registries are the defect. Approach A left them in place. |
| Package name | `forge/contracts/` | Distinct from `forge/schema/` (what data looks like) and `forge/runtime/` (what executes). |
| Primary consumers | envgen agents, and humans authoring environments by hand | Determines that phase 3 (prompt + reviewer adoption) is in scope, not optional. |

## Architecture

### Dependency direction

`forge/contracts/` imports from `forge/schema/` (stdlib + pydantic only, so no cycle) and the standard library at runtime. It must not import from `forge/runtime/`, `forge/envgen/`, `forge/extraction/`, or `backend/` at runtime. Both environment families depend on `contracts/`; `contracts/` depends on neither. A test asserts this directly by walking the package's imports, because a cycle here would be discovered late and be painful to unwind.

Three types the contract signatures need still live in `forge/runtime/` and stay there, because they carry runtime behavior rather than shape: `RuntimeContext` (`forge/runtime/context.py`), `Trajectory` (`forge/runtime/trajectory.py`), and `TransitionResult` (`forge/runtime/transition.py`). `contracts/` imports all three under `if TYPE_CHECKING:` only and annotates them as strings, so they type-check without creating a runtime edge. The import-direction test permits `TYPE_CHECKING` imports and fails on any other import from those packages.

Shared data types the contracts speak in (`Task`, `Observation`, `Action`, `ActionResult`, `Termination`, `StepOutcome`) live in `forge/contracts/types.py`. Existing pydantic models that already serve this role are re-exported rather than duplicated: `ToolSpec` and `ToolParam` from `forge.runtime.snapshot` move to `forge/contracts/types.py` and are re-exported from their old location for one release, because `snapshot.py` also holds `StepSnapshot` and `EnvironmentSpec`, which are runtime concerns and stay put.

`RewardBreakdown` / `RewardComponent` and `VerificationResult` / `CheckResult` likewise move to `forge/contracts/types.py`, since `Rubric` and the verifier contract are both defined in terms of them. `AgentAdapter` (today a Protocol in `forge/runtime/agents/base.py`) moves too, because `EpisodeController.run_episode` takes one — it is part of the contract surface, not a runtime detail.

### Layout

```
forge/contracts/
  __init__.py         re-exports all eleven + Environment + shared types
  types.py            Task, Observation, Action, ActionResult, Termination,
                      StepOutcome, ToolSpec, ToolParam, RewardBreakdown,
                      RewardComponent, VerificationResult, CheckResult,
                      AgentAdapter
  dataset.py          TaskSource
  initial_state.py    InitialStateProvider
  prompting.py        PromptTemplate
  tools.py            ToolProvider
  observation.py      ObservationEncoder
  backend.py          ExecutionBackend, TransitionHandler
  state.py            StateManager
  reward.py           Rubric, Verifier
  termination.py      TerminationPolicy
  episode.py          EpisodeController
  transport.py        Transport, TransportRequest, TransportResponse
  environment.py      Environment facade
```

### The eleven contracts

All take `ctx: RuntimeContext` where they need determinism primitives (clock, seeded RNG, seeded UUIDs). `RuntimeContext` stays in `forge/runtime/context.py` and is imported by `contracts/` under `TYPE_CHECKING` only, with the runtime passing it in — this keeps the dependency rule intact while preserving the existing signatures.

**1. `TaskSource`** (`dataset.py`) — what problems the model should solve.

```python
class TaskSource(ABC):
    @abstractmethod
    def tasks(self) -> Sequence[Task]: ...
    @abstractmethod
    def get(self, task_id: str) -> Task: ...
```

`Task` unifies `TaskTemplate` (compiler) and `Scenario` (envgen): `id`, `objective`, `seed`, `success_conditions`, `failure_conditions`, `metadata`. Implementations: `TemplateTaskSource` (wraps `list[TaskTemplate]`), `ScenarioTaskSource` (wraps `ScenarioSuite`), `ObjectiveTaskSource` (a single natural-language objective, for CLI and browser).

**2. `InitialStateProvider`** (`initial_state.py`) — per-episode setup.

```python
class InitialStateProvider(ABC):
    @abstractmethod
    def reset(self, ctx: RuntimeContext, *, seed: int | None, options: Mapping[str, object]) -> dict: ...
```

Replaces the existing `InitialStateFactory` Protocol, whose `create(ctx, options)` becomes `reset(ctx, seed=..., options=...)` — the seed is currently smuggled through `options` in some call sites and through `ctx.seed` in others, and making it an explicit keyword removes that ambiguity. Implementations: the generated `*InitialStateFactory` classes, `HttpResetProvider` (`POST /forge/reset {"seed": n}`), `ContainerBootProvider`, `PageLoadProvider`.

**3. `PromptTemplate`** (`prompting.py`) — how the task reaches the model.

```python
class PromptTemplate(ABC):
    @abstractmethod
    def system(self, task: Task) -> str: ...
    @abstractmethod
    def user(self, observation: Observation, task: Task) -> str: ...
    @abstractmethod
    def tool_descriptions(self, tools: Sequence[ToolSpec]) -> list[dict]: ...
```

The existing `AgentPrompt` frozen dataclass and its `FORGE_AGENT_PROMPT` instance become the default implementation `ForgeAgentPromptTemplate`, preserving the current strings exactly.

**4. `ToolProvider`** (`tools.py`) — what the model can do.

```python
class ToolProvider(ABC):
    @abstractmethod
    def tools(self) -> Sequence[ToolSpec]: ...
```

Implementations: `SpecToolProvider` (a static list), `OpenAPIToolProvider`, `CapabilityToolProvider` (adapts `ComputerUse` / `BrowserUse` / `MCPUse` / `RESTUse` / `ORPCUse`). `OpenAPIToolProvider` absorbs `ContainerEpisodeRunner._discover_actions`, including its existing exclusion of `/forge/*` and `/ui` paths — action discovery becomes reusable instead of being private to one runner.

**5. `ObservationEncoder`** (`observation.py`) — what the model sees back.

```python
class ObservationEncoder(ABC):
    @abstractmethod
    def encode(self, state: dict, ctx: RuntimeContext) -> Observation: ...
```

`Observation` is a frozen model with `payload: dict`, `text: str | None`, and `blocks: list[dict]`, which covers raw text, structured tool output, and the gym dict in one type. `ObservationFilter` (RBAC) becomes `RbacObservationEncoder`, retaining its `RBACConfig`/`role` constructor and its pass-through behavior when either is `None`.

**6. `ExecutionBackend`** (`backend.py`) — where actions run.

```python
class TransitionHandler(ABC):
    @abstractmethod
    def apply(self, state: dict, action: Action, ctx: RuntimeContext) -> TransitionResult: ...

class ExecutionBackend(ABC):
    @abstractmethod
    def execute(self, action: Action, state: dict, ctx: RuntimeContext) -> ActionResult: ...
    def close(self) -> None: ...
```

Implementations: `InProcessBackend` (owns the `TransitionHandler` registry), `HttpBackend`, `DockerExecBackend`, `PlaywrightBackend`.

**7. `StateManager`** (`state.py`) — how state is tracked across turns.

```python
class StateManager(ABC):
    @abstractmethod
    def get(self) -> dict: ...
    @abstractmethod
    def apply(self, state: dict) -> None: ...
    @abstractmethod
    def hash(self) -> str: ...
    def snapshot(self, slot: str) -> None: ...
    def restore(self, slot: str) -> None: ...
```

`snapshot`/`restore` are concrete and raise `NotImplementedError` by default, because only the container family supports slots today (`POST /forge/snapshot`, `POST /forge/restore/{slot}`). `StateStore` already matches `get`/`apply`/`hash` exactly and becomes `InProcessStateManager`; `HttpStateManager` covers the container family.

**8. `Rubric`** (`reward.py`) — how behavior is scored.

```python
class Verifier(ABC):
    @abstractmethod
    def verify(self, state: dict, trajectory: Trajectory, task: Task | None) -> VerificationResult: ...

class Rubric(ABC):
    @abstractmethod
    def score(
        self,
        state: dict,
        trajectory: Trajectory,
        verifier_results: Sequence[VerificationResult],
        task: Task | None,
    ) -> RewardBreakdown: ...
```

`RewardEngine`'s default behavior (1.0 if any verifier passed, else 0.0) becomes `TaskSuccessRubric`, preserved verbatim as the fallback. `TieredRewardEngine`, `ObjectiveScorer`, and `LayeredVerifier` become implementations — `LayeredVerifier` already exposes `__call__(state, trajectory, task) -> VerificationResult`, so it satisfies `Verifier` by renaming that method to `verify` and keeping `__call__` as an alias for one release.

**9. `TerminationPolicy`** (`termination.py`) — how the episode ends.

```python
class TerminationPolicy(ABC):
    @abstractmethod
    def check(self, outcome: StepOutcome) -> Termination | None: ...
```

`StepOutcome` carries `step_index`, `score`, `state_hash`, `reward`, and `verifier_results`. `Termination` carries `reason: str` and `truncated: bool`. `TerminationMonitor` becomes `ThresholdTerminationPolicy`, keeping its existing priority order (success, then dead-end, then divergence) and its `BaseEpisodeConfig` thresholds. `MaxStepsTerminationPolicy` makes the `step_index == max_steps - 1` → `truncated` rule explicit; today it is inline in each runner's loop.

**10. `EpisodeController`** (`episode.py`) — who drives the multi-turn loop.

```python
class EpisodeController(ABC):
    @abstractmethod
    def run_episode(
        self,
        agent: AgentAdapter,
        *,
        episode_id: str | None = None,
        seed: int | None = None,
        jsonl_path: Path | None = None,
    ) -> BaseEpisodeResult: ...
```

The three runners become subclasses. From `forge/envgen/episode_base.py`: `BaseEpisodeConfig`, `BaseEpisodeResult`, and `TrajectoryWriter` move to `forge/contracts/episode.py`, and `TerminationMonitor` moves to `forge/contracts/termination.py` (renamed `ThresholdTerminationPolicy`, per contract 9). `forge/envgen/episode_base.py` re-exports all four for one release. `CliEpisodeRunner.run_episode` and `BrowserEpisodeRunner.run_episode` gain the `seed` keyword they currently lack, accepting and ignoring it where the family has no seeding path, so the signature is uniform.

**11. `Transport`** (`transport.py`) — how the model talks to the environment.

```python
class Transport(ABC):
    @abstractmethod
    def call(self, request: TransportRequest) -> TransportResponse: ...
    def close(self) -> None: ...
```

`TransportRequest` carries `method`, `target`, `payload`, and `timeout`; `TransportResponse` carries `status`, `body`, and `error`. Implementations: `InProcessTransport`, `RestTransport` (httpx), `SubprocessTransport` (`docker exec`), `CdpTransport` (Playwright). `ExecutionBackend` implementations hold a `Transport` rather than talking to httpx or subprocess directly.

### The `Environment` facade

```python
class Environment(ABC):
    @property
    @abstractmethod
    def task_source(self) -> TaskSource: ...
    @property
    @abstractmethod
    def initial_state(self) -> InitialStateProvider: ...
    @property
    @abstractmethod
    def observations(self) -> ObservationEncoder: ...
    @property
    @abstractmethod
    def backend(self) -> ExecutionBackend: ...
    @property
    @abstractmethod
    def state(self) -> StateManager: ...
    @property
    @abstractmethod
    def rubric(self) -> Rubric: ...
    @property
    @abstractmethod
    def termination(self) -> TerminationPolicy: ...

    # Concerns that do not apply to every family default to None.
    @property
    def prompt(self) -> PromptTemplate | None: return None
    @property
    def tools(self) -> ToolProvider | None: return None
    @property
    def transport(self) -> Transport | None: return None
```

Seven required members, three optional. The optional three are exactly the ones a family can legitimately lack: a CLI environment has no tool schema, a pure-Python environment has no transport, and an environment driven by a trainer that supplies its own prompting has no `PromptTemplate`.

`EpisodeController` is deliberately **not** a member. It drives an environment from the outside; making it a member would imply every environment owns its own loop, which is false for `ForgeEnv` (driven by `parallel_rollout.py` or an external trainer) and would prevent the same environment being run by different controllers.

## Phases

Each phase is separately landable and leaves the suite green.

### Phase 1 — the package

Create `forge/contracts/` with the eleven ABCs, the shared types, and the facade. Move `ToolSpec`/`ToolParam`, `RewardBreakdown`/`RewardComponent`, and `VerificationResult`/`CheckResult` into `contracts/types.py`, re-exporting from their old locations. Add the import-direction test. No existing class changes yet.

### Phase 2 — the rebase

In dependency order:

1. `StateStore` → `InProcessStateManager(StateManager)`; add `HttpStateManager`.
2. `TerminationMonitor` → `ThresholdTerminationPolicy(TerminationPolicy)`; add `MaxStepsTerminationPolicy`.
3. `TransitionEngine` registry accepts `TransitionHandler` instances; `VerifierEngine` accepts `Verifier`; `RewardEngine` accepts `Rubric`. Each `register` raises `TypeError` on a non-conforming object — the defect this work exists to fix.
4. `forge/customization/hooks.py` decorators (`@override_transition`, `@verifier`, `@reward`) wrap the decorated plain function into the corresponding ABC, so the hook API stays exactly as ergonomic as it is today and `CustomizationLoader.apply` keeps working unchanged.
5. `ForgeEnv` takes contract-typed collaborators; `ContainerEnvBase` implements `Environment`.
6. The three runners become `EpisodeController` subclasses.
7. `forge/templates/*.j2` and `forge/compiler/generators/` emit the new shape.
8. `examples/gmail_env` is migrated to the new shape in the same change.

`generated_envs/` contains only `.gitkeep`, so `examples/gmail_env` is the only checked-in package in the template's shape and the migration surface is bounded. Environments a user has generated locally and kept outside the repo would need regenerating; this is called out in the README section added in phase 3.

### Phase 3 — envgen adoption

- `BackendBuilderAgent`, `StateBridgeAgent`, `RewardAgent`, and `ScenarioBuilderAgent` prompts cite the contract names and signatures instead of describing them in prose.
- `ReviewerAgent` gains a static check that generated packages implement the contracts they claim: the state bridge subclasses `Environment`, the reward function satisfies `Rubric`, and every declared action has a `TransitionHandler`. This is a mechanical check, complementing the existing semantic review.
- README documents `forge/contracts/` as the extension surface for hand-authored environments.

## Testing

A single conformance suite in `tests/contracts/`, parameterized over every implementation of each contract, so adding a family means adding one entry to one list.

Per the project's test-diversity standard, alongside the happy path:

- **Negative — missing method.** A subclass omitting a required abstract method fails at instantiation, not at first call.
- **Negative — wrong signature at registration.** `TransitionEngine.register` given a bare two-argument function raises `TypeError`. This is the exact bug the rebase exists to prevent, so it gets a direct test.
- **False-positive guard — legitimate absence.** A CLI environment that supplies no `ToolProvider` still satisfies `Environment`; the optional members must not be silently required.
- **False-positive guard — contracts stay decoupled.** The import-direction test fails if `contracts/` imports from `runtime/`, `envgen/`, `extraction/`, or `backend/`.
- **Behavior preservation.** `ThresholdTerminationPolicy` reproduces `TerminationMonitor`'s decisions on the same inputs, including the priority order between success, dead-end, and divergence.

The determinism suites are the regression tripwire for phase 2, because they exercise the precise seams being rewritten: `tests/backend/test_env_loader_determinism.py`, `tests/envgen/test_container_seed_determinism.py`, and `tests/envgen/test_correctness_validator.py`. They must pass unchanged.

## Risks

| Risk | Mitigation |
|---|---|
| Phase 2 silently changes runtime behavior | Determinism suites must pass unchanged; behavior-preservation tests pin `TerminationMonitor` and `RewardEngine` default semantics before the move. |
| Tightened `register` breaks a hand-written customization | The `hooks.py` decorators wrap plain functions into ABCs, so the documented customization API is unchanged. |
| Moved types break imports | Old locations re-export for one release. |
| Metaclass conflict from ABC + `gymnasium.Env` | Retired. Verified 2026-08-30 against the installed gymnasium: `type(gymnasium.Env)` is plain `type`, a class inheriting both `gymnasium.Env` and `abc.ABC` composes cleanly, and abstract-method enforcement fires on instantiation. |
