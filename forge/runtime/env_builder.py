from __future__ import annotations
import builtins
import socket
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Callable

from forge.runtime.env import ForgeEnv, InitialStateFactory
from forge.runtime.determinism import run_determinism_check
from forge.runtime.errors import DeterminismViolation, EnvironmentBuildError
from forge.runtime.interaction import (
    BrowserUse,
    BrowserUseSchema,
    ComputerUse,
    ComputerUseSchema,
)
from forge.runtime.reward import RewardEngine
from forge.runtime.snapshot import EnvironmentSpec, ToolParam, ToolSpec
from forge.runtime.transition import TransitionEngine, TransitionResult
from forge.runtime.verifier import VerifierEngine


@dataclass(frozen=True)
class DeterminismConfig:
    """Determinism contract enforced on environments produced by EnvBuilder.

    The first four are structural guarantees of the runtime (RuntimeContext
    provides a SimClock, a seeded RNG, and a seeded UUID generator; StateStore
    hashes with sorted keys) — the flags document the contract. The rest are
    actively enforced by the builder at reset/step time.
    """

    virtual_clock: bool = True          # SimClock, never wall clock
    seeded_rng: bool = True             # ctx.rng only, no global random state
    seeded_uuids: bool = True           # ctx.uuid_generator, IDs reproducible per seed
    sorted_serialization: bool = True   # sort keys before hashing set/map content
    integers_only: bool = True          # reject floats anywhere in state
    serialize_concurrency: bool = True  # one transition at a time, lock-ordered
    mock_external_calls: bool = True    # no real network calls inside the env
    mock_filesystem: bool = True        # no filesystem access inside the env
    fresh_startup: bool = True          # drop factory caches; each rollout is a fresh universe
    canonical_json: bool = True         # canonical serializer for cross-library stability




def _assert_no_floats(value, path: str) -> None:
    if isinstance(value, float):
        raise DeterminismViolation(
            f"float at {path} — use integers instead of floats for deterministic state"
        )
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_no_floats(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            _assert_no_floats(child, f"{path}[{i}]")


@contextmanager
def _block_network():
    original = socket.socket.connect

    def blocked(self, *args, **kwargs):
        raise DeterminismViolation(
            "network call attempted inside the environment — mock all external calls"
        )

    socket.socket.connect = blocked
    try:
        yield
    finally:
        socket.socket.connect = original


@contextmanager
def _block_filesystem():
    original = builtins.open

    def blocked(*args, **kwargs):
        raise DeterminismViolation(
            "filesystem access attempted inside the environment — mock all filesystem operations"
        )

    builtins.open = blocked
    try:
        yield
    finally:
        builtins.open = original


@contextmanager
def _guards(config: DeterminismConfig):
    with ExitStack() as stack:
        if config.mock_external_calls:
            stack.enter_context(_block_network())
        if config.mock_filesystem:
            stack.enter_context(_block_filesystem())
        yield


class _DeterministicFactory:
    """Wraps an initial-state factory to enforce the determinism config."""

    def __init__(self, inner: InitialStateFactory, config: DeterminismConfig) -> None:
        self._inner = inner
        self._config = config

    def create(self, ctx, options: dict) -> dict:
        if self._config.fresh_startup and hasattr(self._inner, "clear_cache"):
            self._inner.clear_cache()
        with _guards(self._config):
            state = self._inner.create(ctx, options)
        if self._config.integers_only:
            _assert_no_floats(state, "initial_state")
        return state


class _DeterministicTransitionEngine:
    """Wraps a TransitionEngine to enforce the determinism config on every step."""

    def __init__(self, inner: TransitionEngine, config: DeterminismConfig) -> None:
        self._inner = inner
        self._config = config
        self._lock = threading.Lock()

    @property
    def action_types(self) -> set[str]:
        return self._inner.action_types

    def apply(self, state: dict, action: dict, ctx) -> TransitionResult:
        if self._config.serialize_concurrency:
            with self._lock:
                return self._apply(state, action, ctx)
        return self._apply(state, action, ctx)

    def _apply(self, state: dict, action: dict, ctx) -> TransitionResult:
        with _guards(self._config):
            result = self._inner.apply(state, action, ctx)
        if self._config.integers_only:
            _assert_no_floats(result.state, "state")
        return result


class EnvBuilder:
    """Fluent builder for deterministic ForgeEnv instances.

    Wires the initial-state factory, transitions, verifiers, and reward
    function into a ForgeEnv whose factory and transition engine are wrapped
    to enforce a DeterminismConfig, then verifies determinism on build().
    """

    def __init__(self, name: str, domain: str, max_steps: int = 50) -> None:
        self._name = name
        self._domain = domain
        self._max_steps = max_steps
        self._default_task: dict | None = None
        self._factory: InitialStateFactory | None = None
        self._transitions: dict[str, Callable] = {}
        self._tool_specs: dict[str, ToolSpec] = {}
        self._verifiers: dict[str, Callable] = {}
        self._default_reward: Callable | None = None
        self._task_rewards: dict[str, Callable] = {}
        self._config = DeterminismConfig()
        self._computer_use: ComputerUse | None = None
        self._browser_use: BrowserUse | None = None

    def with_initial_state(self, factory: InitialStateFactory) -> "EnvBuilder":
        self._factory = factory
        return self

    def with_transition(
        self,
        action_type: str,
        handler: Callable,
        description: str = "",
        params: list[ToolParam] | None = None,
    ) -> "EnvBuilder":
        self._transitions[action_type] = handler
        self._tool_specs[action_type] = ToolSpec(
            name=action_type, description=description, params=params or []
        )
        return self

    def with_verifier(self, verifier_id: str, fn: Callable) -> "EnvBuilder":
        self._verifiers[verifier_id] = fn
        return self

    def with_composed_verifier(
        self, task, scenario=None, judge_client: Callable | None = None
    ) -> "EnvBuilder":
        """Register a multi-tiered verifier composed for ``task`` under its name.

        Wraps :class:`~forge.runtime.verifier_composer.VerifierComposer`: the
        task's success/failure conditions and the scenario ground truth are
        mapped onto the state, invariant, trajectory, judge, and negative tiers.
        """
        from forge.runtime.verifier_composer import VerifierComposer

        verifier = VerifierComposer().compose(
            task, scenario=scenario, judge_client=judge_client, verifier_id=task.name
        )
        return self.with_verifier(task.name, verifier)

    def with_scenario_scoring(
        self, mode, weights: dict | None = None, task_name: str | None = None
    ) -> "EnvBuilder":
        """Score episodes binary (pass/fail) or partial (weighted per-tier).

        Registers a reward function that turns the episode's verifier result
        into a :class:`RewardBreakdown` via the composer's scoring policy.
        """
        from forge.runtime.reward import RewardBreakdown
        from forge.runtime.verifier_composer import VerifierComposer

        composer = VerifierComposer()

        def reward_fn(state, trajectory, verifier_results, task):
            if not verifier_results:
                return RewardBreakdown(total_reward=0.0, components=[])
            return composer.score(verifier_results[0], mode, weights)

        return self.with_reward(reward_fn, task_name=task_name)

    def with_reward(self, fn: Callable, task_name: str | None = None) -> "EnvBuilder":
        if task_name is None:
            self._default_reward = fn
        else:
            self._task_rewards[task_name] = fn
        return self

    def with_default_task(self, task: dict) -> "EnvBuilder":
        self._default_task = task
        return self

    def with_determinism(self, config: DeterminismConfig) -> "EnvBuilder":
        self._config = config
        return self

    def with_computer_use(
        self, executor: Callable, schema: ComputerUseSchema | None = None
    ) -> "EnvBuilder":
        """Attach a VM/OS capability (shell, screenshots) to the environment."""
        self._computer_use = ComputerUse(schema=schema or ComputerUseSchema(), executor=executor)
        return self

    def with_browser_use(
        self, executor: Callable, schema: BrowserUseSchema | None = None
    ) -> "EnvBuilder":
        """Attach a browser capability (click/type/navigate) to the environment."""
        self._browser_use = BrowserUse(schema=schema or BrowserUseSchema(), executor=executor)
        return self

    def build(self, verify: bool = True, verify_seed: int = 42) -> ForgeEnv:
        if self._factory is None:
            raise EnvironmentBuildError(
                "EnvBuilder requires an initial state factory (with_initial_state)"
            )
        if not self._transitions:
            raise EnvironmentBuildError(
                "EnvBuilder requires at least one transition (with_transition)"
            )

        te = TransitionEngine()
        for action_type, handler in self._transitions.items():
            te.register(action_type, handler)

        ve = VerifierEngine()
        for verifier_id, fn in self._verifiers.items():
            ve.register(verifier_id, fn)

        re = RewardEngine()
        if self._default_reward:
            re.set_default(self._default_reward)
        for task_name, fn in self._task_rewards.items():
            re.register(task_name, fn)

        env = ForgeEnv(
            env_spec=EnvironmentSpec(
                name=self._name,
                domain=self._domain,
                max_steps=self._max_steps,
                default_task=self._default_task,
            ),
            initial_state_factory=_DeterministicFactory(self._factory, self._config),
            transition_engine=_DeterministicTransitionEngine(te, self._config),
            verifier_engine=ve,
            reward_engine=re,
            tool_specs=list(self._tool_specs.values()),
            computer_use=self._computer_use,
            browser_use=self._browser_use,
        )
        if verify:
            run_determinism_check(env, seed=verify_seed)
        return env
