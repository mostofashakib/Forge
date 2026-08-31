from __future__ import annotations
from typing import TYPE_CHECKING
import uuid
import gymnasium as gym
from forge.contracts import (
    Action,
    Environment,
    EpisodeEvaluation,
    ExecutionBackend,
    InitialStateProvider,
    MaxStepsTerminationPolicy,
    ObservationEncoder,
    PromptTemplate,
    Rubric,
    StateManager,
    StepOutcome,
    TaskSource,
    TerminationPolicy,
    ToolProvider,
    Task,
)
from forge.runtime.action import ActionValidator
from forge.runtime.context import RuntimeContext
from forge.runtime.diff import compute_diff
from forge.runtime.errors import ResetRequiredError
from forge.runtime.interaction import (
    BrowserUse,
    ComputerUse,
    MCPUse,
    ORPCUse,
    RESTUse,
    ToolUse,
    ToolUseSchema,
)
from forge.runtime.reward import RewardEngine
from forge.runtime.snapshot import EnvironmentSpec, InvalidActionError, StepSnapshot, ToolSpec
from forge.runtime.state import InProcessStateManager
from forge.runtime.trajectory import TrajectoryStore
from forge.runtime.transition import TransitionEngine
from forge.runtime.verifier import VerifierEngine
from forge.settings import determinism_enabled

from forge.runtime.policy_engine import PolicyEngine
from forge.runtime.observation_filter import ObservationFilter
from forge.runtime.task_source import StaticTaskSource
from forge.runtime.tools import SpecToolProvider
from forge.runtime.tasks import select_task, task_payload
from forge.runtime.control import SUBMIT_ACTION, is_submit_action

if TYPE_CHECKING:
    from forge.runtime.telemetry import TelemetrySink


# The pre-contracts name. Retained so existing imports keep working.
InitialStateFactory = InitialStateProvider


class ForgeEnv(gym.Env, Environment):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_spec: EnvironmentSpec,
        initial_state_provider: InitialStateProvider,
        transition_engine: TransitionEngine,
        verifier_engine: VerifierEngine,
        reward_engine: RewardEngine,
        telemetry: "TelemetrySink | None" = None,
        policy_engine: "PolicyEngine | None" = None,
        observation_filter: "ObservationFilter | None" = None,
        tool_specs: list[ToolSpec] | None = None,
        computer_use: ComputerUse | None = None,
        browser_use: BrowserUse | None = None,
        mcp_use: MCPUse | None = None,
        rest_use: RESTUse | None = None,
        orpc_use: ORPCUse | None = None,
        deterministic: bool | None = None,
        task_source: TaskSource | None = None,
        prompt_template: PromptTemplate | None = None,
        observation_encoder: ObservationEncoder | None = None,
        execution_backend: ExecutionBackend | None = None,
        termination_policy: TerminationPolicy | None = None,
        online_verifier_engine: VerifierEngine | None = None,
    ) -> None:
        super().__init__()
        self.env_spec = env_spec
        self._initial_state = initial_state_provider
        self._verifier_engine = verifier_engine
        self._reward_engine = reward_engine
        self._task_source = task_source or StaticTaskSource(
            [env_spec.default_task] if env_spec.default_task else []
        )
        self._prompt_template = prompt_template
        self._observations = observation_encoder or observation_filter or ObservationFilter()
        self._backend = execution_backend or transition_engine
        self._termination = termination_policy or MaxStepsTerminationPolicy(
            env_spec.max_steps
        )
        # Final verification is authoritative. An explicitly injected online
        # engine may run cheap monitors, but its results never become reward.
        self._online_verifier_engine = online_verifier_engine
        self._action_validator = ActionValidator(transition_engine.action_types)
        self._telemetry = telemetry
        self._policy_engine = policy_engine
        self._tool_specs = {spec.name: spec for spec in (tool_specs or [])}
        self._tools = SpecToolProvider(self._tool_specs.values())
        self._tool_use: ToolUse | None = None
        self.computer_use = computer_use
        self.browser_use = browser_use
        self.mcp_use = mcp_use
        self.rest_use = rest_use
        self.orpc_use = orpc_use
        self._deterministic = (
            determinism_enabled() if deterministic is None else deterministic
        )

        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})

        self._ctx: RuntimeContext | None = None
        self._state_store = InProcessStateManager({})
        self._traj_store: TrajectoryStore | None = None
        self._current_task: dict | None = None
        self._current_task_model: Task | None = None
        self._step_count: int = 0
        self._episode_id: str | None = None
        self._invalid_action_count: int = 0
        self._total_reward: float = 0.0
        self._evaluation: EpisodeEvaluation | None = None

    # ------------------------------------------------------------------
    # Composed Environment facade
    # ------------------------------------------------------------------

    @property
    def task_source(self) -> TaskSource:
        return self._task_source

    @property
    def initial_state(self) -> InitialStateProvider:
        return self._initial_state

    @property
    def observations(self) -> ObservationEncoder:
        return self._observations

    @property
    def backend(self) -> ExecutionBackend:
        return self._backend

    @property
    def state(self) -> StateManager:
        return self._state_store

    @property
    def rubric(self) -> Rubric:
        return self._reward_engine

    @property
    def termination(self) -> TerminationPolicy:
        return self._termination

    @property
    def prompt(self) -> PromptTemplate | None:
        return self._prompt_template

    @property
    def tools(self) -> ToolProvider:
        return self._tools

    @property
    def action_types(self) -> frozenset:
        return frozenset({*self._action_validator._valid_types, SUBMIT_ACTION})

    @property
    def current_task(self) -> Task | None:
        """The task selected for the active episode, if reset has run."""
        return self._current_task_model

    def current_trajectory(self):
        """Full recorded trajectory of the in-progress episode."""
        if self._traj_store is None:
            raise ResetRequiredError("Must call reset() before reading the trajectory")
        return self._traj_store.to_trajectory()

    def tool_surface(self) -> list[ToolSpec]:
        """Every tool the agent may call, with params — bare spec if undocumented."""
        specs = []
        for name in sorted(self.action_types):
            if name == SUBMIT_ACTION:
                specs.append(ToolSpec(
                    name=SUBMIT_ACTION,
                    description="Finish the episode and grade the current state",
                ))
            else:
                specs.append(self._tool_specs.get(name, ToolSpec(name=name)))
        return specs

    @property
    def tool_use(self) -> ToolUse:
        """Tool-calling contract for this env: validated calls dispatch to step().

        Built once — the tool surface is fixed after construction.
        """
        if self._tool_use is None:
            self._tool_use = ToolUse(
                schema=ToolUseSchema(tools=self.tool_surface()), executor=self.step
            )
        return self._tool_use

    def capabilities(self) -> list[str]:
        """Interaction modes the agent has access to in this environment.

        Every env has ``tool_use``; the rest are present only when attached, so
        an env exposes exactly the modalities its domain needs (MCP tools, REST
        endpoints, oRPC procedures, OS shell, browser).
        """
        modes = ["tool_use"]
        for cap in (self.mcp_use, self.rest_use, self.orpc_use, self.computer_use, self.browser_use):
            if cap is not None:
                modes.append(cap.name)
        return modes

    def capability_surface(self) -> dict[str, list[ToolSpec]]:
        """Every action the agent can take, grouped by interaction modality.

        The full tool surface across modalities: core tool calls plus any
        attached MCP tools, REST endpoints, oRPC procedures, OS primitives, and
        browser primitives — each rendered as ``ToolSpec`` entries.
        """
        surface: dict[str, list[ToolSpec]] = {"tool_use": self.tool_surface()}
        for cap in (self.mcp_use, self.rest_use, self.orpc_use, self.computer_use, self.browser_use):
            if cap is not None:
                surface[cap.name] = cap.schema.tool_specs()
        return surface

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed if self._deterministic else None)
        actual_seed = (
            seed
            if self._deterministic and seed is not None
            else int(self.np_random.integers(0, 2**31))
        )
        opts = options or {}

        self._ctx = RuntimeContext(seed=actual_seed, deterministic=self._deterministic)
        self._episode_id = (
            f"ep_{actual_seed:08x}"
            if self._deterministic
            else f"ep_{uuid.uuid4().hex[:12]}"
        )
        initial_state = self._initial_state.reset(
            self._ctx, seed=actual_seed, options=opts
        )
        self._state_store = InProcessStateManager(initial_state)
        self._traj_store = TrajectoryStore(self._episode_id)
        self._current_task_model = select_task(
            self._task_source,
            seed=actual_seed,
            options=opts,
            fallback=self.env_spec.default_task,
        )
        self._current_task = task_payload(self._current_task_model)
        self._step_count = 0
        self._invalid_action_count = 0
        self._total_reward = 0.0
        self._evaluation = None

        return self._observe(self._state_store.get()), {
            "episode_id": self._episode_id,
            "task": self._current_task,
            "seed": actual_seed,
        }

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        if self._ctx is None:
            raise ResetRequiredError("Must call reset() before step()")
        if self._evaluation is not None:
            raise RuntimeError("Episode is finalized; call reset() before stepping again")

        state_before = self._state_store.get()
        hash_before = self._state_store.hash()

        if is_submit_action(action):
            return self._submit(state_before, hash_before)

        validation_error = self._action_validator.validate(action)
        if validation_error:
            return self._stationary_step_result(
                state_before,
                hash_before,
                action,
                events=[],
                info={"error": validation_error},
                invalid=True,
            )

        if self._policy_engine:
            violations = self._policy_engine.check(state_before, action)
            if violations:
                return self._policy_violation_result(
                    state_before, hash_before, action, violations
                )

        try:
            result = self._backend.execute(
                Action.from_dict(action), state_before, self._ctx
            )
        except InvalidActionError as exc:
            return self._stationary_step_result(
                state_before,
                hash_before,
                action,
                events=[],
                info={"error": exc.to_dict()},
                invalid=True,
            )

        self._state_store.apply(result.state)
        state_after = self._state_store.get()
        hash_after = self._state_store.hash()
        self._ctx.clock.advance()

        diff = compute_diff(state_before, state_after)
        self._step_count += 1
        snapshot = StepSnapshot(
            episode_id=self._episode_id,
            step_index=self._step_count - 1,
            state_hash_before=hash_before,
            state_hash_after=hash_after,
            action=action,
            events=result.events,
            reward=0.0,
            verifier_results=[],
            diff=diff,
            terminated=False,
            truncated=False,
        )
        trajectory = self._traj_store.to_trajectory_with_step(snapshot)
        monitor_results = (
            self._online_verifier_engine.run_all(
                state_after, trajectory, self._current_task
            )
            if self._online_verifier_engine is not None
            else []
        )
        termination = self._termination.check(
            StepOutcome(
                step_index=self._step_count - 1,
                score=max((item.score for item in monitor_results), default=0.0),
                reward=0.0,
                state_hash=hash_after,
                verifier_results=monitor_results,
            )
        )
        terminated = termination is not None and not termination.truncated
        truncated = termination is not None and termination.truncated
        reward = 0.0
        verifier_results = monitor_results
        reward_breakdown = None
        if termination is not None:
            evaluation = self._evaluate_trajectory(
                state_after, trajectory, termination.reason
            )
            reward = evaluation.total_reward
            verifier_results = evaluation.verification_results
            reward_breakdown = evaluation.reward
        snapshot.reward = reward
        snapshot.verifier_results = [item.model_dump() for item in verifier_results]
        snapshot.terminated = terminated
        snapshot.truncated = truncated
        self._record_snapshot(snapshot)
        if (terminated or truncated) and self._telemetry:
            self._telemetry.complete_episode(
                self._total_reward,
                bool(self._evaluation and self._evaluation.passed),
                self._step_count,
                termination.reason,
            )

        info = {
            "episode_id": self._episode_id,
            "events": result.events,
            "termination_reason": termination.reason if termination else None,
        }
        if termination is not None and reward_breakdown is not None:
            info.update({
                "passed": self._evaluation.passed,
                "verifier_results": [item.model_dump() for item in verifier_results],
                "reward_breakdown": reward_breakdown.model_dump(),
            })
        return self._observe(state_after), reward, terminated, truncated, info

    def finalize_episode(self, reason: str = "external") -> EpisodeEvaluation:
        """Grade the active episode once; repeated calls return the same verdict."""
        if self._ctx is None or self._traj_store is None:
            raise ResetRequiredError("Must call reset() before finalizing an episode")
        return self._evaluate_trajectory(
            self._state_store.get(), self._traj_store.to_trajectory(), reason
        )

    def _evaluate_trajectory(self, state, trajectory, reason: str) -> EpisodeEvaluation:
        if self._evaluation is not None:
            return self._evaluation
        verifier_results = self._verifier_engine.run_all(
            state, trajectory, self._current_task
        )
        task_with_meta = {
            **(self._current_task or {}),
            "invalid_action_count": self._invalid_action_count,
        }
        reward = self._reward_engine.compute(
            state, trajectory, verifier_results, task_with_meta
        )
        self._evaluation = EpisodeEvaluation(
            passed=any(result.passed for result in verifier_results),
            reward=reward,
            verification_results=verifier_results,
            reason=reason,
        )
        self._total_reward = reward.total_reward
        return self._evaluation

    def _submit(self, state: dict, state_hash: str):
        self._step_count += 1
        snapshot = StepSnapshot(
            episode_id=self._episode_id,
            step_index=self._step_count - 1,
            state_hash_before=state_hash,
            state_hash_after=state_hash,
            action={"type": SUBMIT_ACTION},
            events=[],
            reward=0.0,
            verifier_results=[],
            diff={"added": {}, "changed": {}, "removed": {}},
            terminated=True,
            truncated=False,
        )
        trajectory = self._traj_store.to_trajectory_with_step(snapshot)
        evaluation = self._evaluate_trajectory(state, trajectory, "submitted")
        snapshot.reward = evaluation.total_reward
        snapshot.verifier_results = [
            item.model_dump() for item in evaluation.verification_results
        ]
        self._record_snapshot(snapshot)
        if self._telemetry:
            self._telemetry.complete_episode(
                evaluation.total_reward,
                evaluation.passed,
                self._step_count,
                "submitted",
            )
        return self._observe(state), evaluation.total_reward, True, False, {
            "episode_id": self._episode_id,
            "passed": evaluation.passed,
            "verifier_results": snapshot.verifier_results,
            "reward_breakdown": evaluation.reward.model_dump(),
            "events": [],
            "termination_reason": "submitted",
        }

    def _stationary_step_result(
        self,
        state: dict,
        state_hash: str,
        action: dict,
        *,
        events: list[dict],
        info: dict,
        invalid: bool = False,
    ) -> tuple[dict, float, bool, bool, dict]:
        """Record a rejected/no-op action while still enforcing episode budgets."""
        self._step_count += 1
        if invalid:
            self._invalid_action_count += 1
        snapshot = StepSnapshot(
            episode_id=self._episode_id,
            step_index=self._step_count - 1,
            state_hash_before=state_hash,
            state_hash_after=state_hash,
            action=action,
            events=events,
            reward=0.0,
            verifier_results=[],
            diff={"added": {}, "changed": {}, "removed": {}},
            terminated=False,
            truncated=False,
        )
        termination = self._termination.check(
            StepOutcome(step_index=snapshot.step_index, state_hash=state_hash)
        )
        reward = 0.0
        if termination is not None:
            snapshot.terminated = not termination.truncated
            snapshot.truncated = termination.truncated
            trajectory = self._traj_store.to_trajectory_with_step(snapshot)
            evaluation = self._evaluate_trajectory(
                state, trajectory, termination.reason
            )
            reward = evaluation.total_reward
            snapshot.reward = reward
            snapshot.verifier_results = [
                item.model_dump() for item in evaluation.verification_results
            ]
            info.update({
                "passed": evaluation.passed,
                "verifier_results": snapshot.verifier_results,
                "reward_breakdown": evaluation.reward.model_dump(),
                "termination_reason": termination.reason,
            })
        self._record_snapshot(snapshot)
        if termination is not None and self._telemetry:
            self._telemetry.complete_episode(
                reward, self._evaluation.passed, self._step_count, termination.reason
            )
        return (
            self._observe(state),
            reward,
            bool(termination and not termination.truncated),
            bool(termination and termination.truncated),
            info,
        )

    def _record_invalid_step(self, hash_before: str, action: dict) -> None:
        """Legacy helper retained for callers that only need to record a step."""
        self._stationary_step_result(
            self._state_store.get(),
            hash_before,
            action,
            events=[],
            info={},
            invalid=True,
        )

    def _policy_violation_result(
        self, state: dict, state_hash: str, action: dict, violations: list
    ) -> tuple[dict, float, bool, bool, dict]:
        violation_events = [
            {"type": "policy_violation", "rule_id": item.rule_id, "severity": item.severity}
            for item in violations
        ]
        result = self._stationary_step_result(
            state,
            state_hash,
            action,
            events=violation_events,
            info={
                "policy_violations": [item.__dict__ for item in violations],
                "events": violation_events,
            },
        )
        if self._telemetry:
            self._telemetry.record_policy_violation(
                step_index=self._step_count - 1,
                action_type=action.get("type", ""),
                violations=violations,
            )
        return result

    def _record_snapshot(self, snapshot: StepSnapshot) -> None:
        self._traj_store.record(snapshot)
        if self._telemetry:
            self._telemetry.record_step(snapshot)

    def _observe(self, state: dict) -> dict:
        if self._ctx is None:
            raise ResetRequiredError("Must call reset() before observing state")
        return self._observations.encode(state, self._ctx).payload
