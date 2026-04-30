from __future__ import annotations
from typing import TYPE_CHECKING, Protocol
import gymnasium as gym
from forge.runtime.action import ActionValidator
from forge.runtime.context import RuntimeContext
from forge.runtime.diff import compute_diff
from forge.runtime.reward import RewardEngine
from forge.runtime.snapshot import EnvironmentSpec, InvalidActionError, StepSnapshot
from forge.runtime.state import StateStore
from forge.runtime.trajectory import TrajectoryStore
from forge.runtime.transition import TransitionEngine
from forge.runtime.verifier import VerifierEngine

from forge.runtime.policy_engine import PolicyEngine, PolicyViolationResult
from forge.runtime.observation_filter import ObservationFilter

if TYPE_CHECKING:
    from forge.runtime.telemetry import TelemetryClient


class InitialStateFactory(Protocol):
    def create(self, ctx: RuntimeContext, options: dict) -> dict: ...


class ForgeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_spec: EnvironmentSpec,
        initial_state_factory: InitialStateFactory,
        transition_engine: TransitionEngine,
        verifier_engine: VerifierEngine,
        reward_engine: RewardEngine,
        telemetry: "TelemetryClient | None" = None,
        policy_engine: "PolicyEngine | None" = None,
        observation_filter: "ObservationFilter | None" = None,
    ) -> None:
        super().__init__()
        self.env_spec = env_spec
        self._factory = initial_state_factory
        self._transition_engine = transition_engine
        self._verifier_engine = verifier_engine
        self._reward_engine = reward_engine
        self._action_validator = ActionValidator(transition_engine.action_types)
        self._telemetry = telemetry
        self._policy_engine = policy_engine
        self._observation_filter = observation_filter

        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})

        self._ctx: RuntimeContext | None = None
        self._state_store: StateStore | None = None
        self._traj_store: TrajectoryStore | None = None
        self._current_task: dict | None = None
        self._step_count: int = 0
        self._episode_id: str | None = None
        self._invalid_action_count: int = 0
        self._total_reward: float = 0.0

    @property
    def action_types(self) -> frozenset:
        return frozenset(self._action_validator._valid_types)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        actual_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        opts = options or {}

        self._ctx = RuntimeContext(seed=actual_seed)
        self._episode_id = f"ep_{actual_seed:08x}"
        initial_state = self._factory.create(self._ctx, opts)
        self._state_store = StateStore(initial_state)
        self._traj_store = TrajectoryStore(self._episode_id)
        self._current_task = opts.get("task", self.env_spec.default_task)
        self._step_count = 0
        self._invalid_action_count = 0
        self._total_reward = 0.0

        _obs = self._state_store.get()
        if self._observation_filter:
            _obs = self._observation_filter.filter(_obs)
        return _obs, {
            "episode_id": self._episode_id,
            "task": self._current_task,
            "seed": actual_seed,
        }

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        if self._ctx is None:
            raise RuntimeError("Must call reset() before step()")

        state_before = self._state_store.get()
        hash_before = self._state_store.hash()

        validation_error = self._action_validator.validate(action)
        if validation_error:
            self._record_invalid_step(hash_before, action)
            return state_before, 0.0, False, False, {"error": validation_error}

        if self._policy_engine:
            violations = self._policy_engine.check(state_before, action)
            if violations:
                violation_events = [
                    {"type": "policy_violation", "rule_id": v.rule_id, "severity": v.severity}
                    for v in violations
                ]
                self._step_count += 1
                snapshot = StepSnapshot(
                    episode_id=self._episode_id,
                    step_index=self._step_count - 1,
                    state_hash_before=hash_before,
                    state_hash_after=hash_before,
                    action=action,
                    events=violation_events,
                    reward=0.0,
                    verifier_results=[],
                    diff={"added": {}, "changed": {}, "removed": {}},
                    terminated=False,
                    truncated=False,
                )
                self._traj_store.record(snapshot)
                if self._telemetry:
                    self._telemetry.record_step(snapshot)
                    self._telemetry.record_policy_violation(
                        step_index=snapshot.step_index,
                        action_type=action.get("type", ""),
                        violations=violations,
                    )
                return_obs = state_before
                if self._observation_filter:
                    return_obs = self._observation_filter.filter(return_obs)
                return return_obs, 0.0, False, False, {
                    "policy_violations": [v.__dict__ for v in violations],
                    "events": violation_events,
                }

        try:
            result = self._transition_engine.apply(state_before, action, self._ctx)
        except InvalidActionError as exc:
            self._record_invalid_step(hash_before, action)
            return state_before, 0.0, False, False, {"error": exc.to_dict()}

        self._state_store.apply(result.state)
        state_after = self._state_store.get()
        hash_after = self._state_store.hash()
        self._ctx.clock.advance()

        diff = compute_diff(state_before, state_after)
        # Build a trajectory that includes the current step's events so verifiers
        # can see actions taken in this step (e.g. email_replied for reply_to_customer).
        trajectory = self._traj_store.to_trajectory_with_events(result.events)
        verifier_results = self._verifier_engine.run_all(
            state_after, trajectory, self._current_task
        )
        task_with_meta = {**(self._current_task or {}), "invalid_action_count": self._invalid_action_count}
        reward_breakdown = self._reward_engine.compute(
            state_after, trajectory, verifier_results, task_with_meta
        )

        self._step_count += 1
        self._total_reward += reward_breakdown.total_reward
        terminated = any(vr.passed for vr in verifier_results)
        truncated = self._step_count >= self.env_spec.max_steps

        snapshot = StepSnapshot(
            episode_id=self._episode_id,
            step_index=self._step_count - 1,
            state_hash_before=hash_before,
            state_hash_after=hash_after,
            action=action,
            events=result.events,
            reward=reward_breakdown.total_reward,
            verifier_results=[vr.model_dump() for vr in verifier_results],
            diff=diff,
            terminated=terminated,
            truncated=truncated,
        )
        self._traj_store.record(snapshot)

        if self._telemetry:
            self._telemetry.record_step(snapshot)
        if (terminated or truncated) and self._telemetry:
            self._telemetry.complete_episode(self._total_reward, terminated, self._step_count)

        _step_obs = state_after
        if self._observation_filter:
            _step_obs = self._observation_filter.filter(_step_obs)
        return _step_obs, reward_breakdown.total_reward, terminated, truncated, {
            "episode_id": self._episode_id,
            "verifier_results": [vr.model_dump() for vr in verifier_results],
            "reward_breakdown": reward_breakdown.model_dump(),
            "events": result.events,
        }

    def _record_invalid_step(self, hash_before: str, action: dict) -> None:
        self._step_count += 1
        self._invalid_action_count += 1
        snapshot = StepSnapshot(
            episode_id=self._episode_id,
            step_index=self._step_count - 1,
            state_hash_before=hash_before,
            state_hash_after=hash_before,
            action=action,
            events=[],
            reward=0.0,
            verifier_results=[],
            diff={"added": {}, "changed": {}, "removed": {}},
            terminated=False,
            truncated=False,
        )
        self._traj_store.record(snapshot)
        if self._telemetry:
            self._telemetry.record_step(snapshot)
