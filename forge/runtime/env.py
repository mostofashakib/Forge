from __future__ import annotations
from typing import Protocol
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
    ) -> None:
        super().__init__()
        self.env_spec = env_spec
        self._factory = initial_state_factory
        self._transition_engine = transition_engine
        self._verifier_engine = verifier_engine
        self._reward_engine = reward_engine
        self._action_validator = ActionValidator(transition_engine.action_types)

        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})

        self._ctx: RuntimeContext | None = None
        self._state_store: StateStore | None = None
        self._traj_store: TrajectoryStore | None = None
        self._current_task: dict | None = None
        self._step_count: int = 0
        self._episode_id: str | None = None

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

        return self._state_store.get(), {
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
        reward_breakdown = self._reward_engine.compute(
            state_after, trajectory, verifier_results, self._current_task
        )

        self._step_count += 1
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

        return state_after, reward_breakdown.total_reward, terminated, truncated, {
            "episode_id": self._episode_id,
            "verifier_results": [vr.model_dump() for vr in verifier_results],
            "reward_breakdown": reward_breakdown.model_dump(),
            "events": result.events,
        }

    def _record_invalid_step(self, hash_before: str, action: dict) -> None:
        self._step_count += 1
        self._traj_store.record(
            StepSnapshot(
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
        )
