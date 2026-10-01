from __future__ import annotations
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timezone
from pathlib import Path

import httpx

from forge.contracts import (
    Action,
    DeadEndTerminationPolicy,
    Environment,
    EpisodeController,
    MaxStepsTerminationPolicy,
    StepOutcome,
    Task,
)
from forge.envgen.agents.container_agent import ContainerAgentBase
from forge.contracts.persona import PersonaPopulation
from forge.contracts.transport import DEFAULT_TIMEOUT_S
from forge.envgen.container_env_base import ContainerEnvBase
from forge.personas.engine import PersonaEngine
from forge.envgen.episode_base import (
    BaseEpisodeConfig,
    BaseEpisodeResult,
    TerminationMonitor as TerminationMonitor,
    TrajectoryWriter,
)
from forge.envgen.objective import ObjectiveScorer
from forge.envgen.objective_grading import ObjectiveGrader
from forge.runtime.tools import OpenAPIToolProvider
from forge.runtime.context import RuntimeContext
from forge.runtime.prompting import ForgeAgentPromptTemplate
from forge.runtime.reward import ObjectiveScoreRubric
from forge.runtime.task_source import StaticTaskSource
from forge.runtime.tasks import select_task
from forge.runtime.control import SUBMIT_ENDPOINT, is_submit_action
from forge.schema.state_schema import StateSchemaManifest
from forge.runtime.reliability import (
    InfrastructureCrash,
    DivergenceError,
    REASON_CONTAINER_UNREACHABLE,
    REASON_RESET_FAILED,
    REASON_STATE_READ_FAILED,
    REASON_PROVIDER_ERROR,
    SnapshotManager,
    get_reliability_settings,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ContainerEpisodeRunner",
    "EpisodeConfig",
    "EpisodeResult",
    "TerminationMonitor",
]


# ---------------------------------------------------------------------------
# Config & result data classes
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class EpisodeConfig(BaseEpisodeConfig):
    base_url: str
    max_steps: int = 50
    consecutive_below_threshold: int = 8
    # httpx timeout per request (seconds)
    http_timeout: float = DEFAULT_TIMEOUT_S
    diff_floor: float = 0.1
    # Simulated humans who act alongside the agent. Loaded from the
    # environment's custom/config.yaml; `None` means the environment runs
    # with nobody else in it.
    personas: PersonaPopulation | None = None


@dataclass
class HashNormalizer:
    manifest: StateSchemaManifest | None

    def hash(self, state: dict) -> str:
        if self.manifest is None:
            canonical = json.dumps(state, sort_keys=True)
        else:
            stable = self.manifest.stable_fields()
            filtered = {k: v for k, v in state.items() if k in stable}
            canonical = json.dumps(filtered, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class StepRecord:
    step_index: int
    state_before: dict
    action: dict
    state_after: dict
    reward: float
    objective_score: float
    state_hash_before: str
    state_hash_after: str
    terminated: bool
    truncated: bool
    termination_reason: str | None


@dataclass(kw_only=True)
class EpisodeResult(BaseEpisodeResult):
    episode_id: str
    config: EpisodeConfig
    steps: list[StepRecord] = field(default_factory=list)

    def _step_to_dict(self, step) -> dict:
        return {
            "step_index": step.step_index,
            "state_before": step.state_before,
            "action": step.action,
            "state_after": step.state_after,
            "reward": step.reward,
            "objective_score": step.objective_score,
            "state_hash_before": step.state_hash_before,
            "state_hash_after": step.state_hash_after,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "termination_reason": step.termination_reason,
        }

    def summary(self) -> dict:
        return {**super().summary(), "episode_id": self.episode_id}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

@dataclass
class _EpisodeRun:
    """What one episode's step loop works with, beyond the runner's own config."""

    agent: Any
    result: "EpisodeResult"
    available_actions: list[dict]
    writer: Any
    dead_end: Any
    max_steps: Any
    replay_steps: list[dict] | None = None
    snapshot_manager: Any = None
    model_output_log: list[dict] | None = None


class ContainerEpisodeRunner(EpisodeController):
    """Runs one or more agent episodes against a containerized FastAPI environment.

    The runner communicates with the app entirely over HTTP:
      - GET  /forge/state  → current state dict
      - POST /forge/reset  → reset to initial state
      - GET  /openapi.json → discover available action endpoints
      - POST /<action>     → execute an action

    Stopping conditions are cheap and independent of grading: explicit submit,
    repeated unchanged state, or the step budget. Objective scoring runs once
    after the loop and supplies the authoritative episode reward.
    """

    def __init__(
        self,
        config: EpisodeConfig,
        scorer: ObjectiveScorer | None = None,
        manifest: StateSchemaManifest | None = None,
        environment: Environment | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._cfg = config
        self._scorer = scorer or ObjectiveScorer()
        self._grader = ObjectiveGrader(
            self._scorer, objective=config.objective, success_threshold=config.success_threshold
        )
        self._normalizer = HashNormalizer(manifest=manifest)
        self._manifest = manifest
        self._http = http_client or httpx.Client(
            base_url=config.base_url,
            timeout=config.http_timeout,
        )
        self._environment = environment
        self._runtime_ctx: RuntimeContext | None = None
        self._selected_task: Task | None = None
        self._tool_provider: OpenAPIToolProvider | None = None  # built on first use

    # ------------------------------------------------------------------
    # Startup health check
    # ------------------------------------------------------------------

    def wait_for_health(self, max_retries: int = 40, delay: float = 3.0) -> bool:
        """Poll /forge/health until the app responds or retries are exhausted.

        The FastAPI container may take several seconds to start uvicorn after
        Docker reports the container as "running".  Without this wait, all
        HTTP calls fail immediately with ECONNREFUSED.
        """
        logger.info(
            "[runner] waiting for %s to become healthy (up to %ds)…",
            self._cfg.base_url,
            int(max_retries * delay),
        )
        for attempt in range(max_retries):
            try:
                resp = self._http.get("/forge/health", timeout=5.0)
                if resp.is_success:
                    logger.info("[runner] %s healthy after %d attempt(s)", self._cfg.base_url, attempt + 1)
                    return True
            except Exception as exc:
                logger.debug("[runner] health attempt %d/%d: %s", attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                time.sleep(delay)
        logger.error("[runner] %s did not become healthy after %d attempts", self._cfg.base_url, max_retries)
        return False

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> dict:
        """Observe DB-backed state via "/forge/state" endpoint or environment state adapter."""
        state = self.environment.state.get()
        ctx = self._runtime_ctx or RuntimeContext(seed=0, deterministic=False)
        return self.environment.observations.encode(state, ctx).payload

    def _reset(self, seed: int | None = None) -> dict:
        # Thread the seed so the app rebuilds a reproducible, seed-specific
        # starting universe; an unseeded reset restores the fixed baseline.
        from forge.settings import determinism_enabled

        actual_seed = seed if seed is not None and determinism_enabled() else 0
        self._runtime_ctx = RuntimeContext(
            seed=actual_seed,
            deterministic=seed is not None and determinism_enabled(),
        )
        self._selected_task = select_task(
            self.environment.task_source,
            seed=actual_seed,
            options={},
            fallback=Task(id="objective", objective=self._cfg.objective),
        )
        state = self.environment.initial_state.reset(
            self._runtime_ctx,
            seed=seed if seed is not None and determinism_enabled() else None,
            options={},
        )
        # This controller drives the environment's collaborators directly
        # rather than through `ContainerEnvBase.reset`, so the cast has to be
        # resolved here too — otherwise personas would exist on the env and
        # never appear in an actual agent run.
        self.environment.personas.reset(actual_seed)
        return self.environment.observations.encode(state, self._runtime_ctx).payload

    @property
    def environment(self) -> Environment:
        """The composed environment driven by this controller."""
        if self._environment is None:
            self._environment = ContainerEnvBase(
                self._cfg.base_url,
                client=self._http,
                timeout=self._cfg.http_timeout,
                max_steps=self._cfg.max_steps,
                task_source=StaticTaskSource([
                    Task(id="objective", objective=self._cfg.objective)
                ]),
                prompt_template=ForgeAgentPromptTemplate(),
                rubric=ObjectiveScoreRubric(self._cfg.diff_floor),
                personas=PersonaEngine(self._cfg.personas)
                if self._cfg.personas is not None
                else None,
            )
        return self._environment

    @property
    def tool_provider(self) -> OpenAPIToolProvider:
        """This app's action surface, discovered from its OpenAPI schema.

        Built lazily so it binds whatever client is on `self._http` at first
        use, and cached so the manifest is fetched once per episode.
        """
        if self._tool_provider is None:
            self._tool_provider = OpenAPIToolProvider(self._http)
        return self._tool_provider

    def _discover_actions(self) -> list[dict]:
        """Build an action manifest from /openapi.json. Cached after first call."""
        return self.tool_provider.action_manifest()

    def _execute_action(self, action: dict, step_index: int = 0) -> dict | None:
        endpoint = action.get("endpoint", "")
        payload = action.get("payload", {})
        try:
            ctx = self._runtime_ctx or RuntimeContext(seed=0, deterministic=False)
            result = self.environment.backend.execute(
                Action(type=endpoint, params={"__payload__": payload}),
                self.environment.state.get(),
                ctx,
            )
            return self._tick_personas(action, result, ctx, step_index)
        except Exception as exc:
            logger.debug("[runner] action %s failed: %s", endpoint, exc)
            return None

    def _tick_personas(self, action: dict, result, ctx, step_index: int) -> dict:
        """Give the cast its turn on the world the agent just changed.

        `ContainerEnvBase.step` does this itself, but this controller bypasses
        `step` and calls the backend directly, so the tick has to happen here.
        The persona turns are recorded on the engine's transcript, which
        `run_episode` reads back into the episode result.
        """
        personas = self.environment.personas
        if not personas.enabled:
            return result.state
        tick = personas.tick(
            backend=self.environment.backend,
            state=result.state,
            ctx=ctx,
            step_index=step_index,
            agent_action={"type": action.get("endpoint", "")},
            events=result.events,
        )
        return tick.state

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------

    def run_episode(
        self,
        agent: ContainerAgentBase,
        *,
        episode_id: str | None = None,
        seed: int | None = None,
        jsonl_path: Path | None = None,
        replay_steps: list[dict] | None = None,
        resume_from_step: int = 0,
        snapshot_manager: SnapshotManager | None = None,
        model_output_log: list[dict] | None = None,
    ) -> EpisodeResult:
        if episode_id is None:
            episode_id = f"cep_{secrets.token_hex(6)}"

        cfg = self._cfg
        result = EpisodeResult(episode_id=episode_id, config=cfg)
        available_actions, state = self._start(episode_id, seed)
        # Write each step as it happens so a crash mid-episode still leaves a
        # durable, replayable partial trace (not just an all-or-nothing dump).
        writer = TrajectoryWriter(jsonl_path, result) if jsonl_path is not None else None
        run = _EpisodeRun(
            agent=agent,
            result=result,
            available_actions=available_actions,
            writer=writer,
            dead_end=DeadEndTerminationPolicy(cfg.dead_end_patience),
            max_steps=MaxStepsTerminationPolicy(cfg.max_steps),
            replay_steps=replay_steps,
            snapshot_manager=snapshot_manager or SnapshotManager(
                env_type="general", interval=get_reliability_settings().snapshot_interval
            ),
            model_output_log=model_output_log,
        )
        try:
            self._run_steps(run, state, resume_from_step=resume_from_step)
        finally:
            if writer is not None:
                writer.close()
        return result

    def _start(self, episode_id: str, seed: int | None) -> tuple[list[dict], dict]:
        """(available actions, initial state) once the app is healthy and reset."""
        # Wait for the container app to be ready before doing anything else.
        # This is the root cause of ECONNREFUSED: Docker marks the container
        # as "running" before uvicorn inside finishes startup.
        if not self.wait_for_health():
            raise InfrastructureCrash(
                reason=REASON_CONTAINER_UNREACHABLE,
                detail=f"container_unreachable: {self._cfg.base_url}",
                step=0,
            )

        available_actions = [
            *self._discover_actions(),
            {
                "endpoint": SUBMIT_ENDPOINT,
                "method": "CONTROL",
                "description": "Finish the episode and grade the current state",
            },
        ]
        try:
            state = self._reset(seed=seed)
        except Exception as exc:
            logger.error("[%s] reset failed: %s", episode_id, exc)
            raise InfrastructureCrash(
                reason=REASON_RESET_FAILED,
                detail=f"reset failed: {exc}",
                original_exc=exc,
                step=0,
            )
        return available_actions, state

    def _run_steps(self, run: "_EpisodeRun", state: dict, *, resume_from_step: int = 0) -> None:
        for step_idx in range(resume_from_step, self._cfg.max_steps):
            state = self._step(run, step_idx, state)
            if state is None:
                break
        run.result.completed_at = datetime.now(timezone.utc)

    def _step(self, run: "_EpisodeRun", step_idx: int, state: dict) -> dict | None:
        """Take one step. Returns the next state, or None once the episode has ended."""
        cfg, result, episode_id = self._cfg, run.result, run.result.episode_id
        state_hash_before = self._normalizer.hash(state)

        if run.snapshot_manager is not None and run.snapshot_manager.should_snapshot(step_idx):
            run.snapshot_manager.capture_snapshot(
                step=step_idx,
                state_hash=state_hash_before,
                http_client=self._http,
                base_url=cfg.base_url,
            )

        action = self._next_action(
            run.agent, state, cfg.objective, run.available_actions, step_idx, state_hash_before,
            run.replay_steps, episode_id,
        )
        # Write model output before action runs
        if run.model_output_log is not None:
            run.model_output_log.append({
                "step_index": step_idx,
                "action": action,
                "state_hash_before": state_hash_before,
            })
        action = self._known_action(action, run.available_actions, step_idx, episode_id)

        if is_submit_action(action):
            self._submit(result, state, action, step_idx, run.writer)
            return None

        self._execute_action(action, step_idx)
        new_state = self._observe_after(step_idx, episode_id)
        state_hash_after = self._normalizer.hash(new_state)

        # Stopping checks use only deterministic state progress and budget.
        outcome = StepOutcome(step_index=step_idx, state_hash=state_hash_after)
        decision = run.dead_end.check(outcome) or run.max_steps.check(outcome)
        step = StepRecord(
            step_index=step_idx,
            state_before=state,
            action=action,
            state_after=new_state,
            reward=0.0,
            objective_score=0.0,
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
            terminated=bool(decision and not decision.truncated),
            truncated=bool(decision and decision.truncated),
            termination_reason=decision.reason if decision else None,
        )
        result.steps.append(step)
        logger.info(
            "[%s] step %02d/%d  hash=%s→%s%s",
            episode_id,
            step_idx + 1,
            cfg.max_steps,
            state_hash_before[:6],
            state_hash_after[:6],
            f"  → {decision.reason}" if decision else "",
        )

        if decision is None:
            if run.writer is not None:
                run.writer.record(step)
            return new_state
        result.termination_reason = decision.reason or (
            "truncated" if decision.truncated else "unknown"
        )
        derived_diff = self._manifest.derived_diff(state, new_state) if self._manifest else {}
        self._finalize_result(
            result,
            new_state,
            action,
            derived_diff=derived_diff or None,
            state_changed=state_hash_before != state_hash_after,
        )
        self._record_graded(step, result, run.writer)
        return None

    @staticmethod
    def _next_action(
        agent, state, objective, available_actions, step_idx, state_hash_before, replay_steps, episode_id,
    ) -> dict:
        """The logged action when replaying, else the agent's choice."""
        if replay_steps is not None and step_idx < len(replay_steps):
            replayed = replay_steps[step_idx]
            expected_hash = replayed.get("state_hash_before")
            if expected_hash and expected_hash != state_hash_before:
                raise DivergenceError(
                    step=step_idx,
                    expected_hash=expected_hash,
                    actual_hash=state_hash_before,
                )
            return replayed["action"]
        try:
            return agent.act(state, objective, available_actions)
        except Exception as exc:
            logger.warning("[%s] step %d: agent.act failed: %s", episode_id, step_idx, exc)
            raise InfrastructureCrash(
                reason=REASON_PROVIDER_ERROR,
                detail=f"agent.act failed at step {step_idx}: {exc}",
                original_exc=exc,
                step=step_idx,
            )

    @staticmethod
    def _known_action(action: dict, available_actions: list[dict], step_idx: int, episode_id: str) -> dict:
        """Replace an endpoint the app does not expose with its first real one."""
        if is_submit_action(action) or not available_actions:
            return action
        if any(a["endpoint"] == action.get("endpoint") for a in available_actions):
            return action
        logger.debug(
            "[%s] step %d: agent chose unknown endpoint %r — falling back",
            episode_id, step_idx, action.get("endpoint"),
        )
        return {"endpoint": available_actions[0]["endpoint"], "payload": {}}

    def _submit(self, result, state: dict, action: dict, step_idx: int, writer) -> None:
        """Record the submit as a terminal step and grade the state as it stands."""
        state_hash = self._normalizer.hash(state)
        step = StepRecord(
            step_index=step_idx,
            state_before=state,
            action={"endpoint": SUBMIT_ENDPOINT, "payload": {}},
            state_after=state,
            reward=0.0,
            objective_score=0.0,
            state_hash_before=state_hash,
            state_hash_after=state_hash,
            terminated=True,
            truncated=False,
            termination_reason="submitted",
        )
        result.steps.append(step)
        result.termination_reason = "submitted"
        self._finalize_result(result, state, action)
        self._record_graded(step, result, writer)

    def _observe_after(self, step_idx: int, episode_id: str) -> dict:
        try:
            return self._get_state()
        except Exception as exc:
            logger.warning("[%s] step %d: get_state failed: %s", episode_id, step_idx, exc)
            raise InfrastructureCrash(
                reason=REASON_STATE_READ_FAILED,
                detail=f"get_state failed at step {step_idx}: {exc}",
                original_exc=exc,
                step=step_idx,
            )

    @staticmethod
    def _record_graded(step: StepRecord, result, writer) -> None:
        step.reward = result.total_reward
        step.objective_score = result.final_objective_score
        if writer is not None:
            writer.record(step)

    def _finalize_result(
        self,
        result: EpisodeResult,
        state: dict,
        action: dict,
        *,
        derived_diff: dict | None = None,
        state_changed: bool = False,
    ) -> None:
        """Run the container's objective judge and rubric exactly once."""
        self._grader.grade(
            result,
            state,
            action,
            task=self._selected_task or Task(id="objective", objective=self._cfg.objective),
            rubric=self.environment.rubric,
            derived_diff=derived_diff,
            state_changed=state_changed or any(
                step.state_hash_before != step.state_hash_after for step in result.steps
            ),
        )

    # ------------------------------------------------------------------
    # Multi-episode rollout
    # ------------------------------------------------------------------

    def run_rollout(
        self,
        agent: ContainerAgentBase,
        num_episodes: int,
        seed_start: int = 0,
        output_dir: Path | None = None,
    ) -> list[EpisodeResult]:
        """Run `num_episodes` episodes in sequence. Returns all results."""
        results: list[EpisodeResult] = []
        for i in range(num_episodes):
            episode_id = f"cep_{seed_start + i:08x}_{secrets.token_hex(3)}"
            jsonl_path: Path | None = None
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                jsonl_path = output_dir / f"{episode_id}.jsonl"
            logger.info(
                "[runner] episode %d/%d  id=%s", i + 1, num_episodes, episode_id
            )
            # Each episode gets a distinct, reproducible seed so rollouts cover
            # different starting universes deterministically.
            result = self.run_episode(
                agent, episode_id=episode_id, jsonl_path=jsonl_path, seed=seed_start + i
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._environment is not None:
            self._environment.backend.close()
        self._http.close()

    def __enter__(self) -> "ContainerEpisodeRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
