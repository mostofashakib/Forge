from __future__ import annotations
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from forge.contracts import (
    Action,
    CheckResult,
    DeadEndTerminationPolicy,
    Environment,
    EpisodeEvaluation,
    EpisodeController,
    MaxStepsTerminationPolicy,
    StepOutcome,
    Task,
    VerificationResult,
)
from forge.envgen.agents.container_agent import ContainerAgentBase
from forge.envgen.container_env_base import ContainerEnvBase
from forge.envgen.episode_base import (
    BaseEpisodeConfig,
    BaseEpisodeResult,
    TerminationMonitor as TerminationMonitor,
    TrajectoryWriter,
)
from forge.envgen.objective import ObjectiveScorer
from forge.runtime.tools import OpenAPIToolProvider
from forge.runtime.context import RuntimeContext
from forge.runtime.prompting import ForgeAgentPromptTemplate
from forge.runtime.reward import ObjectiveScoreRubric
from forge.runtime.task_source import StaticTaskSource
from forge.runtime.tasks import select_task
from forge.runtime.trajectory import Trajectory
from forge.runtime.control import SUBMIT_ENDPOINT, is_submit_action
from forge.schema.state_schema import StateSchemaManifest

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
    http_timeout: float = 15.0
    diff_floor: float = 0.1


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
    ) -> None:
        self._cfg = config
        self._scorer = scorer or ObjectiveScorer()
        self._normalizer = HashNormalizer(manifest=manifest)
        self._manifest = manifest
        self._http = httpx.Client(
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

    def wait_for_health(self, max_retries: int = 15, delay: float = 3.0) -> bool:
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

    def _execute_action(self, action: dict) -> dict | None:
        endpoint = action.get("endpoint", "")
        payload = action.get("payload", {})
        try:
            ctx = self._runtime_ctx or RuntimeContext(seed=0, deterministic=False)
            result = self.environment.backend.execute(
                Action(type=endpoint, params={"__payload__": payload}),
                self.environment.state.get(),
                ctx,
            )
            return result.state
        except Exception as exc:
            logger.debug("[runner] action %s failed: %s", endpoint, exc)
            return None

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
    ) -> EpisodeResult:
        if episode_id is None:
            episode_id = f"cep_{secrets.token_hex(6)}"

        cfg = self._cfg
        result = EpisodeResult(episode_id=episode_id, config=cfg)

        # Wait for the container app to be ready before doing anything else.
        # This is the root cause of ECONNREFUSED: Docker marks the container
        # as "running" before uvicorn inside finishes startup.
        if not self.wait_for_health():
            result.termination_reason = f"container_unreachable: {cfg.base_url}"
            result.completed_at = datetime.now(timezone.utc)
            if jsonl_path is not None:
                result.write_jsonl(jsonl_path)
            return result

        available_actions = [
            *self._discover_actions(),
            {
                "endpoint": SUBMIT_ENDPOINT,
                "method": "CONTROL",
                "description": "Finish the episode and grade the current state",
            },
        ]

        # Reset the environment
        try:
            state = self._reset(seed=seed)
        except Exception as exc:
            logger.error("[%s] reset failed: %s", episode_id, exc)
            result.termination_reason = f"reset_failed: {exc}"
            result.completed_at = datetime.now(timezone.utc)
            return result

        dead_end_policy = DeadEndTerminationPolicy(cfg.dead_end_patience)
        max_steps_policy = MaxStepsTerminationPolicy(cfg.max_steps)
        # Write each step as it happens so a crash mid-episode still leaves a
        # durable, replayable partial trace (not just an all-or-nothing dump).
        writer = TrajectoryWriter(jsonl_path, result) if jsonl_path is not None else None

        try:
            self._run_steps(
                agent, cfg, result, dead_end_policy, available_actions, episode_id,
                writer, state, max_steps_policy=max_steps_policy
            )
        finally:
            if writer is not None:
                writer.close()

        return result

    def _run_steps(
        self, agent, cfg, result, dead_end_policy, available_actions, episode_id,
        writer, state, max_steps_policy=None,
    ):
        max_steps_policy = max_steps_policy or MaxStepsTerminationPolicy(cfg.max_steps)
        for step_idx in range(cfg.max_steps):
            state_hash_before = self._normalizer.hash(state)

            # Agent picks an action
            try:
                action = agent.act(state, cfg.objective, available_actions)
            except Exception as exc:
                logger.warning("[%s] step %d: agent.act failed: %s", episode_id, step_idx, exc)
                # Fall back to no-op
                action = {
                    "endpoint": available_actions[0]["endpoint"] if available_actions else "/forge/state",
                    "payload": {},
                }

            # Ensure the chosen endpoint is in the discovered set (safety)
            if not is_submit_action(action) and available_actions and not any(
                a["endpoint"] == action.get("endpoint") for a in available_actions
            ):
                logger.debug(
                    "[%s] step %d: agent chose unknown endpoint %r — falling back",
                    episode_id, step_idx, action.get("endpoint"),
                )
                action = {"endpoint": available_actions[0]["endpoint"], "payload": {}}

            if is_submit_action(action):
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
                step.reward = result.total_reward
                step.objective_score = result.final_objective_score
                if writer is not None:
                    writer.record(step)
                break

            # Execute the action
            self._execute_action(action)

            # Observe new state
            try:
                new_state = self._get_state()
            except Exception as exc:
                logger.warning("[%s] step %d: get_state failed: %s", episode_id, step_idx, exc)
                new_state = state

            state_hash_after = self._normalizer.hash(new_state)

            # Build derived-field diff for richer LLM judge context
            derived_diff: dict = {}
            if self._manifest is not None:
                for fname, fspec in self._manifest.fields.items():
                    if fspec.derived_from:
                        bv = state.get(fname)
                        av = new_state.get(fname)
                        if bv != av:
                            derived_diff[fname] = {"before": bv, "after": av}

            # Stopping checks use only deterministic state progress and budget.
            outcome = StepOutcome(
                step_index=step_idx,
                state_hash=state_hash_after,
            )
            decision = dead_end_policy.check(outcome) or max_steps_policy.check(outcome)
            termination_reason = decision.reason if decision else None
            truncated = bool(decision and decision.truncated)
            terminated = bool(decision and not decision.truncated)

            step = StepRecord(
                step_index=step_idx,
                state_before=state,
                action=action,
                state_after=new_state,
                reward=0.0,
                objective_score=0.0,
                state_hash_before=state_hash_before,
                state_hash_after=state_hash_after,
                terminated=terminated,
                truncated=truncated,
                termination_reason=termination_reason if (terminated or truncated) else None,
            )
            result.steps.append(step)
            logger.info(
                "[%s] step %02d/%d  hash=%s→%s%s",
                episode_id,
                step_idx + 1,
                cfg.max_steps,
                state_hash_before[:6],
                state_hash_after[:6],
                f"  → {termination_reason}" if termination_reason else "",
            )

            state = new_state

            if terminated or truncated:
                result.termination_reason = termination_reason or (
                    "truncated" if truncated else "unknown"
                )
                self._finalize_result(
                    result,
                    new_state,
                    action,
                    derived_diff=derived_diff or None,
                    state_changed=state_hash_before != state_hash_after,
                )
                step.reward = result.total_reward
                step.objective_score = result.final_objective_score
                if writer is not None:
                    writer.record(step)
                break
            if writer is not None:
                writer.record(step)

        result.completed_at = datetime.now(timezone.utc)

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
        state_changed = state_changed or any(
            step.state_hash_before != step.state_hash_after for step in result.steps
        )
        score = self._scorer.score(
            state,
            self._cfg.objective,
            derived_diff=derived_diff,
            action_taken=action,
        )
        result.llm_verdicts += 1
        verification = VerificationResult.from_checks(
            "objective_scorer",
            [CheckResult(
                name="objective_score",
                passed=score >= self._cfg.success_threshold,
                score=score,
            )],
        )
        selected_task = self._selected_task or Task(
            id="objective", objective=self._cfg.objective
        )
        task = selected_task.model_copy(update={
            "metadata": {
                **selected_task.metadata,
                "state_changed": state_changed,
            }
        })
        reward = self.environment.rubric.score(
            state,
            Trajectory(episode_id=result.episode_id, steps=[]),
            [verification],
            task,
        )
        result.final_objective_score = score
        result.apply_evaluation(EpisodeEvaluation(
            passed=verification.passed,
            reward=reward,
            verification_results=[verification],
            reason=result.termination_reason,
        ))

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
