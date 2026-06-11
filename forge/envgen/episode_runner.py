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

from forge.envgen.agents.container_agent import ContainerAgentBase
from forge.envgen.episode_base import (
    BaseEpisodeConfig,
    BaseEpisodeResult,
    TerminationMonitor,
)
from forge.envgen.objective import ObjectiveScorer
from forge.schema.state_schema import StateSchemaManifest

logger = logging.getLogger(__name__)


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

    def _step_dicts(self) -> list[dict]:
        return [
            {
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
            for step in self.steps
        ]

    def summary(self) -> dict:
        return {**super().summary(), "episode_id": self.episode_id}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

class ContainerEpisodeRunner:
    """Runs one or more agent episodes against a containerized FastAPI environment.

    The runner communicates with the app entirely over HTTP:
      - GET  /forge/state  → current state dict
      - POST /forge/reset  → reset to initial state
      - GET  /openapi.json → discover available action endpoints
      - POST /<action>     → execute an action

    Stopping conditions (evaluated each step, in priority order):
      1. objective_score >= success_threshold → "success"
      2. State unchanged for dead_end_patience steps → "dead_end"
      3. objective_score < divergence_threshold for consecutive_below_threshold
         steps → "diverged"
      4. step_index == max_steps - 1 → "max_steps" (truncated)
    """

    def __init__(
        self,
        config: EpisodeConfig,
        scorer: ObjectiveScorer | None = None,
        manifest: StateSchemaManifest | None = None,
    ) -> None:
        self._cfg = config
        self._scorer = scorer or ObjectiveScorer()
        self._normalizer = HashNormalizer(manifest=manifest)
        self._manifest = manifest
        self._http = httpx.Client(
            base_url=config.base_url,
            timeout=config.http_timeout,
        )
        self._actions: list[dict] | None = None  # cached after first discovery

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
        resp = self._http.get("/forge/state")
        resp.raise_for_status()
        return resp.json()

    def _reset(self) -> dict:
        self._http.post("/forge/reset")
        return self._get_state()

    def _discover_actions(self) -> list[dict]:
        """Build an action manifest from /openapi.json. Cached after first call."""
        if self._actions is not None:
            return self._actions
        try:
            resp = self._http.get("/openapi.json", timeout=10.0)
            schema = resp.json()
            actions: list[dict] = []
            components = schema.get("components", {}).get("schemas", {})
            for path, path_item in schema.get("paths", {}).items():
                if path.startswith("/forge/") or path == "/ui":
                    continue
                post_op = path_item.get("post")
                if post_op is None:
                    continue
                action: dict = {
                    "endpoint": path,
                    "description": post_op.get("summary") or post_op.get("operationId") or path,
                }
                # Resolve request body schema (inline or $ref)
                body = post_op.get("requestBody", {})
                content = body.get("content", {}).get("application/json", {})
                req_schema = content.get("schema", {})
                if "$ref" in req_schema:
                    ref_name = req_schema["$ref"].split("/")[-1]
                    req_schema = components.get(ref_name, {})
                action["request_schema"] = req_schema
                actions.append(action)
            self._actions = actions
            logger.info("[runner] discovered %d action endpoints", len(actions))
            return actions
        except Exception as exc:
            logger.warning("[runner] could not discover actions: %s", exc)
            self._actions = []
            return []

    def _execute_action(self, action: dict) -> dict | None:
        endpoint = action.get("endpoint", "")
        payload = action.get("payload", {})
        try:
            resp = self._http.post(endpoint, json=payload)
            if resp.is_success:
                try:
                    return resp.json()
                except Exception:
                    return {}
            logger.debug("[runner] action %s → HTTP %d", endpoint, resp.status_code)
            return None
        except Exception as exc:
            logger.debug("[runner] action %s failed: %s", endpoint, exc)
            return None

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------

    def run_episode(
        self,
        agent: ContainerAgentBase,
        episode_id: str | None = None,
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

        available_actions = self._discover_actions()

        # Reset the environment
        try:
            state = self._reset()
        except Exception as exc:
            logger.error("[%s] reset failed: %s", episode_id, exc)
            result.termination_reason = f"reset_failed: {exc}"
            result.completed_at = datetime.now(timezone.utc)
            return result

        monitor = TerminationMonitor(cfg)

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
            if available_actions and not any(
                a["endpoint"] == action.get("endpoint") for a in available_actions
            ):
                logger.debug(
                    "[%s] step %d: agent chose unknown endpoint %r — falling back",
                    episode_id, step_idx, action.get("endpoint"),
                )
                action = {"endpoint": available_actions[0]["endpoint"], "payload": {}}

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

            obj_score = self._scorer.score(
                new_state, cfg.objective,
                derived_diff=derived_diff or None,
                action_taken=action or None,
            )

            # StateDiffFloor: reward at least diff_floor if stable state changed
            if self._manifest is not None and self._manifest.state_changed(state, new_state):
                reward = max(obj_score, cfg.diff_floor)
            else:
                reward = obj_score

            # Evaluate stopping conditions (state hash is the progress marker
            # so fluctuating scores over a frozen state still count as dead-end)
            truncated = step_idx >= cfg.max_steps - 1
            termination_reason = monitor.observe(obj_score, marker=state_hash_after)
            terminated = termination_reason is not None
            if not terminated and truncated:
                termination_reason = "max_steps"

            step = StepRecord(
                step_index=step_idx,
                state_before=state,
                action=action,
                state_after=new_state,
                reward=reward,
                objective_score=obj_score,
                state_hash_before=state_hash_before,
                state_hash_after=state_hash_after,
                terminated=terminated,
                truncated=truncated,
                termination_reason=termination_reason if (terminated or truncated) else None,
            )
            result.steps.append(step)
            result.total_reward += reward
            result.final_objective_score = obj_score

            logger.info(
                "[%s] step %02d/%d  score=%.2f  reward=%.2f  hash=%s→%s%s",
                episode_id,
                step_idx + 1,
                cfg.max_steps,
                obj_score,
                reward,
                state_hash_before[:6],
                state_hash_after[:6],
                f"  → {termination_reason}" if termination_reason else "",
            )

            state = new_state

            if terminated or truncated:
                result.termination_reason = termination_reason or (
                    "truncated" if truncated else "unknown"
                )
                break

        result.completed_at = datetime.now(timezone.utc)

        # Normalize total_reward to 0–1 (average objective score across steps)
        if result.steps:
            result.total_reward = result.total_reward / len(result.steps)

        if jsonl_path is not None:
            result.write_jsonl(jsonl_path)

        return result

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
            result = self.run_episode(agent, episode_id=episode_id, jsonl_path=jsonl_path)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ContainerEpisodeRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
