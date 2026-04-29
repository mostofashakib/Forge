# backend/app/services/runner_service.py
from __future__ import annotations
import asyncio
import importlib
import json
import os
import secrets
import sys
from pathlib import Path
from forge.runtime.policy import RandomPolicy
from forge.runtime.telemetry import TelemetryClient
from backend.app.services import episode_service
from backend.app.database import get_session_factory

# episode_id → asyncio.Queue of step event dicts
episode_queues: dict[str, asyncio.Queue] = {}

# episode_id → asyncio.Task
episode_tasks: dict[str, asyncio.Task] = {}


def _load_env(env_name: str, telemetry: TelemetryClient):
    """Dynamically import the generated gym_wrapper and build the ForgeEnv."""
    envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
    parent = str(envs_root.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(f"generated_envs.{env_name}.gym_wrapper")
    build_fn = getattr(module, f"build_{env_name}_env")
    env = build_fn()
    env._telemetry = telemetry
    return env


async def start_episode(
    env_name: str,
    task_name: str,
    seed: int,
    agent_id: str,
) -> str:
    """Create Episode row, initialise queue, spawn background task, return episode_id."""
    episode_id = f"ep_{seed:08x}_{secrets.token_hex(4)}"

    # Compute jsonl_path here so it can be stored in the Episode row
    envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
    jsonl_path = envs_root / env_name / "episodes" / f"{episode_id}.jsonl"

    # Create Episode row synchronously so it's immediately queryable
    SessionFactory = get_session_factory()
    db = SessionFactory()
    try:
        episode_service.create_episode(
            episode_id=episode_id,
            env_name=env_name,
            task_name=task_name,
            seed=seed,
            agent_id=agent_id,
            db=db,
            jsonl_path=str(jsonl_path),
        )
    finally:
        db.close()

    episode_queues[episode_id] = asyncio.Queue()
    task = asyncio.create_task(
        _run_episode(episode_id, env_name, task_name, seed, agent_id, jsonl_path)
    )
    episode_tasks[episode_id] = task
    return episode_id


async def _run_episode(
    episode_id: str,
    env_name: str,
    task_name: str,
    seed: int,
    agent_id: str,
    jsonl_path: Path,
) -> None:
    queue = episode_queues.get(episode_id)
    SessionFactory = get_session_factory()
    db = SessionFactory()
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        telemetry = TelemetryClient(
            episode_id=episode_id,
            db_session=db,
            jsonl_path=jsonl_path,
        )
        env = _load_env(env_name, telemetry)
        policy = RandomPolicy(env.action_types)

        obs, info = env.reset(seed=seed)
        terminated = truncated = False

        while not (terminated or truncated):
            action = policy.act(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)

            event = {
                "type": "step",
                "step_index": env._step_count - 1,
                "action": action,
                "reward": reward,
                "diff": step_info.get("reward_breakdown", {}),
                "verifier_results": step_info.get("verifier_results", []),
                "events": step_info.get("events", []),
                "terminated": terminated,
            }
            if queue:
                await queue.put(event)
            await asyncio.sleep(0)

        complete_event = {
            "type": "complete",
            "total_reward": env._total_reward,
            "passed": terminated,
            "total_steps": env._step_count,
        }
        if queue:
            await queue.put(complete_event)
        await asyncio.sleep(0)  # yield so consumers can drain before cleanup
    except Exception as exc:
        if queue:
            await queue.put({"type": "error", "message": str(exc)})
    finally:
        db.close()
        episode_tasks.pop(episode_id, None)
        episode_queues.pop(episode_id, None)  # safe: consumer had a chance to drain above
