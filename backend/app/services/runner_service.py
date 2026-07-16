# backend/app/services/runner_service.py
from __future__ import annotations
import asyncio
import secrets
from pathlib import Path
from forge.settings import generated_envs_root
from forge.runtime.policy import RandomPolicy
from backend.app.services.episode_collector import EpisodeDataCollector
from backend.app.services import episode_service
from backend.app.database import get_session_factory
from backend.app.utils.env_loader import load_forge_env

# episode_id → asyncio.Queue of step event dicts
episode_queues: dict[str, asyncio.Queue] = {}

# episode_id → asyncio.Task
episode_tasks: dict[str, asyncio.Task] = {}


async def start_episode(
    env_name: str,
    task_name: str,
    seed: int,
    agent_id: str,
) -> str:
    return await _start_episode(env_name, task_name, seed, agent_id, [])


async def start_branched_episode(
    env_name: str,
    task_name: str,
    seed: int,
    agent_id: str,
    prefix_actions: list[dict],
) -> str:
    """Start a new episode from a deterministic replay prefix."""
    return await _start_episode(env_name, task_name, seed, agent_id, prefix_actions)


async def _start_episode(
    env_name: str,
    task_name: str,
    seed: int,
    agent_id: str,
    prefix_actions: list[dict],
) -> str:
    """Create Episode row, initialise queue, spawn background task, return episode_id."""
    episode_id = f"ep_{seed:08x}_{secrets.token_hex(4)}"

    # Compute jsonl_path here so it can be stored in the Episode row
    envs_root = generated_envs_root()
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
        _run_episode(
            episode_id,
            env_name,
            task_name,
            seed,
            agent_id,
            jsonl_path,
            prefix_actions,
        )
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
    prefix_actions: list[dict],
) -> None:
    queue = episode_queues.get(episode_id)
    SessionFactory = get_session_factory()
    db = SessionFactory()
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        telemetry = EpisodeDataCollector(
            episode_id=episode_id,
            db_session=db,
            jsonl_path=jsonl_path,
        )
        env = load_forge_env(env_name, telemetry)
        policy = RandomPolicy(env.action_types, seed=seed)

        obs, info = env.reset(seed=seed)
        terminated = truncated = False

        for action in prefix_actions:
            # Consume the deterministic policy choice that the original run
            # made at this step, then replay the recorded action exactly.
            policy.act(obs)
            obs, _reward, terminated, truncated, _step_info = env.step(action)
            if terminated or truncated:
                break

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
