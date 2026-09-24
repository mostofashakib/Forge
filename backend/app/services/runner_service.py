# backend/app/services/runner_service.py
from __future__ import annotations
import asyncio
import secrets
import threading
from collections.abc import Callable
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
    """Play the episode on a worker thread so env work never blocks the event loop."""
    queue = episode_queues.get(episode_id)
    loop = asyncio.get_running_loop()
    stop = threading.Event()

    def emit(event: dict) -> None:
        if queue is not None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

    try:
        await asyncio.to_thread(
            _play_episode, episode_id, env_name, seed, jsonl_path, prefix_actions, emit, stop
        )
    except asyncio.CancelledError:
        stop.set()
        raise
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
    finally:
        await asyncio.sleep(0)  # let queued events land before the queue is dropped
        episode_tasks.pop(episode_id, None)
        episode_queues.pop(episode_id, None)


def _play_episode(
    episode_id: str,
    env_name: str,
    seed: int,
    jsonl_path: Path,
    prefix_actions: list[dict],
    emit: Callable[[dict], None],
    stop: threading.Event,
) -> None:
    db = get_session_factory()()
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        telemetry = EpisodeDataCollector(
            episode_id=episode_id,
            db_session=db,
            jsonl_path=jsonl_path,
        )
        env = load_forge_env(env_name, telemetry)
        from forge.settings import experiment_seed

        policy = RandomPolicy(env.action_types, seed=experiment_seed(seed))

        obs, info = env.reset(seed=seed)
        terminated = truncated = False
        step_count = 0

        for action in prefix_actions:
            # Consume the deterministic policy choice that the original run
            # made at this step, then replay the recorded action exactly.
            policy.act(obs)
            obs, _reward, terminated, truncated, _step_info = env.step(action)
            step_count += 1
            if terminated or truncated:
                break

        while not (terminated or truncated):
            if stop.is_set():
                return
            action = policy.act(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            step_count += 1
            emit({
                "type": "step",
                "step_index": step_count - 1,
                "action": action,
                "reward": reward,
                "diff": step_info.get("reward_breakdown", {}),
                "verifier_results": step_info.get("verifier_results", []),
                "events": step_info.get("events", []),
                "terminated": terminated,
            })

        emit({
            "type": "complete",
            "total_reward": env.finalize_episode().total_reward,
            "passed": terminated,
            "total_steps": step_count,
        })
    finally:
        db.close()
