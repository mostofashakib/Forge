"""The Celery task catalog.

Each task lives in the module for its concern. Importing them here registers
them through `celery_app`'s include list. Every task keeps its registered name
`backend.app.worker.tasks.<name>`, so jobs already queued still route.
"""
from backend.app.worker.agent_run_tasks import run_container_episode_task, run_container_run_task
from backend.app.worker.benchmark_tasks import run_benchmark_task, run_evaluation_task, run_transfer_task
from backend.app.worker.rollout_tasks import run_episode_task, run_rollout_task
from backend.app.worker.sandbox_build_tasks import build_sandbox_task, cleanup_expired_sandboxes

__all__ = [
    "build_sandbox_task",
    "cleanup_expired_sandboxes",
    "run_benchmark_task",
    "run_container_episode_task",
    "run_container_run_task",
    "run_episode_task",
    "run_evaluation_task",
    "run_rollout_task",
    "run_transfer_task",
]
