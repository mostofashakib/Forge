"""Task selection shared by in-process and container environments."""
from __future__ import annotations

from collections.abc import Mapping

from forge.contracts import Task, TaskSource, normalize_task


def select_task(
    source: TaskSource,
    *,
    seed: int,
    options: Mapping[str, object],
    fallback: Task | Mapping[str, object] | None = None,
) -> Task | None:
    """Select an explicit task, a requested id, or a seeded source entry.

    Source ordering is stable, so ``seed % task_count`` gives reproducible task
    selection while distributing rollouts across the available dataset.
    """
    explicit = options.get("task")
    if isinstance(explicit, (Task, Mapping)):
        return normalize_task(explicit)

    task_id = options.get("task_id")
    if task_id is not None:
        return source.get(str(task_id))

    tasks = source.tasks()
    if tasks:
        return tasks[seed % len(tasks)]
    if fallback is not None:
        return normalize_task(fallback)
    return None


def task_payload(task: Task | None) -> dict | None:
    """Render a typed task for legacy verifiers and reward functions."""
    if task is None:
        return None
    return {
        **task.metadata,
        "id": task.id,
        "name": task.id,
        "objective": task.objective,
        "seed": task.seed,
        "success_conditions": task.success_conditions,
        "failure_conditions": task.failure_conditions,
    }
