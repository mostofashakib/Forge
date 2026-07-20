"""Resolve a generated environment's own compiled tasks as benchmark tasks.

The benchmark's execution and quality path is keyed by environment name
(`envs_root/{name}/port`, `state_schema.json`, per-env `compute_env_quality`).
This module maps a generated environment's compiled :class:`TaskTemplate`s
(persisted in ``CompileJob.compiler_input_json``) onto the benchmark
:class:`~forge.benchmark.task_suite.Task` shape so every benchmarked environment
runs against its own tasks.

Grading is unaffected: the container episode runner scores generated
environments with their own ``reward_fn``/verifier, so ``Task.success_fn`` is
never invoked on this path and is present only to satisfy the dataclass.
"""

from __future__ import annotations

from typing import Callable

from forge.benchmark.task_suite import Task
from forge.extraction.schemas import CompilerInput, TaskTemplate

# Loads the compiled input for an environment by name, or None if it has none.
CompilerInputLoader = Callable[[str], "CompilerInput | None"]


def _derive_difficulty(template: TaskTemplate) -> int:
    """Approximate a 1–5 difficulty from how much the task asserts.

    Compiled tasks carry no explicit difficulty, but the number of success and
    failure conditions is a reasonable proxy — more conditions to satisfy is a
    harder task — and keeps the depth slider meaningful for generated envs.
    """
    conditions = len(template.success_conditions) + len(template.failure_conditions)
    return max(1, min(5, conditions))


def _placeholder_success(_state: dict) -> bool:
    # Generated environments are graded by their own reward_fn/verifier inside
    # the container episode runner; the benchmark never calls Task.success_fn on
    # this path. This exists only to satisfy the Task contract.
    return False


def task_from_template(template: TaskTemplate, env_name: str) -> Task:
    """Map one compiled :class:`TaskTemplate` to a benchmark :class:`Task`."""
    return Task(
        name=template.name,
        domain=env_name,
        objective=template.description,
        success_fn=_placeholder_success,
        difficulty=_derive_difficulty(template),
    )


class CompiledTaskProvider:
    """Serves a generated environment's compiled tasks, filtered by depth."""

    def __init__(self, loader: CompilerInputLoader) -> None:
        self._loader = loader

    def tasks_for(self, domain: str, depth: int) -> list[Task]:
        compiler_input = self._loader(domain)
        if compiler_input is None:
            return []
        tasks = [task_from_template(t, domain) for t in compiler_input.tasks]
        return [t for t in tasks if t.difficulty <= depth]


def db_compiler_input_loader(session_factory) -> CompilerInputLoader:
    """A loader that reads the latest compiled input for an env from the DB.

    Reuses the same ``CompileJob.compiler_input_json`` artifact the
    ``/api/envs/{name}/compiler-input`` endpoint serves, so already-generated
    environments are benchmarkable without regeneration.
    """

    def load(env_name: str) -> CompilerInput | None:
        from backend.app.models import CompileJob

        with session_factory() as db:
            job = (
                db.query(CompileJob)
                .filter_by(project_name=env_name)
                .order_by(CompileJob.created_at.desc())
                .first()
            )
            if job is None or not job.compiler_input_json:
                return None
            return CompilerInput.model_validate_json(job.compiler_input_json)

    return load
