"""Read an environment's latest compiled input from the database."""
from __future__ import annotations

from typing import Callable

from forge.extraction.schemas import CompilerInput

# Loads the compiled input for an environment by name, or None if it has none.
CompilerInputLoader = Callable[[str], "CompilerInput | None"]


def db_compiler_input_loader(session_factory) -> CompilerInputLoader:
    """A loader that reads the latest compiled input for an env from the DB.

    Reuses the same ``CompileJob.compiler_input_json`` artifact the
    ``/api/envs/{name}/compiler-input`` endpoint serves.
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
