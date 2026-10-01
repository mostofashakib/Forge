"""The Celery job that creates one batch of synthetic tasks.

Streams progress to the Redis channel forge:taskfactory:{batch_id}:
  {"stage": ..., "log": ...}      a step started or finished
  {"done": true, "version": N}    the batch is saved
  {"error": "..."}                the batch failed
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager

from sqlalchemy.orm import Session

from backend.app.models import TaskBatch
from backend.app.services import task_registry
from backend.app.services.env_file import environ_with_saved
from backend.app.services.task_factory_targets import open_target as default_open_target
from backend.app.worker.celery_app import celery
from forge.extraction.llm_client import LLMClient, get_client
from forge.settings import redis_url
from forge.taskfactory.model_settings import VALIDATOR_VARS, ModelSpec, require_sdks, resolve_models
from forge.taskfactory.pipeline import TaskFactoryPipeline
from forge.taskfactory.review import TaskReviewer
from forge.taskfactory.taxonomy import TaxonomyBuilder
from forge.taskfactory.writer import WRITER_MAX_TOKENS, TaskWriter

logger = logging.getLogger(__name__)

VALIDATOR_MAX_TOKENS = 8192


def channel_for(batch_id: str) -> str:
    return f"forge:taskfactory:{batch_id}"


def _default_client(spec: ModelSpec, max_tokens: int) -> LLMClient:
    return get_client(max_tokens=max_tokens, provider=spec.provider, model=spec.model)


def execute_batch(
    db: Session,
    batch_id: str,
    *,
    publish: Callable[[dict], None],
    exclusive: Callable[[str], AbstractContextManager[None]],
    environ: dict[str, str] | None = None,
    make_client: Callable[[ModelSpec, int], LLMClient] = _default_client,
    open_target=default_open_target,
) -> None:
    """Run the pipeline for a queued batch and save it, or mark it failed."""
    batch = db.get(TaskBatch, batch_id)
    if batch is None:
        logger.warning("[taskfactory] batch %s vanished before it ran", batch_id)
        return
    env_name, requested, k = batch.env_name, batch.requested, batch.pass_k
    data_type = getattr(batch, "data_type", "rl_tasks")
    try:
        writer, validator = resolve_models(environ if environ is not None else environ_with_saved(VALIDATOR_VARS))
        require_sdks(writer, validator)
        task_registry.mark_running(db, batch_id, writer=writer, validator=validator)
        publish({"stage": "models", "log": f"Writer {writer.model}, validator {validator.model}"})
        writer_client = make_client(writer, WRITER_MAX_TOKENS)
        validator_client = make_client(validator, VALIDATOR_MAX_TOKENS)
        with open_target(db, env_name, batch_id) as target:
            with exclusive(env_name):
                profile = target.profile()
            pipeline = TaskFactoryPipeline(
                profile=profile,
                taxonomy_builder=TaxonomyBuilder(writer_client),
                writer=TaskWriter(writer_client),
                reviewer=TaskReviewer(validator_client),
                runner=target.runner,
                exclusive=lambda: exclusive(env_name),
                progress=publish,
            )
            result = pipeline.run(requested, k, data_type=data_type)
        version = task_registry.save_result(db, batch_id, result)
    except Exception as exc:  # noqa: BLE001 — every failure is recorded on the batch
        logger.exception("[taskfactory] batch %s failed", batch_id)
        db.rollback()
        task_registry.mark_failed(db, batch_id, str(exc))
        publish({"error": str(exc)})
        return
    publish({
        "done": True, "version": version, "status": result.status,
        "delivered": len(result.tasks), "requested": requested,
        "log": f"Saved version {version}: {len(result.tasks)} of {requested} tasks",
    })


@celery.task(name="backend.app.worker.task_factory.create_task_batch_task", ignore_result=True)
def create_task_batch_task(batch_id: str) -> None:
    import redis

    from backend.app.database import get_session_factory
    from backend.app.worker.env_lock import exclusive_environment

    client = redis.from_url(redis_url())

    def publish(message: dict) -> None:
        try:
            client.publish(channel_for(batch_id), json.dumps(message))
        except Exception:  # noqa: BLE001 — progress is best effort, the batch row is the record
            logger.debug("[taskfactory] progress publish failed", exc_info=True)

    with get_session_factory()() as db:
        execute_batch(
            db, batch_id, publish=publish,
            exclusive=lambda env_name: exclusive_environment(client, env_name),
        )
