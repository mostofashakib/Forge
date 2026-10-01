"""Task factory API: create versioned batches of synthetic tasks for an environment."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from fastapi.concurrency import run_in_threadpool

from backend.app.api._pubsub_relay import relay_pubsub
from backend.app.database import get_db, get_session_factory
from backend.app.models import TaskBatch
from backend.app.services import task_registry
from backend.app.services.env_file import environ_with_saved
from backend.app.services.task_factory_targets import list_targets
from backend.app.worker.task_factory import channel_for, create_task_batch_task
from forge.settings import redis_url
from forge.taskfactory.model_settings import VALIDATOR_VARS, TaskFactoryConfigError, require_sdks, resolve_models
from forge.taskfactory.pipeline import MAX_COUNT, MAX_K, MIN_COUNT, MIN_K

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/task-factory")

_FINISHED = {"complete", "short", "failed"}


class CreateBatchRequest(BaseModel):
    env_name: str
    count: int = Field(ge=MIN_COUNT, le=MAX_COUNT)
    k: int = Field(default=3, ge=MIN_K, le=MAX_K)


@router.get("/environments")
def environments(db: Session = Depends(get_db)) -> list[dict]:
    return [target.__dict__ for target in list_targets(db)]


@router.post("/batches", status_code=202)
def create_batch(body: CreateBatchRequest, db: Session = Depends(get_db)) -> dict:
    target = next((t for t in list_targets(db) if t.name == body.env_name), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"no environment named {body.env_name!r}")
    if not target.ready:
        raise HTTPException(status_code=409, detail=target.reason)
    try:
        writer, validator = resolve_models(environ_with_saved(VALIDATOR_VARS))
        require_sdks(writer, validator)
    except TaskFactoryConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    batch_id = task_registry.create_batch(
        db, env_name=body.env_name, requested=body.count, pass_k=body.k, writer=writer, validator=validator,
    )
    create_task_batch_task.delay(batch_id=batch_id)
    logger.info("[taskfactory] queued %s for %s: %d tasks, pass^%d", batch_id, body.env_name, body.count, body.k)
    return {"batch_id": batch_id}


@router.get("/batches")
def list_batches(env_name: str | None = None, db: Session = Depends(get_db)) -> list[dict]:
    return task_registry.list_batches(db, env_name=env_name)


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str, db: Session = Depends(get_db)) -> dict:
    detail = task_registry.get_batch(db, batch_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return detail


@router.get("/batches/{batch_id}/export")
def export_batch(batch_id: str, db: Session = Depends(get_db)) -> JSONResponse:
    detail = task_registry.get_batch(db, batch_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="batch not found")
    if detail["version"] is None:
        raise HTTPException(status_code=409, detail="only a saved batch can be exported")
    date = (detail["created_at"] or "")[:10]
    filename = f"{detail['env_name']}-v{detail['version']}-{date}.json"
    return JSONResponse(detail, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.delete("/batches/{batch_id}", status_code=204)
def delete_batch(batch_id: str, db: Session = Depends(get_db)) -> Response:
    batch = db.get(TaskBatch, batch_id)
    if batch is None or batch.deleted_at is not None:
        raise HTTPException(status_code=404, detail="batch not found")
    if batch.status not in _FINISHED:
        raise HTTPException(status_code=409, detail="a batch can be deleted once it finishes")
    task_registry.delete_batch(db, batch_id)
    return Response(status_code=204)


def _finished_message(batch_id: str) -> dict | None:
    with get_session_factory()() as db:
        batch = db.get(TaskBatch, batch_id)
        if batch is None:
            return {"error": "batch not found"}
        if batch.status == "failed":
            return {"error": batch.error or "batch failed"}
        if batch.status in _FINISHED:
            return {"done": True, "version": batch.version, "status": batch.status, "delivered": batch.delivered}
        return None


@router.websocket("/ws/progress/{batch_id}")
async def batch_progress_ws(websocket: WebSocket, batch_id: str) -> None:
    import redis

    await websocket.accept()
    channel = channel_for(batch_id)
    try:
        connection = redis.asyncio.from_url(redis_url())
        pubsub = connection.pubsub()
        await pubsub.subscribe(channel)
    except Exception:
        logger.exception("[ws:taskfactory] could not connect to Redis")
        await websocket.close(code=1011)
        return
    try:
        # Checked after subscribing, so a batch that ends in between still
        # reaches this client through the channel.
        finished = await run_in_threadpool(_finished_message, batch_id)
        if finished is not None:
            await websocket.send_json(finished)
        else:
            await relay_pubsub(websocket, pubsub, is_final=lambda data: bool(data.get("done") or data.get("error")))
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[ws:taskfactory] unexpected error for %s", batch_id)
    finally:
        await pubsub.unsubscribe(channel)
        await connection.aclose()
        try:
            await websocket.close()
        except RuntimeError:
            pass
