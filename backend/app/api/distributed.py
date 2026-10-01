"""Distributed training routes for inference API contracts, durable queues, and concurrent environment scheduling (Feature Request 8)."""
from __future__ import annotations

from typing import Any, Literal
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from forge.contracts.gpu import (
    APIGatewaySpec,
    GPUInferenceContract,
    InferenceMode,
    compute_status,
)
from forge.contracts.inference import (
    InferenceBatchRequest,
    InferenceBatchResponse,
    InferenceRequest,
    InferenceResponse,
)
from forge.runtime.distributed import (
    DistributedEnvironmentScheduler,
    DurableTaskQueue,
)
from forge.runtime.gpu_inference import HybridGPUInferenceEngine

router = APIRouter(prefix="/api/distributed", tags=["distributed"])

# Shared singleton instances for distributed orchestration
_queue = DurableTaskQueue()
_scheduler = DistributedEnvironmentScheduler(max_concurrent_environments=4)


def _get_inference_engine(
    mode: str = "auto", gateway_url: str | None = None
) -> HybridGPUInferenceEngine:
    parsed_mode = InferenceMode(mode) if mode in ("local_gpu", "api_gateway", "auto") else InferenceMode.AUTO
    gw_spec = APIGatewaySpec(endpoint_url=gateway_url) if gateway_url else APIGatewaySpec()
    contract = GPUInferenceContract(mode=parsed_mode, api_gateway_spec=gw_spec)
    return HybridGPUInferenceEngine(contract)


class EnqueueRequest(BaseModel):
    task_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    max_retries: int = 3
    wallclock_budget_s: float = 300.0


class CompleteTaskRequest(BaseModel):
    result: Any = None


class FailTaskRequest(BaseModel):
    error: str


class ScheduleBuildsRequest(BaseModel):
    env_names: list[str]


class AcquireSlotRequest(BaseModel):
    env_name: str


class ReleaseSlotRequest(BaseModel):
    slot_id: str


# --- Inference API Endpoints ---


@router.get("/hardware")
def get_hardware_status() -> dict[str, Any]:
    """Inspect local GPU accelerators (CUDA/MPS) and cloud gateway connectivity."""
    return compute_status()


@router.post("/inference", response_model=InferenceResponse)
def run_inference(
    request: InferenceRequest,
    mode: Literal["auto", "local_gpu", "api_gateway"] = Query(default="auto"),
    gateway_url: str | None = Query(default=None),
) -> InferenceResponse:
    """Execute a single inference request across local GPU or cloud API gateway."""
    try:
        engine = _get_inference_engine(mode=mode, gateway_url=gateway_url)
        return engine.generate(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/inference/batch", response_model=InferenceBatchResponse)
def run_inference_batch(
    batch: InferenceBatchRequest,
    mode: Literal["auto", "local_gpu", "api_gateway"] = Query(default="auto"),
    gateway_url: str | None = Query(default=None),
) -> InferenceBatchResponse:
    """Execute a batch of inference requests across local GPU or cloud API gateway."""
    try:
        engine = _get_inference_engine(mode=mode, gateway_url=gateway_url)
        return engine.generate_batch(batch)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --- Durable Task Queue Endpoints ---


@router.post("/queue/enqueue")
def enqueue_task(req: EnqueueRequest) -> dict[str, Any]:
    """Enqueue a distributed training task with retry and wallclock timeout budget."""
    item = _queue.enqueue(
        task_id=req.task_id,
        payload=req.payload,
        max_retries=req.max_retries,
        wallclock_budget_s=req.wallclock_budget_s,
    )
    return {
        "task_id": item.task_id,
        "status": item.status,
        "max_retries": item.max_retries,
        "wallclock_budget_s": item.wallclock_budget_s,
        "created_at": item.created_at,
    }


@router.post("/queue/dequeue")
def dequeue_task() -> dict[str, Any]:
    """Dequeue the next available task from the durable queue."""
    item = _queue.dequeue()
    if item is None:
        return {"item": None}
    return {
        "item": {
            "task_id": item.task_id,
            "status": item.status,
            "payload": item.payload,
            "retries": item.retries,
            "max_retries": item.max_retries,
            "wallclock_budget_s": item.wallclock_budget_s,
            "started_at": item.started_at,
        }
    }


@router.post("/queue/{task_id}/complete")
def complete_task(task_id: str, req: CompleteTaskRequest) -> dict[str, Any]:
    """Mark a queued distributed task as successfully completed."""
    success = _queue.complete(task_id, result=req.result)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found in queue")
    return {"task_id": task_id, "status": "completed"}


@router.post("/queue/{task_id}/fail")
def fail_task(task_id: str, req: FailTaskRequest) -> dict[str, Any]:
    """Mark a queued distributed task as failed, triggering automatic retry or truncation."""
    success = _queue.fail(task_id, error=req.error)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found in queue")
    item = _queue.get(task_id)
    return {
        "task_id": task_id,
        "status": item.status if item else "unknown",
        "retries": item.retries if item else 0,
        "error": item.error if item else req.error,
    }


@router.get("/queue/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    """Get the status, retry details, and result of a queued distributed task."""
    item = _queue.get(task_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found in queue")
    return {
        "task_id": item.task_id,
        "status": item.status,
        "payload": item.payload,
        "retries": item.retries,
        "max_retries": item.max_retries,
        "wallclock_budget_s": item.wallclock_budget_s,
        "created_at": item.created_at,
        "started_at": item.started_at,
        "completed_at": item.completed_at,
        "error": item.error,
        "result": item.result,
    }


@router.get("/queue")
def list_tasks() -> dict[str, Any]:
    """List all queued, running, completed, or failed tasks."""
    items = _queue.list_all()
    return {
        "tasks": [
            {
                "task_id": item.task_id,
                "status": item.status,
                "retries": item.retries,
                "max_retries": item.max_retries,
                "wallclock_budget_s": item.wallclock_budget_s,
                "error": item.error,
            }
            for item in items
        ],
        "size": _queue.size(),
    }


# --- Concurrent Environment Scheduler Endpoints ---


@router.post("/schedule")
def schedule_environments(req: ScheduleBuildsRequest) -> dict[str, Any]:
    """Schedule concurrent environment builds up to the concurrency limit."""
    slots = _scheduler.schedule_concurrent_builds(req.env_names)
    return {
        "scheduled": [
            {
                "slot_id": s.slot_id,
                "env_name": s.env_name,
                "status": s.status,
                "container_id": s.container_id,
                "port": s.port,
            }
            for s in slots
        ]
    }


@router.get("/scheduler/status")
def get_scheduler_status() -> dict[str, Any]:
    """Retrieve current concurrency state of the distributed environment scheduler."""
    return _scheduler.status()


@router.post("/scheduler/acquire")
def acquire_slot(req: AcquireSlotRequest) -> dict[str, Any]:
    """Acquire an idle environment slot for execution."""
    slot = _scheduler.acquire(req.env_name)
    if not slot:
        raise HTTPException(status_code=404, detail=f"No idle slot available for {req.env_name}")
    return {
        "slot_id": slot.slot_id,
        "env_name": slot.env_name,
        "status": slot.status,
        "port": slot.port,
    }


@router.post("/scheduler/release")
def release_slot(req: ReleaseSlotRequest) -> dict[str, Any]:
    """Release an allocated environment slot back to idle."""
    success = _scheduler.release(req.slot_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Slot {req.slot_id} not found")
    return {"slot_id": req.slot_id, "status": "idle"}
