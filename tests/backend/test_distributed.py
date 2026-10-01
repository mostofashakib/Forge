"""Tests for distributed training: contracts, durable queue, concurrent scheduler, and API routes (Feature Request 8)."""
import time
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from forge.contracts.inference import (
    InferenceBatchRequest,
    InferenceRequest,
    InferenceResponse,
)
from forge.runtime.distributed import (
    DistributedEnvironmentScheduler,
    DurableTaskQueue,
    MockInferenceProvider,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_inference_contracts_and_mock_provider():
    provider = MockInferenceProvider(default_response="Mock output")

    req = InferenceRequest(model="test-llm", prompt="Hello distributed agent")
    res = provider.generate(req)
    assert isinstance(res, InferenceResponse)
    assert res.model == "test-llm"
    assert "Processed prompt: Hello distributed" in res.content
    assert res.latency_ms >= 0

    batch = InferenceBatchRequest(
        requests=[
            InferenceRequest(model="test-llm", prompt="prompt 1"),
            InferenceRequest(model="test-llm", prompt="prompt 2"),
        ]
    )
    b_res = provider.generate_batch(batch)
    assert len(b_res.responses) == 2
    assert "prompt 1" in b_res.responses[0].content
    assert "prompt 2" in b_res.responses[1].content


def test_durable_queue_retries_and_wallclock_truncation():
    queue = DurableTaskQueue()

    # Enqueue with 2 retries and 0.2s wallclock budget
    item = queue.enqueue(
        task_id="durable_task_1",
        payload={"step": 1},
        max_retries=2,
        wallclock_budget_s=0.2,
    )
    assert item.task_id == "durable_task_1"
    assert item.status == "queued"

    # Dequeue to start running
    dequeued = queue.dequeue()
    assert dequeued is not None
    assert dequeued.task_id == "durable_task_1"
    assert dequeued.status == "running"

    # First failure -> retry 1
    queue.fail("durable_task_1", error="Transient RPC timeout")
    task = queue.get("durable_task_1")
    assert task.status == "queued"
    assert task.retries == 1

    # Dequeue again
    dequeued = queue.dequeue()
    assert dequeued.status == "running"

    # Exceed wallclock budget
    time.sleep(0.25)
    # Failure after wallclock timeout should mark as truncated
    queue.fail("durable_task_1", error="Another error")
    task = queue.get("durable_task_1")
    assert task.status == "truncated"
    assert "budget" in task.error


def test_distributed_environment_scheduler_concurrency():
    scheduler = DistributedEnvironmentScheduler(max_concurrent_environments=2)

    slots = scheduler.schedule_concurrent_builds(["env_a", "env_b"])
    assert len(slots) == 2
    assert {s.env_name for s in slots} == {"env_a", "env_b"}
    assert {s.status for s in slots} == {"idle"}

    # Acquire slot
    slot_a = scheduler.acquire("env_a")
    assert slot_a is not None
    assert slot_a.status == "allocated"

    # Cannot acquire again while allocated
    assert scheduler.acquire("env_a") is None

    # Status check
    status = scheduler.status()
    assert status["total_slots"] == 2
    assert status["allocated"] == 1
    assert status["idle"] == 1

    # Release slot
    assert scheduler.release(slot_a.slot_id) is True
    assert scheduler.acquire("env_a") is not None


def test_distributed_api_endpoints(client):
    # 1. Test single inference API
    inf_res = client.post(
        "/api/distributed/inference",
        json={"model": "deepseek-coder", "prompt": "def solve(): pass"},
    )
    assert inf_res.status_code == 200
    inf_data = inf_res.json()
    assert inf_data["model"] == "deepseek-coder"
    assert "solve" in inf_data["content"]

    # 2. Test batch inference API
    batch_res = client.post(
        "/api/distributed/inference/batch",
        json={
            "requests": [
                {"model": "deepseek-coder", "prompt": "task 1"},
                {"model": "deepseek-coder", "prompt": "task 2"},
            ]
        },
    )
    assert batch_res.status_code == 200
    assert len(batch_res.json()["responses"]) == 2

    # 3. Test queue enqueue, dequeue, complete
    enq_res = client.post(
        "/api/distributed/queue/enqueue",
        json={"task_id": "api_test_task", "payload": {"foo": "bar"}, "max_retries": 2},
    )
    assert enq_res.status_code == 200
    assert enq_res.json()["task_id"] == "api_test_task"

    deq_res = client.post("/api/distributed/queue/dequeue")
    assert deq_res.status_code == 200
    assert deq_res.json()["item"]["task_id"] == "api_test_task"

    comp_res = client.post(
        "/api/distributed/queue/api_test_task/complete",
        json={"result": {"loss": 0.05}},
    )
    assert comp_res.status_code == 200
    assert comp_res.json()["status"] == "completed"

    get_res = client.get("/api/distributed/queue/api_test_task")
    assert get_res.status_code == 200
    assert get_res.json()["result"] == {"loss": 0.05}

    # 4. Test scheduler endpoints
    sched_res = client.post(
        "/api/distributed/schedule",
        json={"env_names": ["api_env_1", "api_env_2"]},
    )
    assert sched_res.status_code == 200
    scheduled = sched_res.json()["scheduled"]
    assert len(scheduled) == 2

    status_res = client.get("/api/distributed/scheduler/status")
    assert status_res.status_code == 200
    assert status_res.json()["total_slots"] >= 2
