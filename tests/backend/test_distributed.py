"""Tests for the inference contracts and the durable task queue."""
import time
import pytest
from forge.contracts.inference import (
    InferenceBatchRequest,
    InferenceRequest,
    InferenceResponse,
)
from forge.runtime.distributed import (
    DurableTaskQueue,
    MockInferenceProvider,
)


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
