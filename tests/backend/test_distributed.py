"""Tests for the durable task queue."""
import time
from forge.runtime.distributed import DurableTaskQueue


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
