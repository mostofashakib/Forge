import copy
import time
import pytest

from forge.contracts import InitialStateProvider
from forge.runtime.env_builder import EnvBuilder
from forge.runtime.transition import TransitionResult
from forge.runtime.verification import CheckResult, VerificationResult
from forge.runtime.parallel_rollout import RolloutSpec
from forge.runtime.distributed import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_QUEUE_TTL_S,
    DeterministicSeedPartitioner,
    DistributedParallelRolloutRunner,
    DistributedRolloutBatch,
    DurableQueueItem,
    DurableTaskQueue,
)


class SeededCounterFactory(InitialStateProvider):
    def reset(self, ctx, *, seed: int | None, options: dict) -> dict:
        return {"counter": {"c_0": {"id": "c_0", "value": ctx.rng.randint(0, 1000)}}}


def increment(state, action, ctx):
    new_state = copy.deepcopy(state)
    new_state["counter"]["c_0"]["value"] += 50
    return TransitionResult(state=new_state, events=[{"type": "incremented"}])


def reach_900(state, trajectory, task):
    passed = state["counter"]["c_0"]["value"] >= 900
    return VerificationResult.from_checks(
        "reach_900", [CheckResult(name="reached", passed=passed, score=1.0 if passed else 0.0)]
    )


def make_env_factory(created_envs=None):
    def factory():
        env = (
            EnvBuilder("dist_rollout_env", domain="test", max_steps=5)
            .with_initial_state(SeededCounterFactory())
            .with_transition("increment", increment)
            .with_verifier("reach_900", reach_900)
            .build(verify=False)
        )
        if created_envs is not None:
            created_envs.append(env)
        return env

    return factory


TASK = {"name": "reach_900", "verifier_id": "reach_900", "inputs": {}}


def increment_policy(obs, action_types):
    return {"type": "increment"}


# ---------------------------------------------------------------------------
# 1. Bursty Queue, Retries, DLQ, TTL (10 mins)
# ---------------------------------------------------------------------------

def test_queue_default_ttl_is_10_minutes():
    assert DEFAULT_QUEUE_TTL_S == 600.0
    item = DurableQueueItem(task_id="t1", payload={})
    assert item.ttl_s == 600.0


def test_queue_burst_enqueue_and_backpressure():
    queue = DurableTaskQueue(max_queue_depth=50)

    # Enqueue a burst of 30 tasks
    burst = [{"task_id": f"burst_{i}", "val": i} for i in range(30)]
    items = queue.enqueue_batch(burst)
    assert len(items) == 30
    assert queue.size() == 30

    # Overflowing queue beyond backpressure capacity raises RuntimeError
    with pytest.raises(RuntimeError, match="backpressure"):
        queue.enqueue_batch([{"task_id": f"burst_overflow_{i}"} for i in range(25)])


def test_queue_ttl_expiration_sweeps_stale_tasks():
    # Use a small TTL (0.1s) to test 10-minute expiration logic
    queue = DurableTaskQueue(default_ttl_s=0.1)
    queue.enqueue(task_id="stale_task", payload={"data": 1}, ttl_s=0.1)

    time.sleep(0.15)
    # Dequeue sweeps expired tasks
    dequeued = queue.dequeue()
    assert dequeued is None

    stale = queue.get("stale_task")
    assert stale.status == "truncated"
    assert "TTL" in stale.error


def test_dlq_after_3_failures_and_redrive():
    queue = DurableTaskQueue(default_max_retries=3)
    queue.enqueue(task_id="failing_task", payload={"run": 1}, max_retries=3)

    # 1st failure -> retried
    queue.dequeue()
    queue.fail("failing_task", error="Failure 1")
    assert queue.get("failing_task").status == "queued"
    assert queue.get("failing_task").retries == 1

    # 2nd failure -> retried
    queue.dequeue()
    queue.fail("failing_task", error="Failure 2")
    assert queue.get("failing_task").status == "queued"
    assert queue.get("failing_task").retries == 2

    # 3rd failure -> retried
    queue.dequeue()
    queue.fail("failing_task", error="Failure 3")
    assert queue.get("failing_task").status == "queued"
    assert queue.get("failing_task").retries == 3

    # 4th failure (after 3 retries exhausted) -> moved to DLQ!
    queue.dequeue()
    queue.fail("failing_task", error="Failure 4 - permanent")
    item = queue.get("failing_task")
    assert item.status == "dlq"
    assert len(item.failure_history) == 4
    assert len(queue.get_dlq()) == 1

    # Redrive DLQ task
    redriven = queue.redrive_dlq("failing_task")
    assert len(redriven) == 1
    assert queue.get("failing_task").status == "queued"
    assert queue.get("failing_task").retries == 0
    assert len(queue.get_dlq()) == 0


def test_queue_stats():
    queue = DurableTaskQueue()
    queue.enqueue("q1")
    queue.enqueue("q2")
    t = queue.dequeue()
    queue.complete(t.task_id)

    stats = queue.stats()
    assert stats["completed"] == 1
    assert stats["queued"] == 1
    assert stats["total"] == 2


# ---------------------------------------------------------------------------
# 2. Deterministic Seed Assignment
# ---------------------------------------------------------------------------

def test_deterministic_seed_partitioning():
    p1 = DeterministicSeedPartitioner.partition_seeds(base_seed=100, num_rollouts=10, num_workers=3)
    p2 = DeterministicSeedPartitioner.partition_seeds(base_seed=100, num_rollouts=10, num_workers=3)

    assert p1 == p2
    assert p1[0] == [100, 103, 106, 109]
    assert p1[1] == [101, 104, 107]
    assert p1[2] == [102, 105, 108]

    # Verify all seeds are present without duplicates
    all_seeds = sorted(p1[0] + p1[1] + p1[2])
    assert all_seeds == list(range(100, 110))


# ---------------------------------------------------------------------------
# 3. Distributed Parallel Rollout Runner Execution & Aggregation
# ---------------------------------------------------------------------------

def test_distributed_runner_multi_worker_execution():
    created = []
    runner = DistributedParallelRolloutRunner(
        make_env_factory(created),
        num_workers=4,
    )

    specs = [
        RolloutSpec(seed=s, task=TASK, policy=increment_policy)
        for s in range(8)
    ]
    batch = runner.run(specs)

    assert isinstance(batch, DistributedRolloutBatch)
    assert len(batch.records) == 8
    assert batch.total_scheduled == 8
    assert len(created) == 8
    assert len({id(e) for e in created}) == 8

    # Check run-level aggregation
    summary = batch.to_summary_dict()
    assert summary["completed"] == 8
    assert summary["dlq_count"] == 0
    assert "outcome_counts" in summary
    assert "mean_reward" in summary
    assert "pass_rate" in summary
    assert len(summary["worker_distribution"]) > 1


def test_distributed_runner_reproducible_per_seed():
    specs = [RolloutSpec(seed=s, task=TASK, policy=increment_policy) for s in range(6)]

    runner1 = DistributedParallelRolloutRunner(make_env_factory(), num_workers=4)
    runner2 = DistributedParallelRolloutRunner(make_env_factory(), num_workers=1)

    batch1 = runner1.run(specs)
    batch2 = runner2.run(specs)

    by_seed_1 = {r.seed: (r.total_reward, r.steps, r.outcome) for r in batch1.records}
    by_seed_2 = {r.seed: (r.total_reward, r.steps, r.outcome) for r in batch2.records}
    assert by_seed_1 == by_seed_2


def test_distributed_runner_dlq_tracking_on_persistent_failures():
    failing_queue = DurableTaskQueue(default_max_retries=1)

    # Env factory that crashes
    def crashing_env_factory():
        raise RuntimeError("Worker host out of memory")

    runner = DistributedParallelRolloutRunner(
        crashing_env_factory,
        num_workers=2,
        queue=failing_queue,
        max_retries=1,
    )

    specs = [RolloutSpec(seed=1, task=TASK, policy=increment_policy)]
    batch = runner.run(specs, timeout_s=2.0)

    # Task exceeded max_retries and ended in DLQ
    assert len(batch.dlq_items) == 1
    assert batch.dlq_items[0].status == "dlq"
    assert "out of memory" in batch.dlq_items[0].error
