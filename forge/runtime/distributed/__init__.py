"""Distributed parallel runs package: durable queues, scheduler, inference, and rollout runners."""
from __future__ import annotations

from forge.runtime.distributed.inference import (
    InferenceProvider,
    InferenceRequest,
    InferenceResponse,
    MockInferenceProvider,
)
from forge.runtime.distributed.partitioner import DeterministicSeedPartitioner
from forge.runtime.distributed.queue import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_QUEUE_TTL_S,
    DurableQueueItem,
    DurableTaskQueue,
)
from forge.runtime.distributed.runner import (
    DistributedParallelRolloutRunner,
    DistributedRolloutBatch,
)
from forge.runtime.distributed.scheduler import (
    DistributedEnvironmentScheduler,
    EnvironmentSlot,
)


def create_rollout_runner(*args, **kwargs):
    from forge.runtime.parallel_rollout import create_rollout_runner as _create

    return _create(*args, **kwargs)


def run_parallel_rollouts(*args, **kwargs):
    from forge.runtime.parallel_rollout import run_parallel_rollouts as _run

    return _run(*args, **kwargs)


__all__ = [
    "DurableQueueItem",
    "DurableTaskQueue",
    "EnvironmentSlot",
    "DistributedEnvironmentScheduler",
    "InferenceProvider",
    "InferenceRequest",
    "InferenceResponse",
    "MockInferenceProvider",
    "DeterministicSeedPartitioner",
    "DistributedRolloutBatch",
    "DistributedParallelRolloutRunner",
    "create_rollout_runner",
    "run_parallel_rollouts",
    "DEFAULT_QUEUE_TTL_S",
    "DEFAULT_MAX_RETRIES",
]

