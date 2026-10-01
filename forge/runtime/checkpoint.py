"""A point-in-time copy of an in-process environment's episode, for snapshot recovery."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from forge.personas.engine import PersonaCheckpoint
from forge.runtime.context import RuntimeContext
from forge.runtime.trajectory import TrajectoryStore


@dataclass(frozen=True)
class EnvCheckpoint:
    """Everything the next step depends on.

    `context` carries the simulated clock, the episode RNG and the ID
    generators. `np_random_state` is gymnasium's generator state. Every field
    is a private copy, so later steps never change a checkpoint and one
    checkpoint restores any number of times.
    """

    state: dict
    context: RuntimeContext
    np_random_state: dict[str, Any]
    trajectory: TrajectoryStore
    step_count: int
    invalid_action_count: int
    total_reward: float
    personas: PersonaCheckpoint
