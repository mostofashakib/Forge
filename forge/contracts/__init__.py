"""Interface contracts every Forge environment implements.

Eleven concerns recur across all four environment families (in-process,
container, CLI, browser). Each is one ABC here; the `Environment` facade
composes the ten that describe an environment's state and behavior.
`EpisodeController` is not part of the facade because it drives an environment
from the outside — the same environment may be run by different controllers.
"""
from forge.contracts.dataset import TaskSource
from forge.contracts.initial_state import InitialStateProvider
from forge.contracts.state import StateManager
from forge.contracts.termination import TerminationPolicy
from forge.contracts.types import (
    Action,
    ActionResult,
    AgentAdapter,
    CheckResult,
    Observation,
    RewardBreakdown,
    RewardComponent,
    StepOutcome,
    Task,
    Termination,
    ToolParam,
    ToolSpec,
    VerificationResult,
)

__all__ = [
    "Action",
    "ActionResult",
    "AgentAdapter",
    "CheckResult",
    "InitialStateProvider",
    "Observation",
    "RewardBreakdown",
    "RewardComponent",
    "StateManager",
    "StepOutcome",
    "Task",
    "TaskSource",
    "Termination",
    "TerminationPolicy",
    "ToolParam",
    "ToolSpec",
    "VerificationResult",
]
