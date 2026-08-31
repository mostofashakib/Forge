"""Interface contracts every Forge environment implements.

Eleven concerns recur across all four environment families (in-process,
container, CLI, browser). Each is one ABC here; the `Environment` facade
composes the ten that describe an environment's state and behavior.
`EpisodeController` is not part of the facade because it drives an environment
from the outside — the same environment may be run by different controllers.
"""
from forge.contracts.backend import ExecutionBackend, TransitionHandler
from forge.contracts.dataset import TaskSource, normalize_task
from forge.contracts.environment import Environment
from forge.contracts.episode import (
    BaseEpisodeConfig,
    BaseEpisodeResult,
    EpisodeController,
    TrajectoryWriter,
)
from forge.contracts.evaluation import EpisodeEvaluation
from forge.contracts.initial_state import InitialStateProvider
from forge.contracts.observation import ObservationEncoder
from forge.contracts.prompting import PromptTemplate
from forge.contracts.reward import Rubric, Verifier
from forge.contracts.rollout import RolloutOutcome, RolloutRecord
from forge.contracts.state import StateManager
from forge.contracts.termination import (
    CompositeTerminationPolicy,
    DeadEndTerminationPolicy,
    MaxStepsTerminationPolicy,
    TerminationMonitor,
    TerminationPolicy,
    ThresholdTerminationPolicy,
    VerifierTerminationPolicy,
)
from forge.contracts.tools import ToolProvider
from forge.contracts.transport import Transport, TransportRequest, TransportResponse
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
    "BaseEpisodeConfig",
    "BaseEpisodeResult",
    "CheckResult",
    "CompositeTerminationPolicy",
    "DeadEndTerminationPolicy",
    "Environment",
    "EpisodeController",
    "EpisodeEvaluation",
    "ExecutionBackend",
    "InitialStateProvider",
    "MaxStepsTerminationPolicy",
    "Observation",
    "ObservationEncoder",
    "PromptTemplate",
    "RewardBreakdown",
    "RewardComponent",
    "RolloutOutcome",
    "RolloutRecord",
    "Rubric",
    "StateManager",
    "StepOutcome",
    "Task",
    "TaskSource",
    "normalize_task",
    "Termination",
    "TerminationMonitor",
    "TerminationPolicy",
    "ThresholdTerminationPolicy",
    "ToolParam",
    "ToolProvider",
    "ToolSpec",
    "TrajectoryWriter",
    "TransitionHandler",
    "Transport",
    "TransportRequest",
    "TransportResponse",
    "VerificationResult",
    "VerifierTerminationPolicy",
    "Verifier",
]
