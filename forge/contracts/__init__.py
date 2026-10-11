"""Interface contracts every Forge environment implements.

Eleven concerns recur across all four environment families (in-process,
container, CLI, browser). Each is one ABC here; the `Environment` facade
composes the ten that describe an environment's state and behavior.
`EpisodeController` is not part of the facade because it drives an environment
from the outside — the same environment may be run by different controllers.
"""
from forge.contracts.backend import ExecutionBackend, TransitionHandler
from forge.contracts.checkpoint import PolicyCheckpoint
from forge.contracts.dataset import TaskSource, normalize_task
from forge.contracts.environment import Environment
from forge.contracts.episode import (
    BaseEpisodeConfig,
    BaseEpisodeResult,
    EpisodeController,
    TrajectoryWriter,
)
from forge.contracts.evaluation import EpisodeEvaluation
from forge.contracts.gpu import (
    APIGatewaySpec,
    GPUDeviceSpec,
    GPUInferenceContract,
    InferenceMode,
)
from forge.contracts.inference import (
    InferenceBatchRequest,
    InferenceBatchResponse,
    InferenceProvider,
    InferenceRequest,
    InferenceResponse,
)
from forge.contracts.initial_state import InitialStateProvider
from forge.contracts.observation import ObservationEncoder
from forge.contracts.persona import (
    PersonaBehavior,
    PersonaDriver,
    PersonaPopulation,
    PersonaProfile,
    PersonaScheduler,
    PersonaSpec,
    PersonaTick,
    PersonaTraits,
    PersonaTurn,
    PersonaView,
)
from forge.contracts.prompting import PromptTemplate
from forge.contracts.reward import Rubric, Verifier
from forge.contracts.rollout import RolloutOutcome, RolloutRecord
from forge.contracts.sandbox import (
    FirecrackerVMConfig,
    IsolationBackendType,
    SandboxInstance,
    SandboxIsolationBackend,
)
from forge.contracts.state import StateManager
from forge.contracts.termination import (
    BUDGET_REASONS,
    BudgetTerminationPolicy,
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
from forge.contracts.verification import (
    GroupVerificationSpec,
    TrustBoundaryReport,
    TrustLevel,
    VerifierGroup,
    VerifierGroupAdapter,
)

__all__ = [
    "Action",
    "ActionResult",
    "AgentAdapter",
    "BUDGET_REASONS",
    "BaseEpisodeConfig",
    "BaseEpisodeResult",
    "BudgetTerminationPolicy",
    "CheckResult",
    "CompositeTerminationPolicy",
    "DeadEndTerminationPolicy",
    "Environment",
    "EpisodeController",
    "EpisodeEvaluation",
    "ExecutionBackend",
    "FirecrackerVMConfig",
    "GPUDeviceSpec",
    "GPUInferenceContract",
    "GroupVerificationSpec",
    "InferenceBatchRequest",
    "InferenceBatchResponse",
    "InferenceMode",
    "InferenceProvider",
    "InferenceRequest",
    "InferenceResponse",
    "InitialStateProvider",
    "IsolationBackendType",
    "MaxStepsTerminationPolicy",
    "Observation",
    "ObservationEncoder",
    "PersonaBehavior",
    "PersonaDriver",
    "PersonaPopulation",
    "PersonaProfile",
    "PersonaScheduler",
    "PersonaSpec",
    "PersonaTick",
    "PersonaTraits",
    "PersonaTurn",
    "PersonaView",
    "PolicyCheckpoint",
    "PromptTemplate",
    "RewardBreakdown",
    "RewardComponent",
    "RolloutOutcome",
    "RolloutRecord",
    "Rubric",
    "SandboxInstance",
    "SandboxIsolationBackend",
    "StateManager",
    "StepOutcome",
    "Task",
    "TaskSource",
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
    "TrustBoundaryReport",
    "TrustLevel",
    "VerificationResult",
    "Verifier",
    "VerifierGroup",
    "VerifierGroupAdapter",
    "VerifierTerminationPolicy",
    "normalize_task",
]
