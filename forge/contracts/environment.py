"""The composed contract a complete Forge environment satisfies."""
from __future__ import annotations

from abc import ABC, abstractmethod

from forge.contracts.backend import ExecutionBackend
from forge.contracts.dataset import TaskSource
from forge.contracts.initial_state import InitialStateProvider
from forge.contracts.observation import ObservationEncoder
from forge.contracts.prompting import PromptTemplate
from forge.contracts.reward import Rubric
from forge.contracts.state import StateManager
from forge.contracts.termination import TerminationPolicy
from forge.contracts.tools import ToolProvider
from forge.contracts.transport import Transport


class Environment(ABC):
    """Ten of the eleven concerns, composed.

    Seven members are required because every environment family has them. The
    three optional ones are exactly those a family can legitimately lack: a
    shell environment has no tool schema, a pure-Python environment has no
    transport, and an environment driven by a trainer that supplies its own
    prompting has no template.

    EpisodeController is deliberately absent — see forge/contracts/episode.py.
    """

    @property
    @abstractmethod
    def task_source(self) -> TaskSource: ...

    @property
    @abstractmethod
    def initial_state(self) -> InitialStateProvider: ...

    @property
    @abstractmethod
    def observations(self) -> ObservationEncoder: ...

    @property
    @abstractmethod
    def backend(self) -> ExecutionBackend: ...

    @property
    @abstractmethod
    def state(self) -> StateManager: ...

    @property
    @abstractmethod
    def rubric(self) -> Rubric: ...

    @property
    @abstractmethod
    def termination(self) -> TerminationPolicy: ...

    # ------------------------------------------------------------------
    # Optional concerns
    # ------------------------------------------------------------------

    @property
    def prompt(self) -> PromptTemplate | None:
        return None

    @property
    def tools(self) -> ToolProvider | None:
        return None

    @property
    def transport(self) -> Transport | None:
        return None
