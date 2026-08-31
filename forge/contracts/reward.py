"""How the model's behavior is scored."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from forge.contracts._arity import check_subclass_arity
from forge.contracts.types import RewardBreakdown, Task, VerificationResult

if TYPE_CHECKING:
    from forge.runtime.trajectory import Trajectory


class Verifier(ABC):
    """Decides whether a task was accomplished.

    Separate from Rubric because passing and scoring are different questions:
    a verifier answers "did it happen", a rubric answers "how much is that
    worth". Keeping them apart lets one rubric weigh several verifiers.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        check_subclass_arity(cls, "verify", ("state", "trajectory", "task"))

    @abstractmethod
    def verify(
        self, state: dict, trajectory: "Trajectory", task: Task | None
    ) -> VerificationResult:
        """Check the task's conditions against the final state and trajectory."""


class Rubric(ABC):
    """Turns verification into a reward.

    Implemented by string matching, unit tests, an LLM judge, a tiered engine,
    or any combination — the contract does not care which, only that a
    breakdown comes back so the components are auditable.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        check_subclass_arity(
            cls, "score", ("state", "trajectory", "verifier_results", "task")
        )

    @abstractmethod
    def score(
        self,
        state: dict,
        trajectory: "Trajectory",
        verifier_results: Sequence[VerificationResult],
        task: Task | None,
    ) -> RewardBreakdown:
        """Return the total reward and the components that produced it."""
