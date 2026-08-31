"""How per-episode state is set up at the start of a rollout."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.runtime.context import RuntimeContext


class InitialStateProvider(ABC):
    """Produces the starting state for one episode.

    `seed` is an explicit keyword rather than an entry in `options` because
    call sites previously disagreed about where it lived — some passed it in
    `options`, others read `ctx.seed`. A seed of None means the provider's
    fixed baseline, which is distinct from seed 0.
    """

    @abstractmethod
    def reset(
        self,
        ctx: "RuntimeContext",
        *,
        seed: int | None,
        options: Mapping[str, object],
    ) -> dict:
        """Return the initial state for an episode."""
