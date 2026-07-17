from __future__ import annotations
from abc import ABC, abstractmethod
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext


class EnvGenAgent(ABC):
    agent_id: str = "agent"
    depends_on: list[str] = []
    optional_depends_on: list[str] = []
    produces: list[str] = []

    @abstractmethod
    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None: ...
