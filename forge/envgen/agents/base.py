from __future__ import annotations
from abc import ABC, abstractmethod
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext


class EnvGenAgent(ABC):
    depends_on: list[str] = []
    produces: str = ""

    @abstractmethod
    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None: ...
