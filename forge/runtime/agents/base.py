from typing import Protocol


class AgentAdapter(Protocol):
    def act(self, obs: dict, action_types: frozenset[str]) -> dict: ...
