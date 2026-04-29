import random
import yaml
from pathlib import Path


class ScriptedAgent:
    def __init__(self, path: str) -> None:
        self._actions: list[dict] = yaml.safe_load(Path(path).read_text()) or []
        self._index = 0

    def act(self, obs: dict, action_types: frozenset[str]) -> dict:
        if not self._actions:
            return {"type": random.choice(sorted(action_types))}
        action = self._actions[self._index % len(self._actions)]
        self._index += 1
        if action.get("type") not in action_types:
            return {"type": random.choice(sorted(action_types))}
        return dict(action)
