import random


class RandomAgent:
    def act(self, obs: dict, action_types: frozenset[str]) -> dict:
        return {"type": random.choice(sorted(action_types))}
