# tests/runtime/test_replay_episode.py
import copy
import json
from forge.runtime.env_builder import EnvBuilder
from forge.runtime.replay import replay_episode
from forge.runtime.transition import TransitionResult


class CounterFactory:
    def create(self, ctx, options):
        return {"counter": {"c_0": {"id": "c_0", "value": ctx.rng.randint(0, 1000)}}}


def increment(state, action, ctx):
    new_state = copy.deepcopy(state)
    new_state["counter"]["c_0"]["value"] += 1
    return TransitionResult(state=new_state, events=[{"type": "incremented", "entity_id": "c_0"}])


def build_env():
    return (
        EnvBuilder("replay_env", domain="test", max_steps=10)
        .with_initial_state(CounterFactory())
        .with_transition("increment", increment)
        .build(verify=False)
    )


def record_episode(seed: int, n_steps: int):
    env = build_env()
    env.reset(seed=seed)
    for _ in range(n_steps):
        env.step({"type": "increment"})
    return env._traj_store._steps


def test_replay_reproduces_recorded_episode():
    recorded = record_episode(seed=42, n_steps=4)
    result = replay_episode(build_env(), seed=42, steps=recorded)
    assert result.matched is True
    assert result.steps_replayed == 4
    assert result.mismatches == []


def test_replay_detects_tampered_state_hash():
    recorded = record_episode(seed=42, n_steps=3)
    recorded[1] = recorded[1].model_copy(update={"state_hash_after": "sha256:tampered"})
    result = replay_episode(build_env(), seed=42, steps=recorded)
    assert result.matched is False
    assert result.mismatches[0].step_index == 1
    assert result.mismatches[0].field == "state_hash_after"


def test_replay_detects_tampered_reward():
    recorded = record_episode(seed=42, n_steps=3)
    recorded[2] = recorded[2].model_copy(update={"reward": 99.0})
    result = replay_episode(build_env(), seed=42, steps=recorded)
    assert result.matched is False
    assert result.mismatches[0].step_index == 2
    assert result.mismatches[0].field == "reward"


def test_replay_with_wrong_seed_mismatches():
    recorded = record_episode(seed=42, n_steps=2)
    result = replay_episode(build_env(), seed=7, steps=recorded)
    assert result.matched is False


def test_replay_accepts_dict_steps_from_jsonl():
    recorded = record_episode(seed=42, n_steps=3)
    jsonl_steps = [json.loads(step.model_dump_json()) for step in recorded]
    result = replay_episode(build_env(), seed=42, steps=jsonl_steps)
    assert result.matched is True
    assert result.steps_replayed == 3


def test_replay_total_reward_matches_recording():
    recorded = record_episode(seed=42, n_steps=4)
    result = replay_episode(build_env(), seed=42, steps=recorded)
    assert result.total_reward == sum(s.reward for s in recorded)
