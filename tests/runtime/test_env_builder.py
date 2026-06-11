# tests/runtime/test_env_builder.py
import copy
import socket
import pytest
from forge.runtime.determinism import DeterminismError
from forge.runtime.env import ForgeEnv
from forge.runtime.env_builder import (
    DeterminismConfig,
    DeterminismViolation,
    EnvBuilder,
    canonical_dumps,
)
from forge.runtime.transition import TransitionResult


class CounterFactory:
    def create(self, ctx, options):
        return {"counter": {"c_0": {"id": "c_0", "value": ctx.rng.randint(0, 1000)}}}


def increment(state, action, ctx):
    new_state = copy.deepcopy(state)
    new_state["counter"]["c_0"]["value"] += 1
    return TransitionResult(state=new_state, events=[{"type": "incremented", "entity_id": "c_0"}])


def make_builder(factory=None):
    return (
        EnvBuilder("builder_env", domain="test", max_steps=10)
        .with_initial_state(factory or CounterFactory())
        .with_transition("increment", increment)
    )


def test_build_returns_working_forge_env():
    env = make_builder().build()
    assert isinstance(env, ForgeEnv)
    obs, info = env.reset(seed=42)
    assert obs["counter"]["c_0"]["value"] >= 0
    obs, reward, terminated, truncated, _ = env.step({"type": "increment"})
    assert obs["counter"]["c_0"]["value"] >= 1


def test_build_runs_determinism_check_by_default():
    class WallClockFactory:
        def create(self, ctx, options):
            import time
            return {"counter": {"c_0": {"id": "c_0", "value": time.time_ns()}}}

    with pytest.raises(DeterminismError):
        make_builder(WallClockFactory()).build()


def test_build_can_skip_determinism_check():
    class WallClockFactory:
        def create(self, ctx, options):
            import time
            return {"counter": {"c_0": {"id": "c_0", "value": time.time_ns()}}}

    env = make_builder(WallClockFactory()).build(verify=False)
    assert isinstance(env, ForgeEnv)


def test_floats_in_initial_state_rejected():
    class FloatFactory:
        def create(self, ctx, options):
            return {"counter": {"c_0": {"id": "c_0", "value": 0.5}}}

    env = make_builder(FloatFactory()).build(verify=False)
    with pytest.raises(DeterminismViolation, match="float"):
        env.reset(seed=1)


def test_floats_in_transition_result_rejected():
    def float_transition(state, action, ctx):
        new_state = copy.deepcopy(state)
        new_state["counter"]["c_0"]["value"] = 1.5
        return TransitionResult(state=new_state, events=[])

    env = (
        EnvBuilder("builder_env", domain="test", max_steps=10)
        .with_initial_state(CounterFactory())
        .with_transition("go_float", float_transition)
        .build(verify=False)
    )
    env.reset(seed=1)
    with pytest.raises(DeterminismViolation, match="float"):
        env.step({"type": "go_float"})


def test_floats_allowed_when_config_disabled():
    class FloatFactory:
        def create(self, ctx, options):
            return {"counter": {"c_0": {"id": "c_0", "value": 0.5}}}

    env = (
        EnvBuilder("builder_env", domain="test", max_steps=10)
        .with_initial_state(FloatFactory())
        .with_transition("increment", increment)
        .with_determinism(DeterminismConfig(integers_only=False))
        .build(verify=False)
    )
    obs, _ = env.reset(seed=1)
    assert obs["counter"]["c_0"]["value"] == 0.5


def test_network_calls_inside_transition_blocked():
    def network_transition(state, action, ctx):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("example.com", 80))
        return TransitionResult(state=state, events=[])

    env = (
        EnvBuilder("builder_env", domain="test", max_steps=10)
        .with_initial_state(CounterFactory())
        .with_transition("call_api", network_transition)
        .build(verify=False)
    )
    env.reset(seed=1)
    with pytest.raises(DeterminismViolation, match="network"):
        env.step({"type": "call_api"})


def test_filesystem_access_inside_transition_blocked(tmp_path):
    target = tmp_path / "leak.txt"

    def fs_transition(state, action, ctx):
        with open(target, "w") as fh:
            fh.write("side effect")
        return TransitionResult(state=state, events=[])

    env = (
        EnvBuilder("builder_env", domain="test", max_steps=10)
        .with_initial_state(CounterFactory())
        .with_transition("write_file", fs_transition)
        .build(verify=False)
    )
    env.reset(seed=1)
    with pytest.raises(DeterminismViolation, match="filesystem"):
        env.step({"type": "write_file"})
    assert not target.exists()


def test_fresh_startup_clears_factory_cache_on_reset():
    class CachingFactory:
        def __init__(self):
            self.cache = {"stale": True}
            self.cleared = 0

        def clear_cache(self):
            self.cache.clear()
            self.cleared += 1

        def create(self, ctx, options):
            return {"counter": {"c_0": {"id": "c_0", "value": 0}}}

    factory = CachingFactory()
    env = make_builder(factory).build(verify=False)
    env.reset(seed=1)
    env.reset(seed=2)
    assert factory.cleared == 2
    assert factory.cache == {}


def test_seeded_uuid_generator_reproducible():
    class UUIDFactory:
        def create(self, ctx, options):
            return {"ids": {"u_0": {"id": ctx.uuid_generator.next(), "value": 0}}}

    env = make_builder(UUIDFactory()).build(verify=False)
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    obs3, _ = env.reset(seed=43)
    assert obs1["ids"]["u_0"]["id"] == obs2["ids"]["u_0"]["id"]
    assert obs1["ids"]["u_0"]["id"] != obs3["ids"]["u_0"]["id"]


def test_canonical_dumps_sorts_keys():
    assert canonical_dumps({"b": 1, "a": {"d": 2, "c": 3}}) == '{"a":{"c":3,"d":2},"b":1}'


def test_built_env_is_compatible_with_determinism_check():
    from forge.runtime.determinism import run_determinism_check

    env = make_builder().build()
    report = run_determinism_check(env, seed=7, num_steps=4)
    assert report.passed is True
