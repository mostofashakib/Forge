# tests/runtime/test_tool_surface.py
import copy
from forge.runtime.env_builder import EnvBuilder
from forge.runtime.snapshot import ToolParam, ToolSpec
from forge.runtime.transition import TransitionResult


class CounterFactory:
    def create(self, ctx, options):
        return {"counter": {"c_0": {"id": "c_0", "value": ctx.rng.randint(0, 1000)}}}


def increment(state, action, ctx):
    new_state = copy.deepcopy(state)
    new_state["counter"]["c_0"]["value"] += action.get("amount", 1)
    return TransitionResult(state=new_state, events=[])


def build_env():
    return (
        EnvBuilder("surface_env", domain="test", max_steps=10)
        .with_initial_state(CounterFactory())
        .with_transition(
            "increment",
            increment,
            description="Increase the counter by the given amount",
            params=[ToolParam(name="amount", type="integer", required=False)],
        )
        .with_transition("noop", lambda s, a, c: TransitionResult(state=s, events=[]))
        .build(verify=False)
    )


def test_tool_surface_lists_every_registered_tool():
    env = build_env()
    surface = env.tool_surface()
    assert {spec.name for spec in surface} == {"increment", "noop"}
    assert all(isinstance(spec, ToolSpec) for spec in surface)


def test_tool_surface_carries_description_and_params():
    env = build_env()
    spec = next(s for s in env.tool_surface() if s.name == "increment")
    assert spec.description == "Increase the counter by the given amount"
    assert spec.params[0].name == "amount"
    assert spec.params[0].type == "integer"
    assert spec.params[0].required is False


def test_tool_surface_defaults_for_undocumented_tools():
    env = build_env()
    spec = next(s for s in env.tool_surface() if s.name == "noop")
    assert spec.description == ""
    assert spec.params == []


def test_tool_surface_is_sorted_and_serializable():
    env = build_env()
    surface = env.tool_surface()
    assert [s.name for s in surface] == sorted(s.name for s in surface)
    payload = [s.model_dump() for s in surface]
    assert payload[0]["name"] == "increment"


def test_tool_surface_matches_action_types():
    env = build_env()
    assert {s.name for s in env.tool_surface()} == set(env.action_types)
