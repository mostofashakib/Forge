# tests/envgen/test_container_env_base.py
import json
import httpx
import pytest
from forge.envgen.container_env_base import ContainerEnvBase


def make_env(handler, env_cls=ContainerEnvBase):
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://testserver")
    return env_cls("http://testserver", client=client)


def app_handler(state: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/forge/reset":
            state.clear()
            state.update({"todos": {}, "resets": state.get("resets", 0) + 1})
            return httpx.Response(200, json={"status": "reset"})
        if request.url.path == "/forge/state":
            return httpx.Response(200, json=state)
        if request.url.path == "/create_todo":
            payload = json.loads(request.content)
            state.setdefault("todos", {})[payload.get("title", "t")] = payload
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"error": "no such endpoint"})

    return handler


def test_reset_calls_forge_reset_and_returns_state():
    state = {"todos": {"stale": {}}}
    env = make_env(app_handler(state))
    obs, info = env.reset()
    assert obs["todos"] == {}
    assert info == {}


def test_step_posts_action_type_endpoint_and_observes():
    state = {}
    env = make_env(app_handler(state))
    env.reset()
    obs, reward, terminated, truncated, info = env.step(
        {"type": "create_todo", "title": "write tests"}
    )
    assert "write tests" in obs["todos"]
    assert reward == 1.0
    assert terminated is False and truncated is False
    assert info["status_code"] == 200


def test_failed_action_gets_zero_reward():
    env = make_env(app_handler({}))
    env.reset()
    obs, reward, *_ , info = env.step({"type": "explode"})
    assert reward == 0.0
    assert info["status_code"] == 404


def test_subclass_customizes_endpoint_and_reward_only():
    class TodoEnv(ContainerEnvBase):
        def action_endpoint(self, action: dict) -> str:
            return "/create_todo"  # domain routes everything to one endpoint

        def compute_reward(self, response, obs) -> float:
            return float(len(obs.get("todos", {})))

    state = {}
    env = make_env(app_handler(state), env_cls=TodoEnv)
    env.reset()
    _, reward1, *_ = env.step({"type": "anything", "title": "a"})
    _, reward2, *_ = env.step({"type": "anything", "title": "b"})
    assert (reward1, reward2) == (1.0, 2.0)


def test_is_a_gymnasium_env_with_dict_spaces():
    import gymnasium

    env = make_env(app_handler({}))
    assert isinstance(env, gymnasium.Env)
    assert isinstance(env.observation_space, gymnasium.spaces.Dict)
    assert isinstance(env.action_space, gymnasium.spaces.Dict)


def test_llm_builder_prompt_instructs_extending_the_base():
    from forge.envgen.agents.state_bridge import _SYSTEM

    assert "ContainerEnvBase" in _SYSTEM
    assert "forge.envgen.container_env_base" in _SYSTEM
    # The old instruction to regenerate everything from scratch must be gone.
    assert "standalone gymnasium.Env subclass" not in _SYSTEM
