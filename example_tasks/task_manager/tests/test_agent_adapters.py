#!/usr/bin/env python3
"""The agent's two adapter layers: which model answers, and where it acts.

The point of the package under test is that neither half knows the other, so
these checks mostly try to break that: send one provider's dialect to another,
hand the decoder arguments no model should have produced, and drive the loop
with a backend that fails. The last group is the one that matters most -- a
scripted episode against the real world, graded by the real verifier, so that
"the loop ran" and "the loop did the task" cannot be confused.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from task_sim.service import export_state, seed_database
from task_sim.tool_definitions import get_tool_definitions

try:
    from agent.backends import BackendError, InProcessBackend, ToolBackend
    from agent.config import DEFAULT_MODEL, AgentConfig
    from agent.loop import ToolLoop
    from agent.prompts import system_prompt
    from agent.providers import (
        AnthropicProvider,
        Completion,
        OllamaProvider,
        OpenAIProvider,
        OpenRouterProvider,
        Provider,
        ProviderError,
        ProviderRejected,
        ToolCall,
        _decode_arguments,
        _encode_anthropic,
        create_provider,
        get_provider_class,
        provider_names,
        split_spec,
    )
    from agent.schemas import (
        SchemaViolation,
        build_action_schema,
        drop_nulls,
        render_catalogue,
        strict_parameters,
        validate,
    )
    from agent.trajectory import build_trajectory
except ImportError:
    # The agent runs on the machine that starts the run, so `.dockerignore`
    # keeps it out of the build context. Harbor still uploads all of `tests/`,
    # and a suite that failed in the graded image over a package that was never
    # meant to ship would report the environment broken. This runs from a
    # checkout, where it is the point, and stands aside inside the image.
    print(__doc__)
    print("\n  skip  the agent package is host-side and does not ship in the graded image")
    sys.exit(0)

FAILURES: list[str] = []

TOOLS = [
    {
        "name": "list_tasks",
        "description": "List tasks.",
        "input_schema": {"type": "object", "properties": {"assignee": {"type": "string"}}},
    }
]


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


# --------------------------------------------------------------------------
# A transport that records instead of sending
# --------------------------------------------------------------------------


class Recorder:
    """Stand in for the network. Captures the payload, returns a canned reply."""

    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.url = ""
        self.payload: dict[str, Any] = {}
        self.headers: dict[str, str] = {}

    def __call__(self, url, payload, headers, timeout_sec):  # noqa: ANN001
        self.url = url
        self.payload = json.loads(json.dumps(payload))
        self.headers = dict(headers)
        return self.reply


def with_transport(provider: Provider, reply: dict[str, Any]) -> tuple[Completion, Recorder]:
    from agent import providers as module

    recorder = Recorder(reply)
    original = module._post_json
    module._post_json = recorder
    try:
        completion = provider.complete(
            system="be careful",
            messages=[
                {"role": "user", "content": "do the thing"},
                {
                    "role": "assistant",
                    "content": "looking",
                    "tool_calls": [ToolCall("c1", "list_tasks", {"assignee": "U004"})],
                },
                {"role": "tool", "tool_call_id": "c1", "name": "list_tasks", "content": "{}"},
            ],
            tools=TOOLS,
        )
    finally:
        module._post_json = original
    return completion, recorder


# Ollama answers with the action object its `format` schema constrained, not
# with a `tool_calls` array: `format` and `tools` cannot both be sent.
OLLAMA_REPLY = {
    "message": {
        "content": json.dumps(
            {
                "thought": "on it",
                "actions": [{"tool": "list_tasks", "arguments": {"assignee": "U004"}}],
                "answer": "",
            }
        )
    },
    "done_reason": "stop",
    "prompt_eval_count": 11,
    "eval_count": 7,
}
OPENAI_REPLY = {
    "choices": [
        {
            "finish_reason": "tool_calls",
            "message": {
                "content": "on it",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {
                            "name": "list_tasks",
                            "arguments": '{"assignee": "U004"}',
                        },
                    }
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
}
ANTHROPIC_REPLY = {
    "content": [
        {"type": "text", "text": "on it"},
        {"type": "tool_use", "id": "toolu_a", "name": "list_tasks", "input": {"assignee": "U004"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 11, "output_tokens": 7},
}


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def test_a_model_spec_resolves_to_the_provider_it_names() -> None:
    check("the default provider is ollama", DEFAULT_MODEL.startswith("ollama/"), DEFAULT_MODEL)
    cases = {
        # A bare model name is a model, not a provider.
        "qwen3.6:35b": ("ollama", "qwen3.6:35b"),
        "gemma4:26b": ("ollama", "gemma4:26b"),
        "ollama/gemma4:26b": ("ollama", "gemma4:26b"),
        "anthropic/claude-opus-5": ("anthropic", "claude-opus-5"),
        # Only the first segment is consumed, so a provider whose model ids
        # contain slashes still resolves.
        "openrouter/anthropic/claude-opus-5": ("openrouter", "anthropic/claude-opus-5"),
        # A registered name with nothing after it means that provider's default.
        "openai": ("openai", ""),
        # An unregistered head is not a provider, and must not be eaten as one.
        "acme/some-model": ("ollama", "acme/some-model"),
        "": ("ollama", ""),
    }
    for spec, expected in cases.items():
        check(f"{spec!r} -> {expected}", split_spec(spec) == expected, str(split_spec(spec)))


def test_the_default_needs_nothing_and_the_others_say_what_they_need() -> None:
    provider = create_provider(environ={})
    check("the default builds with an empty environment", provider.name == "ollama")
    check("and needs no credential", provider.api_key is None)
    check("and points at localhost", "127.0.0.1" in provider.base_url, provider.base_url)

    try:
        create_provider("openai/gpt-5", environ={})
    except ProviderError as exc:
        check("a keyed provider refuses without one", "OPENAI_API_KEY" in str(exc), str(exc))
    else:
        check("a keyed provider refuses without one", False, "it did not")

    keyed = create_provider("openai/gpt-5", environ={"OPENAI_API_KEY": "sk-test"})
    check("and accepts one from the environment", keyed.api_key == "sk-test")
    # This repository's own .env spells it OPEN_ROUTER_KEY.
    router = create_provider(
        "openrouter/anthropic/claude-opus-5", environ={"OPEN_ROUTER_KEY": "sk-or"}
    )
    check("openrouter reads OPEN_ROUTER_KEY", router.api_key == "sk-or")
    check("a credential is never described", "sk-or" not in json.dumps(router.describe()))

    unknown = False
    try:
        create_provider("acme", environ={})
    except (KeyError, ProviderError):
        unknown = True
    check("an unregistered bare name is a model, not a provider", not unknown)


def test_config_defaults_survive_an_override_that_was_not_given() -> None:
    # Harbor passes model_name=None when no -m was given. That must not erase
    # the default, which is the difference between "no model" and "the default
    # model".
    config = AgentConfig.from_env({}, model=None, max_turns=None)
    check("an absent override is not an override", config.model == DEFAULT_MODEL, config.model)
    from_env = AgentConfig.from_env({"TASK_AGENT_MODEL": "openai/gpt-5", "TASK_AGENT_MAX_TURNS": "9"})
    check("the environment is read", from_env.model == "openai/gpt-5", from_env.model)
    check("including numbers", from_env.max_turns == 9, str(from_env.max_turns))
    explicit = AgentConfig.from_env({"TASK_AGENT_MODEL": "openai/gpt-5"}, model="ollama/gemma4:26b")
    check("an explicit value beats the environment", explicit.model == "ollama/gemma4:26b")


# --------------------------------------------------------------------------
# Wire formats
# --------------------------------------------------------------------------


def test_every_provider_decodes_the_same_turn_identically() -> None:
    cases = [
        (OllamaProvider("m"), OLLAMA_REPLY, "action-1"),
        (OpenAIProvider("m", api_key="k"), OPENAI_REPLY, "call_a"),
        (OpenRouterProvider("m", api_key="k"), OPENAI_REPLY, "call_a"),
        (AnthropicProvider("m", api_key="k"), ANTHROPIC_REPLY, "toolu_a"),
    ]
    for provider, reply, expected_id in cases:
        completion, _ = with_transport(provider, reply)
        check(f"{provider.name}: text", completion.text == "on it", completion.text)
        check(f"{provider.name}: one call", len(completion.tool_calls) == 1)
        call = completion.tool_calls[0]
        check(f"{provider.name}: name", call.name == "list_tasks", call.name)
        check(f"{provider.name}: arguments", call.arguments == {"assignee": "U004"}, str(call.arguments))
        check(f"{provider.name}: call id", call.id == expected_id, call.id)
        check(f"{provider.name}: prompt tokens", completion.prompt_tokens == 11)
        check(f"{provider.name}: completion tokens", completion.completion_tokens == 7)


def test_each_provider_is_sent_its_own_dialect() -> None:
    _, ollama = with_transport(OllamaProvider("m"), OLLAMA_REPLY)
    check("ollama posts to /api/chat", ollama.url.endswith("/api/chat"), ollama.url)
    check("ollama gets the system prompt as a message",
          ollama.payload["messages"][0]["role"] == "system")
    check("ollama gets a schema, not a tool list", "format" in ollama.payload)
    # Sending both is the failure this pins: the reply comes back schema-shaped
    # with tool_calls null, and native tool calling stops with no error.
    check("and never both", "tools" not in ollama.payload, str(sorted(ollama.payload)))
    check("the schema enumerates the real tools",
          ollama.payload["format"]["properties"]["actions"]["items"]["properties"]["tool"]["const"]
          == "list_tasks")
    check("the catalogue reaches the prompt",
          "list_tasks" in ollama.payload["messages"][0]["content"])
    # The history is replayed in the shape the schema constrains, so previous
    # turns do not contradict the one being decoded.
    replayed = json.loads(ollama.payload["messages"][2]["content"])
    check("ollama replays its own action object", replayed["actions"][0]["tool"] == "list_tasks",
          str(replayed))
    check("with arguments as an object",
          isinstance(replayed["actions"][0]["arguments"], dict))
    check("ollama is told which tool a result came from",
          ollama.payload["messages"][3].get("tool_name") == "list_tasks")

    _, openai = with_transport(OpenAIProvider("m", api_key="k"), OPENAI_REPLY)
    check("openai posts to /chat/completions", openai.url.endswith("/chat/completions"), openai.url)
    check("openai sends a bearer token", openai.headers.get("Authorization") == "Bearer k")
    function = openai.payload["tools"][0]["function"]
    check("openai gets strict tools", function.get("strict") is True, str(function.get("strict")))
    check("openai gets `parameters`, not `input_schema`", "parameters" in function)
    check("strict mode requires every property",
          set(function["parameters"]["required"]) == set(function["parameters"]["properties"]),
          str(function["parameters"]["required"]))
    replayed = openai.payload["messages"][2]["tool_calls"][0]["function"]["arguments"]
    check("openai gets replayed arguments as a string", isinstance(replayed, str), repr(replayed))
    check("openai gets the tool result keyed by call id",
          openai.payload["messages"][3]["tool_call_id"] == "c1")

    _, anthropic = with_transport(AnthropicProvider("m", api_key="k"), ANTHROPIC_REPLY)
    check("anthropic posts to /messages", anthropic.url.endswith("/messages"), anthropic.url)
    check("anthropic sends x-api-key", anthropic.headers.get("x-api-key") == "k")
    check("anthropic sends a version", "anthropic-version" in anthropic.headers)
    check("anthropic gets the system prompt out of band",
          anthropic.payload.get("system") == "be careful")
    check("anthropic gets `input_schema`, not `parameters`",
          "input_schema" in anthropic.payload["tools"][0])
    check("anthropic gets a tool_use block",
          anthropic.payload["messages"][1]["content"][1]["type"] == "tool_use")
    check("and the result as a user turn",
          anthropic.payload["messages"][2]["content"][0]["type"] == "tool_result")


def test_anthropic_folds_a_turn_of_parallel_results_into_one_message() -> None:
    encoded = _encode_anthropic(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    ToolCall("a", "get_task", {"task_id": "TASK001"}),
                    ToolCall("b", "get_task", {"task_id": "TASK002"}),
                ],
            },
            {"role": "tool", "tool_call_id": "a", "name": "get_task", "content": "{}"},
            {"role": "tool", "tool_call_id": "b", "name": "get_task", "content": "{}"},
        ]
    )
    check("the turns alternate", [m["role"] for m in encoded] == ["user", "assistant", "user"],
          str([m["role"] for m in encoded]))
    check("both results land in one turn", len(encoded[2]["content"]) == 2)
    # An assistant turn is illegal with an empty content list, and a model that
    # emits a call and no text produces exactly that.
    check("an empty assistant turn is filled in", len(encoded[1]["content"]) == 2)
    silent = _encode_anthropic([{"role": "assistant", "content": ""}])
    check("a silent assistant turn is still legal", len(silent[0]["content"]) == 1)


def test_arguments_are_decoded_without_ever_raising() -> None:
    # A model that emits malformed arguments has made a mistake it should be
    # told about by the world, not one that should end the episode here.
    cases: list[tuple[Any, str]] = [
        ({"a": 1}, "an object"),
        ('{"a": 1}', "a JSON string"),
        ("", "an empty string"),
        ("   ", "whitespace"),
        (None, "null"),
        ('{"a": 1', "a truncated object"),
        ("[1, 2]", "a JSON list"),
        ("not json at all", "prose"),
        (12345, "a number"),
    ]
    for raw, label in cases:
        decoded = _decode_arguments(raw, "list_tasks")
        check(f"{label} decodes to a dict", isinstance(decoded, dict), repr(decoded))
    check("a good object survives", _decode_arguments({"a": 1}, "t") == {"a": 1})
    check("a good string survives", _decode_arguments('{"a": 1}', "t") == {"a": 1})
    check("garbage is preserved for the error message",
          "__unparsable_arguments__" in _decode_arguments("{oops", "t"))


def test_a_service_failure_is_a_provider_error_and_not_a_crash() -> None:
    from agent import providers as module

    def explode(url, payload, headers, timeout_sec):  # noqa: ANN001
        raise ProviderRejected("HTTP 429: slow down")

    original = module._post_json
    module._post_json = explode
    try:
        try:
            OllamaProvider("m").complete(system="s", messages=[], tools=TOOLS)
        except ProviderError as exc:
            check("a rejection surfaces as ProviderError", "429" in str(exc), str(exc))
        else:
            check("a rejection surfaces as ProviderError", False, "it did not")
    finally:
        module._post_json = original

    empty = OpenAIProvider("m", api_key="k")
    from agent import providers as module2

    original = module2._post_json
    module2._post_json = Recorder({"choices": []})
    try:
        try:
            empty.complete(system="s", messages=[], tools=TOOLS)
        except ProviderError:
            check("an answer with no choices is an error, not an empty turn", True)
        else:
            check("an answer with no choices is an error, not an empty turn", False, "it passed")
    finally:
        module2._post_json = original


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


class ScriptedProvider(Provider):
    """Answers from a list. The loop's behaviour, with the model taken out."""

    name = "scripted"
    default_model = "scripted"

    def __init__(self, turns: list[Completion]) -> None:
        super().__init__("scripted")
        self.turns = list(turns)
        self.seen: list[list[dict[str, Any]]] = []

    def complete(self, *, system, messages, tools):  # noqa: ANN001
        self.seen.append(list(messages))
        if not self.turns:
            return Completion(text="done")
        return self.turns.pop(0)


class BrokenBackend(ToolBackend):
    async def list_tools(self):
        return list(TOOLS)

    async def call(self, name, arguments):  # noqa: ANN001
        raise BackendError("the socket is gone")


def _call(name: str, arguments: dict[str, Any], call_id: str = "c1") -> Completion:
    return Completion(text="", tool_calls=(ToolCall(call_id, name, arguments),))


def test_the_loop_stops_for_the_right_reasons() -> None:
    with Sandbox() as sandbox:
        answered = asyncio.run(
            ToolLoop(
                ScriptedProvider([_call("list_tasks", {"assignee": "U004"}), Completion(text="finished")]),
                sandbox.backend,
                system="s",
            ).run("go")
        )
        check("a text-only answer ends the episode", answered.stop_reason == "answered",
              answered.stop_reason)
        check("and is kept as the final text", answered.final_text == "finished", answered.final_text)

        capped = asyncio.run(
            ToolLoop(
                ScriptedProvider([_call("list_tasks", {"assignee": f"U{i:03d}"}) for i in range(10)]),
                sandbox.backend,
                system="s",
                max_turns=3,
            ).run("go")
        )
        check("the turn budget is enforced", capped.stop_reason == "max_turns", capped.stop_reason)
        check("and it is the budget, not one more", len(capped.turns) == 3, str(len(capped.turns)))

        stuck = asyncio.run(
            ToolLoop(
                ScriptedProvider([_call("list_tasks", {"assignee": "U004"}) for _ in range(10)]),
                sandbox.backend,
                system="s",
                max_turns=20,
                repeat_limit=3,
            ).run("go")
        )
        check("an identical call repeated is stopped", stuck.stop_reason == "repeating",
              stuck.stop_reason)
        check("and it stops at the limit", len(stuck.turns) == 3, str(len(stuck.turns)))

        # The false positive for the check above: different arguments are
        # progress, not a loop, however many turns it takes.
        varied = asyncio.run(
            ToolLoop(
                ScriptedProvider([_call("list_tasks", {"assignee": f"U{i:03d}"}) for i in range(6)]),
                sandbox.backend,
                system="s",
                max_turns=20,
                repeat_limit=3,
            ).run("go")
        )
        check("varying calls are not mistaken for a loop", varied.stop_reason != "repeating",
              varied.stop_reason)

        broken = asyncio.run(
            ToolLoop(
                ScriptedProvider([_call("list_tasks", {})]), BrokenBackend(), system="s"
            ).run("go")
        )
        check("a dead backend stops the loop", broken.stop_reason == "backend_error",
              broken.stop_reason)
        check("and says so", "socket is gone" in (broken.error or ""), str(broken.error))
        check("with the turn still recorded", len(broken.turns) == 1)


def test_a_refusal_reaches_the_model_intact() -> None:
    with Sandbox() as sandbox:
        provider = ScriptedProvider([_call("get_task", {"task_id": "NOPE"}), Completion(text="ok")])
        result = asyncio.run(ToolLoop(provider, sandbox.backend, system="s").run("go"))
        envelope = result.turns[0].results[0]
        check("the world refused", envelope["ok"] is False, str(envelope))
        shown = [m for m in provider.seen[-1] if m["role"] == "tool"][-1]["content"]
        check("the code reaches the model", "task_not_found" in shown, shown[:200])
        check("so does the message", "not found" in shown.lower(), shown[:200])
        check("the loop did not stop on it", result.stop_reason == "answered", result.stop_reason)


def test_a_long_result_is_truncated_and_says_so() -> None:
    with Sandbox() as sandbox:
        provider = ScriptedProvider([_call("list_tasks", {}), Completion(text="ok")])
        asyncio.run(
            ToolLoop(provider, sandbox.backend, system="s", max_tool_output_chars=200).run("go")
        )
        shown = [m for m in provider.seen[-1] if m["role"] == "tool"][-1]["content"]
        check("the result is cut", len(shown) < 400, str(len(shown)))
        check("and the model is told it was", "truncated" in shown, shown[-120:])

        # The false positive: a short result must arrive whole, with no notice.
        short = ScriptedProvider([_call("get_task", {"task_id": "TASK001"}), Completion(text="ok")])
        asyncio.run(ToolLoop(short, sandbox.backend, system="s").run("go"))
        whole = [m for m in short.seen[-1] if m["role"] == "tool"][-1]["content"]
        check("a short result is not touched", "truncated" not in whole)
        check("and parses back", json.loads(whole)["ok"] is True)


def test_the_system_prompt_carries_the_role_and_the_rules() -> None:
    prompt = system_prompt("handover_coordinator")
    check("the role is present", "handover" in prompt.lower(), prompt[:80])
    check("so is how to finish", "no tool call" in prompt, prompt[-300:])
    check("an unknown role falls back rather than failing",
          bool(system_prompt("no-such-role")))


# --------------------------------------------------------------------------
# End to end: the loop against the real world, graded by the real verifier
# --------------------------------------------------------------------------


class Sandbox:
    """A throwaway workspace behind the in-process backend."""

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="agent-test-"))
        self.db_path = self.root / "tasks.db"
        seed_database(self.db_path, self.root / "seed.sql")
        self.backend = InProcessBackend(self.db_path)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *_exc: Any) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def reward(self) -> float:
        from verifiers.contracts.reassignment import build_contract
        from verifiers.episode import Episode
        from verifiers.presets import TieredRewardEngine

        evaluation = TieredRewardEngine.for_preset(build_contract(), None).evaluate(
            Episode.from_state(export_state(self.db_path))
        )
        return float(evaluation.reward)


CORRECT = [
    _call("list_tasks", {"assignee": "U004"}),
    _call("get_task", {"task_id": "TASK031"}),
    _call("update_task", {"task_id": "TASK006", "assignee": "U002"}),
    _call("update_task", {"task_id": "TASK008", "assignee": "U002"}),
    _call("update_task", {"task_id": "TASK009", "assignee": "U002"}),
    _call("update_task", {"task_id": "TASK032", "assignee": "U002"}),
    _call(
        "update_task",
        {"task_id": "TASK031", "assignee": "U003", "labels": ["design", "frontend", "needs-triage"]},
    ),
    _call(
        "submit_handover_report",
        {
            "summary": "Reassigned Jordan Kim's queue.",
            "task_ids": ["TASK006", "TASK008", "TASK009", "TASK031", "TASK032"],
        },
    ),
    Completion(text="Done."),
]

BULK = [
    _call("list_tasks", {"assignee": "U004"}),
    *[
        _call("update_task", {"task_id": task_id, "assignee": "U002"})
        for task_id in ("TASK006", "TASK008", "TASK009", "TASK031", "TASK032")
    ],
    _call(
        "submit_handover_report",
        {
            "summary": "Moved everything to Morgan.",
            "task_ids": ["TASK006", "TASK008", "TASK009", "TASK031", "TASK032"],
        },
    ),
    Completion(text="Done."),
]


def test_the_loop_actually_changes_the_world_it_is_graded_on() -> None:
    with Sandbox() as sandbox:
        result = asyncio.run(
            ToolLoop(ScriptedProvider(CORRECT), sandbox.backend, system="s", max_turns=20).run("go")
        )
        check("the scripted episode finished", result.stop_reason == "answered", result.stop_reason)
        check("every call was accepted",
              all(envelope["ok"] for turn in result.turns for envelope in turn.results),
              str([e for t in result.turns for e in t.results if not e["ok"]])[:300])
        reward = sandbox.reward()
        check("a correct episode scores 1.0", abs(reward - 1.0) < 1e-9, f"{reward:.4f}")

    # The false positive this guards against: a harness that pays for running
    # rather than for the state it left behind. Same loop, same tool count, one
    # wrong branch.
    with Sandbox() as sandbox:
        asyncio.run(
            ToolLoop(ScriptedProvider(BULK), sandbox.backend, system="s", max_turns=20).run("go")
        )
        reward = sandbox.reward()
        check("bulk-reassigning everyone does not", reward < 1.0, f"{reward:.4f}")
        check("and is not scored as nothing either", reward > 0.0, f"{reward:.4f}")

    # And the other end: a loop that ran, called nothing, and answered.
    with Sandbox() as sandbox:
        asyncio.run(
            ToolLoop(ScriptedProvider([Completion(text="I would rather not.")]),
                     sandbox.backend, system="s").run("go")
        )
        reward = sandbox.reward()
        check("abstaining scores exactly zero", reward == 0.0, f"{reward:.4f}")


def test_the_trajectory_is_valid_atif() -> None:
    with Sandbox() as sandbox:
        result = asyncio.run(
            ToolLoop(ScriptedProvider(CORRECT), sandbox.backend, system="s", max_turns=20).run("go")
        )
    trajectory = build_trajectory(
        result,
        agent_name="tracker-adapter",
        agent_version="1.0.0",
        model_name="ollama/qwen3.6:35b",
        tool_definitions=list(TOOLS),
        session_id="test",
    )
    steps = trajectory["steps"]
    check("step ids are sequential from 1",
          [s["step_id"] for s in steps] == list(range(1, len(steps) + 1)))
    check("the first step is the request", steps[0]["source"] == "user")
    agent_only = {"model_name", "tool_calls", "metrics", "reasoning_content"}
    check("no agent-only field on the request step", not agent_only & set(steps[0]))
    for step in steps[1:]:
        results = (step.get("observation") or {}).get("results", [])
        ids = {call["tool_call_id"] for call in step.get("tool_calls") or []}
        check(f"step {step['step_id']} results reference its own calls",
              all(entry["source_call_id"] in ids for entry in results),
              str([entry["source_call_id"] for entry in results]))
    check("the stop reason is recorded", trajectory["extra"]["stop_reason"] == "answered")

    try:
        from harbor.models.trajectories.trajectory import Trajectory
    except ImportError:
        print("  skip  Harbor is not installed; the structural checks above stand alone")
    else:
        Trajectory.model_validate(trajectory)
        check("Harbor's own schema accepts it", True)



# --------------------------------------------------------------------------
# Output schemas
# --------------------------------------------------------------------------


REAL_TOOLS = list(get_tool_definitions())


def test_the_action_schema_can_only_describe_real_tools() -> None:
    schema = build_action_schema(REAL_TOOLS)
    variants = schema["properties"]["actions"]["items"]["oneOf"]
    named = {variant["properties"]["tool"]["const"] for variant in variants}
    check("every tool has a variant", named == {t["name"] for t in REAL_TOOLS},
          str(named ^ {t["name"] for t in REAL_TOOLS}))
    check("the object is closed", schema["additionalProperties"] is False)
    check("a thought, actions and an answer are all required",
          set(schema["required"]) == {"thought", "actions", "answer"})

    # What the constraint actually buys: a tool that does not exist cannot be
    # expressed, and neither can an argument that is not in its schema.
    item = schema["properties"]["actions"]["items"]
    for payload, why in (
        ({"tool": "list_tasks", "arguments": {"assignee": "U004"}}, None),
        ({"tool": "delete_everything", "arguments": {}}, "an invented tool"),
        ({"tool": "list_tasks", "arguments": {"assignee": 4}}, "a wrongly typed argument"),
        ({"tool": "list_tasks", "arguments": {"nonsense": "x"}}, "an argument that does not exist"),
        ({"tool": "list_tasks", "arguments": {"status": "SHIPPED"}}, "a value outside the enum"),
        ({"tool": "list_tasks"}, "a missing arguments object"),
    ):
        rejected = True
        try:
            validate(payload, item, "action")
            rejected = False
        except SchemaViolation:
            pass
        if why is None:
            check("a real call validates", not rejected, str(payload))
        else:
            check(f"{why} does not", rejected, str(payload))


def test_the_catalogue_says_what_the_schema_cannot() -> None:
    catalogue = render_catalogue(REAL_TOOLS)
    for tool in REAL_TOOLS:
        check(f"{tool['name']} is listed", tool["name"] in catalogue)
    check("optional arguments are marked", "assignee?" in catalogue, catalogue[:200])
    check("required ones are not", "task_id?" not in catalogue.split("get_task(")[1][:40],
          catalogue.split("get_task(")[1][:40])
    check("enum values are spelled out", "PENDING|IN_PROGRESS" in catalogue)
    check("descriptions come along",
          "REPLACES" in catalogue or "replaces" in catalogue.lower())


def test_strict_mode_normalisation_keeps_the_same_tool() -> None:
    update = next(t for t in REAL_TOOLS if t["name"] == "update_task")
    strict = strict_parameters(update["input_schema"])
    original = update["input_schema"]
    check("no property is lost", set(strict["properties"]) == set(original["properties"]))
    check("every property is required", set(strict["required"]) == set(strict["properties"]))
    check("the object stays closed", strict["additionalProperties"] is False)
    check("a required argument keeps its type", strict["properties"]["task_id"]["type"] == "string")
    check("an optional one becomes nullable",
          strict["properties"]["assignee"]["type"] == ["string", "null"])
    check("an optional enum admits null", None in strict["properties"]["status"]["enum"])
    # And the reason it is safe to do that: the nulls never reach the world,
    # which reads `{"title": null}` as "blank the title" rather than "skip it".
    cleaned = drop_nulls({"task_id": "TASK031", "assignee": "U003", "title": None, "labels": None})
    check("nulls are dropped before dispatch", cleaned == {"task_id": "TASK031", "assignee": "U003"},
          str(cleaned))
    check("but a legitimate false is not", drop_nulls({"include_archived": False})
          == {"include_archived": False})
    check("and neither is an empty list", drop_nulls({"labels": []}) == {"labels": []})


def test_the_validator_never_refuses_what_the_world_would_accept() -> None:
    """Soundness, not equality.

    The world's validator is deliberately more forgiving in places -- it matches
    enums case-insensitively and leaves per-property types to the handlers,
    which suits a human at the CLI. The client's job is the published schema, so
    it is allowed to be stricter; what it may never be is stricter in a
    direction the world does not share, because then it would refuse work the
    tracker would have done.
    """
    from task_sim.tool_definitions import validate_tool_payload

    schemas = {t["name"]: t["input_schema"] for t in REAL_TOOLS}

    def client_accepts(name: str, payload: dict[str, Any]) -> bool:
        try:
            validate(payload, schemas[name])
        except SchemaViolation:
            return False
        return True

    # Both must accept: ordinary, correct calls.
    for name, payload in (
        ("update_task", {"task_id": "TASK031", "assignee": "U003"}),
        ("update_task", {"task_id": "TASK031", "labels": ["design", "needs-triage"]}),
        ("update_task", {"task_id": "TASK031", "status": "PENDING"}),
        # The tracker normalizes case; the client must not undo that.
        ("update_task", {"task_id": "TASK031", "status": "pending"}),
        ("list_tasks", {}),
        ("list_tasks", {"include_archived": True}),
        ("get_task", {"task_id": "TASK001"}),
        ("submit_handover_report", {"summary": "s", "task_ids": ["TASK006"]}),
    ):
        check(f"both accept {name}{payload}",
              client_accepts(name, payload) and validate_tool_payload(name, payload) is None,
              f"client={client_accepts(name, payload)} world={validate_tool_payload(name, payload)}")

    # Both must refuse: the schema says so and so does the tracker.
    for name, payload in (
        ("update_task", {"task_id": "TASK031", "status": "SHIPPED"}),
        ("list_tasks", {"assignee": "U004", "sneaky": True}),
        ("get_task", {}),
        ("get_task", {"taskid": "TASK001"}),
    ):
        check(f"both refuse {name}{payload}",
              not client_accepts(name, payload) and validate_tool_payload(name, payload) is not None,
              f"client={client_accepts(name, payload)} world={validate_tool_payload(name, payload)}")

    # The client alone refuses these, and should: the schema declares a type,
    # the model was decoded against that schema, and a mismatch is a violation
    # even though the handler would have coerced it.
    for name, payload in (
        ("update_task", {"task_id": 31}),
        ("update_task", {"task_id": "TASK031", "labels": "needs-triage"}),
        ("list_tasks", {"include_archived": "yes"}),
    ):
        check(f"the client alone refuses {name}{payload}", not client_accepts(name, payload),
              str(payload))

    # And the property that matters: never the other way round.
    for name, payload in (
        ("update_task", {"task_id": "TASK031", "status": "SHIPPED"}),
        ("list_tasks", {"nope": 1}),
        ("get_task", {}),
    ):
        world_refused = validate_tool_payload(name, payload) is not None
        check(f"the client does not accept what the world refuses: {name}{payload}",
              not (world_refused and client_accepts(name, payload)))


def test_a_call_the_schema_does_not_describe_never_reaches_the_world() -> None:
    with Sandbox() as sandbox:
        provider = ScriptedProvider([
            _call("obliterate_task", {"task_id": "TASK031"}),
            _call("update_task", {"task_id": "TASK031", "status": "SHIPPED"}),
            _call("update_task", {"task_id": "TASK031", "assignee": "U003"}),
            Completion(text="done"),
        ])
        result = asyncio.run(
            ToolLoop(provider, sandbox.backend, system="s", max_turns=10).run("go")
        )
        first, second, third = (turn.results[0] for turn in result.turns[:3])
        check("an invented tool is refused", first["ok"] is False, str(first))
        check("and named as unknown", first["error"]["code"] == "unknown_tool", str(first))
        check("with the real ones listed", "update_task" in first["error"]["message"])
        check("a value outside the enum is refused", second["ok"] is False, str(second))
        check("as a schema violation", second["error"]["code"] == "schema_violation", str(second))
        check("naming the field", "status" in second["error"]["message"], str(second))
        check("a good call still goes through", third["ok"] is True, str(third))

        # The refusal has to reach the model in the same vocabulary the world
        # uses, or it has two error formats to learn.
        shown = [m for m in provider.seen[-1] if m["role"] == "tool"]
        check("the model was told", "schema_violation" in json.dumps(shown), str(shown)[:200])

    # And the end of the chain that used to be the soft spot: arguments that
    # could not be decoded at all. Constrained decoding should make it
    # unreachable, but if a service ever lets one through it is refused here
    # rather than dispatched as a call with invented contents.
    with Sandbox() as sandbox:
        broken = Completion(
            tool_calls=(ToolCall("c1", "update_task", _decode_arguments("{oops", "update_task")),)
        )
        result = asyncio.run(
            ToolLoop(ScriptedProvider([broken, Completion(text="done")]),
                     sandbox.backend, system="s").run("go")
        )
        envelope = result.turns[0].results[0]
        check("an undecodable payload is refused", envelope["ok"] is False, str(envelope))
        check("as a schema violation", envelope["error"]["code"] == "schema_violation", str(envelope))
        check("and the workspace was not touched",
              sandbox.reward() == 0.0, f"{sandbox.reward():.4f}")


def test_every_provider_declares_how_its_output_is_constrained() -> None:
    for name in provider_names():
        provider_class = get_provider_class(name)
        check(f"{name} declares an output mode",
              provider_class.output_mode in {"schema", "strict_tools"},
              provider_class.output_mode)
    # And it is recorded, so a run can say how its output was pinned rather
    # than leaving a reader to infer it from the provider name.
    check("and reports it", create_provider(environ={}).describe()["output_mode"] == "schema")


def main() -> int:
    print(__doc__)
    for test in (
        test_a_model_spec_resolves_to_the_provider_it_names,
        test_the_default_needs_nothing_and_the_others_say_what_they_need,
        test_config_defaults_survive_an_override_that_was_not_given,
        test_every_provider_decodes_the_same_turn_identically,
        test_each_provider_is_sent_its_own_dialect,
        test_anthropic_folds_a_turn_of_parallel_results_into_one_message,
        test_arguments_are_decoded_without_ever_raising,
        test_a_service_failure_is_a_provider_error_and_not_a_crash,
        test_the_loop_stops_for_the_right_reasons,
        test_a_refusal_reaches_the_model_intact,
        test_a_long_result_is_truncated_and_says_so,
        test_the_system_prompt_carries_the_role_and_the_rules,
        test_the_loop_actually_changes_the_world_it_is_graded_on,
        test_the_trajectory_is_valid_atif,
        test_the_action_schema_can_only_describe_real_tools,
        test_the_catalogue_says_what_the_schema_cannot,
        test_strict_mode_normalisation_keeps_the_same_tool,
        test_the_validator_never_refuses_what_the_world_would_accept,
        test_a_call_the_schema_does_not_describe_never_reaches_the_world,
        test_every_provider_declares_how_its_output_is_constrained,
    ):
        print(f"\n{test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"all agent-adapter checks passed ({len(provider_names())} providers registered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
