"""LLM provider adapters.

One class per service, all speaking the same three-word vocabulary --
:class:`Message`, :class:`ToolCall`, :class:`Completion` -- so ``agent.loop``
never learns which service answered it. Adding a provider is a subclass and a
decorator; no other module in this package changes.

Every call carries a schema, so the shape of what comes back is decided before
generation rather than discovered afterwards. The mechanism differs by service
and :mod:`agent.schemas` builds each one from the same tool definitions; see
``output_mode`` on :class:`Provider`.

Standard library only, for the same reason the rest of the task is: this has to
run wherever the task runs, without an install step. That rules out each
vendor's SDK, which is no loss -- every one of these APIs is a single JSON POST,
and the differences that matter are in how tools, tool results and output
constraints are spelled, which an SDK would hide rather than remove.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from agent.schemas import (
    ACTIONS_KEY,
    ANSWER_KEY,
    build_action_schema,
    drop_nulls,
    render_catalogue,
    strict_parameters,
)

DEFAULT_TIMEOUT_SEC = 600.0

# The neutral conversation record. A message is one of:
#   {"role": "user",      "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
#   {"role": "tool",      "tool_call_id": str, "name": str, "content": str}
# The system prompt is passed to `complete` separately, because two of the three
# providers here carry it outside the message list.
Message = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One requested action, already decoded into arguments.

    ``id`` is the provider's correlation handle. Ollama does not issue one, so
    the adapter mints it; the loop only requires that it is unique within a turn
    and that the tool result quotes it back.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Completion:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ProviderError(RuntimeError):
    """The model could not be consulted. Distinct from a model that answered
    badly: this one is about the transport, the credential, or the service."""


class ProviderUnavailable(ProviderError):
    """The endpoint could not be reached at all."""


class ProviderRejected(ProviderError):
    """The endpoint answered, and the answer was a refusal."""


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_sec: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise ProviderRejected(f"{url} answered HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProviderUnavailable(f"Could not reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ProviderUnavailable(f"{url} did not answer within {timeout_sec}s") from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderRejected(f"{url} returned content that is not JSON: {body[:500]}") from exc
    if not isinstance(decoded, dict):
        raise ProviderRejected(f"{url} returned {type(decoded).__name__}, expected an object.")
    return decoded


def _decode_arguments(raw: Any, tool_name: str) -> dict[str, Any]:
    """Normalize whatever the provider called arguments into a dict.

    Providers disagree: Anthropic and Ollama send an object, OpenAI sends a JSON
    string. A model can also send a malformed string, and that is a mistake the
    model should be told about rather than a crash -- so it becomes an empty
    payload here and a validation error from the world one step later.
    """
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return {"__unparsable_arguments__": text[:500]}
        if isinstance(decoded, Mapping):
            return dict(decoded)
        return {"__unparsable_arguments__": text[:500]}
    if raw is None:
        return {}
    return {"__unparsable_arguments__": repr(raw)[:500]}


class Provider(ABC):
    """A model that can be shown tools and asked for the next move.

    Every adapter constrains its own output with a schema; ``output_mode`` says
    which mechanism it uses, because the services do not agree on one. Neither
    the loop nor the caller has to care -- both modes answer with the same
    :class:`Completion`, and it is only recorded so a run can say how its output
    was pinned.
    """

    #: "schema" -- the whole message is constrained to a generated action
    #: schema, and the tool surface travels in the prompt. "strict_tools" --
    #: the tool definitions themselves carry the constraint.
    output_mode: ClassVar[str] = "strict_tools"
    # Registry key, and the head of a "provider/model" spec.
    name: ClassVar[str] = ""
    # Used when neither the caller nor the environment names one.
    default_model: ClassVar[str] = ""
    default_base_url: ClassVar[str] = ""
    # Environment variables consulted, in order, for the credential. Empty for
    # providers that need none -- which is how `create_provider` knows not to
    # demand one.
    api_key_env: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        model: str = "",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        max_output_tokens: int = 4096,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model or self.default_model
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.max_output_tokens = max_output_tokens
        self.options: dict[str, Any] = dict(options or {})

    @property
    def spec(self) -> str:
        return f"{self.name}/{self.model}"

    def describe(self) -> dict[str, Any]:
        """What was actually configured, for the trajectory and the run log.

        The credential is deliberately reported as a boolean.
        """
        return {
            "provider": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "authenticated": bool(self.api_key),
            "output_mode": self.output_mode,
        }

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]],
    ) -> Completion:
        """Answer one turn.

        ``tools`` arrive in the environment's own shape --
        ``{"name", "description", "input_schema"}`` -- and each adapter
        translates. Raises :class:`ProviderError` when the service could not be
        consulted; a model that answers with nothing useful is not an error
        here, it is a :class:`Completion` the loop has to deal with.
        """

    # -- helpers shared by the concrete adapters ---------------------------

    def _headers(self) -> dict[str, str]:
        return {}

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return _post_json(
            f"{self.base_url}{path}", payload, self._headers(), self.timeout_sec
        )


_PROVIDERS: dict[str, type[Provider]] = {}


def register_provider(cls: type[Provider]) -> type[Provider]:
    """Add a provider to the registry keyed by its ``name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must declare a name.")
    if cls.name in _PROVIDERS and _PROVIDERS[cls.name] is not cls:
        raise ValueError(f"Provider {cls.name!r} is already registered.")
    _PROVIDERS[cls.name] = cls
    return cls


def provider_names() -> list[str]:
    return sorted(_PROVIDERS)


def get_provider_class(name: str) -> type[Provider]:
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown provider {name!r}. Registered: {', '.join(provider_names())}."
        ) from None


# --------------------------------------------------------------------------
# Ollama -- the default, because it is the one that needs no account.
# --------------------------------------------------------------------------


@register_provider
class OllamaProvider(Provider):
    """Ollama's ``/api/chat``.

    The default provider: a local daemon, no credential, no per-token cost, and
    a tool-calling surface close enough to OpenAI's that the translation below
    is mostly renaming. Everything runs on the machine that starts the run,
    which is what makes it usable against a task whose environment has no
    network at all.
    """

    name = "ollama"
    default_model = "qwen3.6:35b"
    default_base_url = "http://127.0.0.1:11434"
    # `format` and `tools` are mutually exclusive here: send both and the reply
    # comes back schema-shaped with `tool_calls: null`, so native tool calling
    # stops working with no error to say so. Given the choice, take the schema
    # -- it is the half that is actually enforced.
    output_mode = "schema"

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]],
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _with_catalogue(system, tools)},
                *_encode_schema_mode(messages),
            ],
            "stream": False,
            # Reasoning is off by default: it costs minutes per turn on a local
            # box and this task is decided by which records the model reads
            # before it writes, not by depth of thought. Turn it back on with
            # options={"think": True} when comparing thinking models.
            "think": bool(self.options.get("think", False)),
            "options": {
                "temperature": self.temperature,
                # Ollama's default context is small and it truncates silently:
                # the instruction and a dozen tool results do not fit, and the
                # symptom is a model that "forgets" what it just read.
                "num_ctx": int(self.options.get("num_ctx", 32768)),
                **{
                    key: value
                    for key, value in self.options.items()
                    if key not in {"think", "num_ctx"}
                },
            },
        }
        if tools:
            payload["format"] = build_action_schema(tools)
        data = self._post("/api/chat", payload)
        message = data.get("message") or {}
        text, calls = _decode_action_object(str(message.get("content") or ""), tools)
        return Completion(
            text=text,
            tool_calls=calls,
            finish_reason=str(data.get("done_reason") or "") or None,
            prompt_tokens=_as_int(data.get("prompt_eval_count")),
            completion_tokens=_as_int(data.get("eval_count")),
        )


# --------------------------------------------------------------------------
# OpenAI-compatible /v1/chat/completions
# --------------------------------------------------------------------------


@register_provider
class OpenAIProvider(Provider):
    """``/v1/chat/completions``.

    Not only OpenAI: vLLM, LM Studio, Together, Groq and llama.cpp all serve
    this shape, so pointing ``base_url`` at one of those needs no new adapter.
    """

    name = "openai"
    default_model = "gpt-5"
    default_base_url = "https://api.openai.com/v1"
    api_key_env = ("OPENAI_API_KEY",)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]],
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *_encode_openai(messages)],
            "temperature": self.temperature,
            **self.options,
        }
        if tools:
            # `strict` is the switch that makes the arguments come back
            # constrained instead of merely likely. It is also the reason the
            # schemas are normalized: strict mode requires every declared
            # property to be required, so optional ones are made nullable and
            # the nulls are dropped again below.
            payload["tools"] = [_openai_tool(tool) for tool in tools]
            payload["tool_choice"] = "auto"
        data = self._post("/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise ProviderRejected(f"{self.spec} returned no choices: {json.dumps(data)[:500]}")
        choice = choices[0] or {}
        message = choice.get("message") or {}
        calls: list[ToolCall] = []
        for index, entry in enumerate(message.get("tool_calls") or []):
            function = (entry or {}).get("function") or {}
            name = str(function.get("name") or "")
            if not name:
                continue
            calls.append(
                ToolCall(
                    id=str(entry.get("id") or f"call-{index + 1}"),
                    name=name,
                    arguments=drop_nulls(_decode_arguments(function.get("arguments"), name)),
                )
            )
        usage = data.get("usage") or {}
        return Completion(
            text=str(message.get("content") or ""),
            tool_calls=tuple(calls),
            finish_reason=str(choice.get("finish_reason") or "") or None,
            prompt_tokens=_as_int(usage.get("prompt_tokens")),
            completion_tokens=_as_int(usage.get("completion_tokens")),
        )


@register_provider
class OpenRouterProvider(OpenAIProvider):
    """OpenRouter, which is OpenAI's wire format in front of everyone else's
    models. Registered separately so ``openrouter/anthropic/claude-opus-5``
    resolves without also having to name a base URL, and so this repository's
    own ``OPEN_ROUTER_KEY`` is found where it already lives."""

    name = "openrouter"
    default_model = "anthropic/claude-opus-5"
    default_base_url = "https://openrouter.ai/api/v1"
    api_key_env = ("OPENROUTER_API_KEY", "OPEN_ROUTER_KEY")


# --------------------------------------------------------------------------
# Anthropic /v1/messages
# --------------------------------------------------------------------------


@register_provider
class AnthropicProvider(Provider):
    """Anthropic's Messages API.

    The one genuinely different shape here: the system prompt is a top-level
    field, tool results are user-turn content blocks rather than a role of their
    own, and arguments arrive already decoded.
    """

    name = "anthropic"
    default_model = "claude-opus-5"
    default_base_url = "https://api.anthropic.com/v1"
    api_key_env = ("ANTHROPIC_API_KEY",)

    def _headers(self) -> dict[str, str]:
        headers = {"anthropic-version": "2023-06-01"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]],
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": system,
            "messages": _encode_anthropic(messages),
            "temperature": self.temperature,
            **self.options,
        }
        if tools:
            # The schema travels on every tool and is enforced server-side, so
            # arguments arrive decoded and shaped; there is no strict flag to
            # set because there is no unconstrained mode to opt out of.
            payload["tools"] = [
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("input_schema") or {"type": "object"},
                }
                for tool in tools
            ]
        data = self._post("/messages", payload)
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content") or []:
            kind = (block or {}).get("type")
            if kind == "text":
                texts.append(str(block.get("text") or ""))
            elif kind == "tool_use":
                name = str(block.get("name") or "")
                if not name:
                    continue
                calls.append(
                    ToolCall(
                        id=str(block.get("id") or f"toolu-{len(calls) + 1}"),
                        name=name,
                        arguments=_decode_arguments(block.get("input"), name),
                    )
                )
        usage = data.get("usage") or {}
        return Completion(
            text="\n".join(part for part in texts if part),
            tool_calls=tuple(calls),
            finish_reason=str(data.get("stop_reason") or "") or None,
            prompt_tokens=_as_int(usage.get("input_tokens")),
            completion_tokens=_as_int(usage.get("output_tokens")),
        )


# --------------------------------------------------------------------------
# Wire-format translation
# --------------------------------------------------------------------------


def _openai_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    """A function definition with its constraint switched on.

    Always strict: an endpoint that rejects the flag is an endpoint that was
    not going to honour the schema either, and that is worth failing loudly
    rather than discovering from a malformed argument six turns later.
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": strict_parameters(tool.get("input_schema")),
            "strict": True,
        },
    }


def _with_catalogue(system: str, tools: Sequence[Mapping[str, Any]]) -> str:
    """Schema mode constrains the shape; the catalogue supplies the meaning.

    Without it a model answers in perfect JSON naming a tool it invented the
    description of, because the schema carries names and types and nothing else.
    """
    return f"{system}\n\n{render_catalogue(tools)}" if tools else system


def _decode_action_object(
    content: str, tools: Sequence[Mapping[str, Any]]
) -> tuple[str, tuple[ToolCall, ...]]:
    """Read one constrained action object back into the neutral vocabulary.

    Actions become tool calls and an empty action list becomes a final answer,
    which is exactly what a native tool-calling turn looks like -- so the loop
    cannot tell the two modes apart.
    """
    text = content.strip()
    if not text:
        return "", ()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        # Constrained decoding should make this unreachable; if a service ever
        # lets it through, the text is still the model's answer and is worth
        # more than an exception.
        return text, ()
    if not isinstance(decoded, dict):
        return text, ()
    actions = decoded.get(ACTIONS_KEY) or []
    calls: list[ToolCall] = []
    if isinstance(actions, list):
        for index, action in enumerate(actions):
            if not isinstance(action, Mapping):
                continue
            name = str(action.get("tool") or "")
            if not name:
                continue
            calls.append(
                ToolCall(
                    id=f"action-{index + 1}",
                    name=name,
                    arguments=_decode_arguments(action.get("arguments"), name),
                )
            )
    answer = str(decoded.get(ANSWER_KEY) or "")
    if calls:
        # A model that both acts and answers has not finished; its answer would
        # be about work it has not done yet, so the reasoning is kept as the
        # turn's text and the answer is left for the turn that ends the episode.
        return str(decoded.get("thought") or ""), tuple(calls)
    return answer or str(decoded.get("thought") or ""), ()


def _encode_schema_mode(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Replay the history in the shape the model is constrained to produce.

    An assistant turn goes back as the action object it was, not as an
    OpenAI-shaped message with a `tool_calls` field the request never declared
    tools for. Otherwise every previous turn in the context contradicts the
    schema the current turn is being decoded against, which is a strange thing
    to ask a model to generalise from.
    """
    encoded: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "assistant":
            calls = message.get("tool_calls") or ()
            encoded.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "thought": message.get("content") or "",
                            ACTIONS_KEY: [
                                {"tool": call.name, "arguments": call.arguments}
                                for call in calls
                            ],
                            ANSWER_KEY: "" if calls else (message.get("content") or ""),
                        },
                        sort_keys=True,
                    ),
                }
            )
        elif role == "tool":
            encoded.append(
                {
                    "role": "tool",
                    "tool_name": message.get("name", ""),
                    "content": message.get("content") or "",
                }
            )
        else:
            encoded.append({"role": role, "content": message.get("content") or ""})
    return encoded


def _encode_openai(
    messages: Sequence[Message], *, arguments_as_json_string: bool = True
) -> list[dict[str, Any]]:
    """Encode the history for a chat-completions-shaped API.

    ``arguments_as_json_string`` is the one place the two dialects actually
    differ. OpenAI serialises a tool call's arguments to a string and expects
    that string back; Ollama expects the object, and rejects the string with
    "Value looks like object, but can't find closing '}' symbol" -- an error
    about the *replayed* call, one turn after the call it describes, which is
    why it is worth naming here.
    """
    encoded: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
            calls = message.get("tool_calls") or ()
            if calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": (
                                json.dumps(call.arguments, sort_keys=True)
                                if arguments_as_json_string
                                else call.arguments
                            ),
                        },
                    }
                    for call in calls
                ]
            encoded.append(entry)
        elif role == "tool":
            encoded.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    # Ollama matches results to calls by name rather than by id;
                    # OpenAI ignores the extra key. Sending both is what lets
                    # one encoder serve the two of them.
                    "name": message.get("name", ""),
                    "tool_name": message.get("name", ""),
                    "content": message.get("content") or "",
                }
            )
        else:
            encoded.append({"role": role, "content": message.get("content") or ""})
    return encoded


def _encode_anthropic(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Fold the neutral history into Anthropic's alternating user/assistant turns.

    Consecutive tool results belong to one user turn, so they are gathered
    rather than emitted one message each.
    """
    encoded: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush() -> None:
        if pending_results:
            encoded.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in messages:
        role = message["role"]
        if role == "tool":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": message.get("content") or "",
                }
            )
            continue
        flush()
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = message.get("content") or ""
            if text:
                blocks.append({"type": "text", "text": text})
            for call in message.get("tool_calls") or ():
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            # An assistant turn with neither text nor a call is not a legal
            # message; the loop only records one when the model said nothing,
            # and Anthropic would reject the empty block list.
            encoded.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no content)"}]})
        else:
            encoded.append({"role": "user", "content": message.get("content") or ""})
    flush()
    return encoded


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

DEFAULT_PROVIDER = OllamaProvider.name


def split_spec(spec: str, *, default_provider: str = DEFAULT_PROVIDER) -> tuple[str, str]:
    """Split ``"provider/model"`` into its two halves.

    Only the first segment is considered, and only when it names a registered
    provider -- so ``anthropic/claude-opus-5`` reaches the Anthropic adapter
    while ``openrouter/anthropic/claude-opus-5`` reaches OpenRouter with the
    rest as the model id, and a bare ``qwen3.6:35b`` stays a model name for the
    default provider rather than becoming a provider called ``qwen3.6:35b``.
    """
    spec = (spec or "").strip()
    if not spec:
        return default_provider, ""
    head, separator, tail = spec.partition("/")
    if separator and head in _PROVIDERS:
        return head, tail
    if not separator and spec in _PROVIDERS:
        return spec, ""
    return default_provider, spec


def resolve_api_key(
    provider_class: type[Provider], environ: Mapping[str, str] | None = None
) -> str | None:
    source = os.environ if environ is None else environ
    for name in provider_class.api_key_env:
        value = source.get(name)
        if value:
            return value
    return None


def create_provider(
    spec: str = "",
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    max_output_tokens: int = 4096,
    options: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Provider:
    """Build the adapter named by ``spec``.

    ``spec`` is ``"provider/model"``, or a bare model name for the default
    provider. The credential is looked up in the provider's own environment
    variables when one is not passed; a provider that declares none is never
    asked for one, which is why the default path needs no configuration at all.
    """
    provider_name, model = split_spec(spec)
    provider_class = get_provider_class(provider_name)
    if api_key is None:
        api_key = resolve_api_key(provider_class, environ)
    if provider_class.api_key_env and not api_key:
        raise ProviderError(
            f"Provider {provider_name!r} needs a credential. Set one of: "
            f"{', '.join(provider_class.api_key_env)}."
        )
    return provider_class(
        model or provider_class.default_model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout_sec=timeout_sec,
        max_output_tokens=max_output_tokens,
        options=options,
    )
