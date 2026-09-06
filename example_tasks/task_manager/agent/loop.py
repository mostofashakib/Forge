"""The agent loop: one model, one tool surface, and the turns between them.

Deliberately free of both halves it joins. It never names a provider and never
names a tool, so swapping ``ollama`` for ``anthropic`` or the container backend
for an in-process one changes nothing here. What it does own is the part that
belongs to neither: how a turn is recorded, how a tool result is shown back to
the model, and the conditions under which a rollout stops.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.backends import BackendError, ToolBackend
from agent.providers import Completion, Provider, ProviderError, ToolCall
from agent.schemas import SchemaViolation, validate

DEFAULT_MAX_TURNS = 40
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 6000
# A local model that has decided on an answer will repeat it forever rather than
# stop. Three identical calls in a row is past coincidence -- a list call twice
# is plausible, the same arguments a third time is a stuck decode.
DEFAULT_REPEAT_LIMIT = 3


@dataclass(slots=True)
class Turn:
    """One model answer and everything the world said back to it."""

    index: int
    text: str
    tool_calls: tuple[ToolCall, ...]
    results: tuple[dict[str, Any], ...] = ()
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    started_at: float = 0.0
    duration_sec: float = 0.0


@dataclass(slots=True)
class RunResult:
    instruction: str
    system: str
    turns: list[Turn] = field(default_factory=list)
    final_text: str = ""
    # Why the loop stopped: "answered", "max_turns", "repeating",
    # "provider_error", "backend_error", or "deadline".
    stop_reason: str = "answered"
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_sec: float = 0.0

    @property
    def tool_call_count(self) -> int:
        return sum(len(turn.tool_calls) for turn in self.turns)

    def summary(self) -> dict[str, Any]:
        return {
            "turns": len(self.turns),
            "tool_calls": self.tool_call_count,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "duration_sec": round(self.duration_sec, 3),
        }


class ToolLoop:
    """Run one episode: ask, act, show the result, repeat."""

    def __init__(
        self,
        provider: Provider,
        backend: ToolBackend,
        *,
        system: str,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS,
        repeat_limit: int = DEFAULT_REPEAT_LIMIT,
        deadline_sec: float | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.provider = provider
        self.backend = backend
        self.system = system
        self.max_turns = max_turns
        self.max_tool_output_chars = max_tool_output_chars
        self.repeat_limit = repeat_limit
        self.deadline_sec = deadline_sec
        self.logger = logger or logging.getLogger(__name__)

    async def run(self, instruction: str) -> RunResult:
        started = time.monotonic()
        result = RunResult(instruction=instruction, system=self.system)
        try:
            tools = await self.backend.list_tools()
        except BackendError as exc:
            result.stop_reason = "backend_error"
            result.error = str(exc)
            result.duration_sec = time.monotonic() - started
            return result

        # Kept by name so a call can be checked before it is dispatched. Every
        # provider is asked to constrain its output with these already; this is
        # the assertion that it did, and the reason a malformed call becomes a
        # schema error the model can read rather than a round trip to the world
        # or, worse, a plausible-looking call the world happens to accept.
        schemas = {
            str(tool.get("name")): tool.get("input_schema") or {} for tool in tools
        }
        messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
        signatures: list[str] = []

        for index in range(1, self.max_turns + 1):
            if self.deadline_sec is not None and time.monotonic() - started > self.deadline_sec:
                result.stop_reason = "deadline"
                break

            turn_started = time.monotonic()
            try:
                # urllib blocks, and the backend may be an async container
                # call; keeping the provider off the event loop lets the two
                # coexist without each adapter having to be written twice.
                completion: Completion = await asyncio.to_thread(
                    self.provider.complete,
                    system=self.system,
                    messages=messages,
                    tools=tools,
                )
            except ProviderError as exc:
                result.stop_reason = "provider_error"
                result.error = str(exc)
                break

            turn = Turn(
                index=index,
                text=completion.text,
                tool_calls=completion.tool_calls,
                finish_reason=completion.finish_reason,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                started_at=turn_started,
                duration_sec=time.monotonic() - turn_started,
            )
            result.turns.append(turn)
            result.prompt_tokens += completion.prompt_tokens or 0
            result.completion_tokens += completion.completion_tokens or 0
            messages.append(
                {
                    "role": "assistant",
                    "content": completion.text,
                    "tool_calls": completion.tool_calls,
                }
            )

            if not completion.tool_calls:
                # No action requested: the model is answering, and the episode
                # is over. Its text is the answer of record only for the
                # transcript -- what the task is graded on is what it wrote into
                # the workspace.
                result.final_text = completion.text
                result.stop_reason = "answered"
                break

            results: list[dict[str, Any]] = []
            for call in completion.tool_calls:
                self.logger.debug("tool %s %s", call.name, call.arguments)
                rejection = _schema_error(call, schemas)
                if rejection is not None:
                    results.append(rejection)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": self._render(rejection),
                        }
                    )
                    continue
                try:
                    envelope = await self.backend.call(call.name, call.arguments)
                except BackendError as exc:
                    result.stop_reason = "backend_error"
                    result.error = str(exc)
                    envelope = {
                        "ok": False,
                        "error": {
                            "code": "workspace_unavailable",
                            "type": "unavailable",
                            "message": str(exc),
                        },
                    }
                results.append(envelope)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": self._render(envelope),
                    }
                )
            turn.results = tuple(results)
            if result.stop_reason == "backend_error":
                break

            signatures.append(_signature(completion.tool_calls))
            if _repeating(signatures, self.repeat_limit):
                # Stop rather than nudge: a model that has issued the same call
                # with the same arguments this many times is not going to be
                # argued out of it, and every further turn is spend without
                # information.
                result.stop_reason = "repeating"
                result.error = (
                    f"The same tool call was repeated {self.repeat_limit} times: "
                    f"{signatures[-1]}"
                )
                break
        else:
            result.stop_reason = "max_turns"

        if not result.final_text and result.turns:
            result.final_text = result.turns[-1].text
        result.duration_sec = time.monotonic() - started
        return result

    def _render(self, envelope: Mapping[str, Any]) -> str:
        """Turn the world's envelope into what the model reads.

        Refusals are passed through whole. They are the most informative thing
        the world says -- `label_unknown`, `not_found`, a status-machine
        rejection -- and a loop that flattened them to "error" would be hiding
        the feedback the next turn depends on.
        """
        text = json.dumps(envelope, sort_keys=True, indent=2, default=str)
        if len(text) <= self.max_tool_output_chars:
            return text
        keep = self.max_tool_output_chars
        return (
            f"{text[:keep]}\n... [truncated {len(text) - keep} characters; "
            f"narrow the query with a filter to see the rest]"
        )


def _schema_error(
    call: ToolCall, schemas: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    """The refusal for a call the schema does not describe, or None.

    Shaped like the world's own refusals, so a model reads it the same way it
    reads every other one and does not have to learn a second error vocabulary
    for the client that is standing in front of the world.
    """
    if call.name not in schemas:
        known = ", ".join(sorted(schemas)) or "none"
        return _refusal("unknown_tool", f"Unknown tool: {call.name}. Available: {known}")
    try:
        validate(call.arguments, schemas[call.name])
    except SchemaViolation as exc:
        return _refusal("schema_violation", f"{call.name}: {exc}")
    return None


def _refusal(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "type": "validation_error", "message": message}}


def _signature(calls: Sequence[ToolCall]) -> str:
    return json.dumps(
        [[call.name, call.arguments] for call in calls], sort_keys=True, default=str
    )


def _repeating(signatures: Sequence[str], limit: int) -> bool:
    if limit <= 0 or len(signatures) < limit:
        return False
    tail = signatures[-limit:]
    return all(entry == tail[0] for entry in tail)
