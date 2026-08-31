"""Model I/O for external agents: adapters and generation-health monitoring."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

# The single place the default model and host are defined. Every agent and
# CLI imports these rather than repeating the strings, so switching the fleet
# to another model — including hosted providers via litellm prefixes such as
# "openai/..." or "anthropic/..." — is one value, or one env var:
#   FLEET_MODEL=anthropic/claude-sonnet-4-5 ./run.sh smoke
DEFAULT_MODEL = os.environ.get("FLEET_MODEL", "gemma4:26b")
DEFAULT_OLLAMA_HOST = os.environ.get("FLEET_OLLAMA_HOST", "http://127.0.0.1:11434")


class ModelGenerationTimeout(RuntimeError):
    """The token stream stalled or the generation exceeded its wall-clock
    budget. Time-based evidence only: the model may simply need longer than
    the configured budget, so this does not by itself mean the model is stuck."""


class DegenerateGenerationError(RuntimeError):
    """The model is emitting a repeating token cycle (greedy-decoding
    repetition trap) and will not terminate on its own. Content-based
    evidence: the model is provably stuck."""


def looks_degenerate(tail: str, window: int = 240, max_period: int = 60, min_repeats: int = 4) -> bool:
    """True when the last `window` chars are one short substring repeated
    end-to-end. Valid actions are compact, varied JSON; hundreds of chars of
    an exact repeating cycle only occur in a runaway decode."""
    if len(tail) < window:
        return False
    tail = tail[-window:]
    for period in range(1, max_period + 1):
        repeats = window // period
        if repeats < min_repeats:
            break
        unit = tail[-period:]
        if unit * repeats == tail[-period * repeats:]:
            return True
    return False


class StreamHealthMonitor:
    """Watches one generation stream and raises when it is provably looping
    (degenerate repetition) or out of time budget.

    Thinking tokens are observed for health monitoring only; they are never
    part of the answer text, so they can never reach the prompt or tool
    history.
    """

    tail_window = 240

    def __init__(self, model_name: str, max_generation_sec: float) -> None:
        self.model_name = model_name
        self.max_generation_sec = max_generation_sec
        self.deadline = time.monotonic() + max_generation_sec
        self.answer_tail = ""
        self.answer_chunks = 0
        self.thinking_tail = ""
        self.thinking_chars = 0

    def observe_answer(self, part: str) -> None:
        self.answer_chunks += 1
        self.answer_tail = (self.answer_tail + part)[-self.tail_window:]
        if looks_degenerate(self.answer_tail):
            raise DegenerateGenerationError(
                f"degenerate_repetition: {self.model_name} is looping on {self.answer_tail[-40:]!r}"
            )

    def observe_thinking(self, part: str) -> None:
        self.thinking_chars += len(part)
        self.thinking_tail = (self.thinking_tail + part)[-self.tail_window:]
        if looks_degenerate(self.thinking_tail):
            raise DegenerateGenerationError(
                f"degenerate_repetition: {self.model_name} is looping in its thinking "
                f"on {self.thinking_tail[-40:]!r}"
            )

    def check_budget(self) -> None:
        if time.monotonic() > self.deadline:
            raise ModelGenerationTimeout(
                f"generation_budget_exceeded: {self.model_name} produced no completed action "
                f"within {self.max_generation_sec}s "
                f"({self.answer_chunks} answer chunks, {self.thinking_chars} thinking chars so far)"
            )


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, model_name: str, format_schema: dict[str, Any] | None = None) -> str:
        """Generate a response text from the model, optionally constrained to a JSON schema."""
        pass


class OllamaAdapter(ModelAdapter):
    # Streaming lets us tell three situations apart instead of one blanket
    # timeout: a thinking model keeps emitting varied tokens (allowed, up to
    # max_generation_sec), a degenerate decode emits a repeating cycle
    # (detected by content), and a wedged server emits nothing (stall).
    stream_stall_timeout_sec = 120
    max_generation_sec = 480

    def __init__(
        self,
        ollama_host: str,
        stream_stall_timeout_sec: float | None = None,
        max_generation_sec: float | None = None,
    ) -> None:
        self.ollama_host = ollama_host
        if stream_stall_timeout_sec is not None:
            self.stream_stall_timeout_sec = stream_stall_timeout_sec
        if max_generation_sec is not None:
            self.max_generation_sec = max_generation_sec

    def generate(self, prompt: str, model_name: str, format_schema: dict[str, Any] | None = None) -> str:
        request = self._build_request(prompt, model_name, format_schema)
        try:
            return self._consume_stream(request, model_name)
        except TimeoutError as exc:
            raise ModelGenerationTimeout(
                f"stalled_stream: no tokens from {model_name} for {self.stream_stall_timeout_sec}s"
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ModelGenerationTimeout(
                    f"stalled_stream: no tokens from {model_name} for {self.stream_stall_timeout_sec}s"
                ) from exc
            raise RuntimeError(f"Could not reach Ollama at {self.ollama_host}: {exc}") from exc

    def _build_request(self, prompt: str, model_name: str, format_schema: dict[str, Any] | None) -> urllib.request.Request:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            # A schema constrains decoding at the grammar level (e.g. the
            # action "type" enum), so the model cannot emit free-form values
            # that greedy decoding could spiral on.
            "format": format_schema or "json",
            # Thinking-capable models default to think=true on /api/generate,
            # and Ollama's structured outputs are incompatible with that mode:
            # the constrained answer comes back garbled and truncated
            # mid-string (verified against Ollama 0.30.7 with gemma4:26b),
            # which the agent would then misclassify as a stuck model. The
            # action's "thought" field is the model's reasoning channel.
            "think": False,
            # Ollama defaults to a small context window and silently truncates
            # the prompt beyond it; raise it so the instruction and history
            # always fit.
            "options": {"temperature": 0, "num_ctx": 32768},
        }
        return urllib.request.Request(
            f"{self.ollama_host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _consume_stream(self, request: urllib.request.Request, model_name: str) -> str:
        monitor = StreamHealthMonitor(model_name, self.max_generation_sec)
        parts: list[str] = []
        with urllib.request.urlopen(request, timeout=self.stream_stall_timeout_sec) as response:
            for line in response:
                if not line.strip():
                    continue
                chunk = json.loads(line.decode("utf-8"))
                part = str(chunk.get("response", ""))
                if part:
                    parts.append(part)
                    monitor.observe_answer(part)
                thinking_part = str(chunk.get("thinking", ""))
                if thinking_part:
                    monitor.observe_thinking(thinking_part)
                if chunk.get("done"):
                    break
                monitor.check_budget()
        return "".join(parts)


class LiteLLMAdapter(ModelAdapter):
    def generate(self, prompt: str, model_name: str, format_schema: dict[str, Any] | None = None) -> str:
        import litellm

        try:
            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            # Callers handle adapter failures as RuntimeError regardless of
            # provider; litellm's exception hierarchy must not leak through.
            raise RuntimeError(f"litellm generation failed for {model_name}: {exc}") from exc
        return str(response.choices[0].message.content)


def create_adapter(
    model: str,
    ollama_host: str,
    stream_stall_timeout_sec: float | None = None,
    max_generation_sec: float | None = None,
) -> ModelAdapter:
    external_prefixes = ("openai/", "anthropic/", "openrouter/", "gemini/", "cohere/", "groq/", "vertex_ai/", "azure/", "deepseek/")
    is_external = (
        model.startswith(external_prefixes)
        or any(provider in model.lower() for provider in ["gpt-", "claude-", "gemini-", "deepseek-", "openrouter"])
    )
    if is_external:
        return LiteLLMAdapter()
    return OllamaAdapter(ollama_host, stream_stall_timeout_sec, max_generation_sec)
