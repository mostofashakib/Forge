"""One place where an agent run is configured.

The whole of the switch between providers is the ``model`` field: a
``provider/model`` string that names a registered adapter, or a bare model name
for the default one. Everything else -- credential, endpoint, sampling -- has a
working default per provider, so the common case is a model name and nothing
else, and the default case is not even that.

Values come from three sources, later beating earlier: the class defaults, the
environment, and whatever the caller passes explicitly (the CLI's flags, or
Harbor's ``-m`` and ``--ak``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from agent.loop import DEFAULT_MAX_TOOL_OUTPUT_CHARS, DEFAULT_MAX_TURNS, DEFAULT_REPEAT_LIMIT
from agent.providers import DEFAULT_TIMEOUT_SEC, Provider, create_provider

# Ollama, and a model that is on the machine rather than behind an account. A
# run of this task should need nothing bought and nothing configured; every
# other provider here is an opt-in from that starting point.
DEFAULT_MODEL = "ollama/qwen3.6:35b"

ENV_PREFIX = "TASK_AGENT_"


def _env(name: str, environ: Mapping[str, str]) -> str | None:
    value = environ.get(f"{ENV_PREFIX}{name}")
    return value if value else None


def _env_float(name: str, environ: Mapping[str, str]) -> float | None:
    raw = _env(name, environ)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}.") from None


def _env_int(name: str, environ: Mapping[str, str]) -> int | None:
    raw = _env(name, environ)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}.") from None


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = DEFAULT_MODEL
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    max_output_tokens: int = 4096
    max_turns: int = DEFAULT_MAX_TURNS
    max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS
    repeat_limit: int = DEFAULT_REPEAT_LIMIT
    deadline_sec: float | None = None
    prompt_id: str = "incident_coordinator"
    think: bool = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, **overrides: Any) -> "AgentConfig":
        source = os.environ if environ is None else environ
        settings: dict[str, Any] = {}
        for field_name, reader in (
            ("model", _env),
            ("base_url", _env),
            ("api_key", _env),
            ("prompt_id", _env),
        ):
            value = reader(field_name.upper(), source)
            if value is not None:
                settings[field_name] = value
        for field_name in ("temperature", "timeout_sec", "deadline_sec"):
            value = _env_float(field_name.upper(), source)
            if value is not None:
                settings[field_name] = value
        for field_name in ("max_output_tokens", "max_turns", "max_tool_output_chars", "repeat_limit"):
            value = _env_int(field_name.upper(), source)
            if value is not None:
                settings[field_name] = value
        think = _env("THINK", source)
        if think is not None:
            settings["think"] = think.strip().lower() in {"1", "true", "yes", "on"}
        # An override that was not asked for is not an override: Harbor passes
        # model_name=None when no -m was given, and that must not erase the
        # default.
        settings.update({key: value for key, value in overrides.items() if value is not None})
        return replace(cls(), **settings)

    def build_provider(self, environ: Mapping[str, str] | None = None) -> Provider:
        return create_provider(
            self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            timeout_sec=self.timeout_sec,
            max_output_tokens=self.max_output_tokens,
            options={"think": self.think} if self.think else None,
            environ=environ,
        )
