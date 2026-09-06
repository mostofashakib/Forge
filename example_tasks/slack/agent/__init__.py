"""A small, provider-agnostic agent for driving the Slack workspace.

Two independent adapter layers meet in :class:`agent.loop.ToolLoop`:

* :mod:`agent.providers` -- which model answers. Ollama by default, because it
  needs no account and no key; OpenAI-compatible endpoints, OpenRouter and
  Anthropic ship alongside it, and a new one is a subclass plus a decorator.
  Every call carries a schema generated from the tool definitions, so what comes
  back is constrained at decode time rather than parsed and hoped over.
* :mod:`agent.backends` -- where its tool calls run. A Harbor-managed container,
  the ``slack`` CLI over a socket, or the world in this process.

Neither layer knows about the other, and the loop knows about neither's
contents: it never names a provider and never names a tool.

Standard library only. Harbor is imported by :mod:`agent.harbor_agent` alone,
which is the only module that needs it.
"""

from agent.backends import (
    BackendError,
    HarborBackend,
    InProcessBackend,
    SubprocessBackend,
    ToolBackend,
)
from agent.config import DEFAULT_MODEL, AgentConfig
from agent.loop import RunResult, ToolLoop, Turn
from agent.prompts import system_prompt
from agent.schemas import (
    SchemaViolation,
    build_action_schema,
    render_catalogue,
    strict_parameters,
    validate,
)
from agent.providers import (
    Completion,
    Provider,
    ProviderError,
    ProviderRejected,
    ProviderUnavailable,
    ToolCall,
    create_provider,
    provider_names,
    register_provider,
)
from agent.trajectory import build_trajectory, write_trajectory

__all__ = [
    "DEFAULT_MODEL", "AgentConfig", "BackendError", "Completion", "HarborBackend",
    "InProcessBackend", "Provider", "ProviderError", "ProviderRejected",
    "ProviderUnavailable", "RunResult", "SchemaViolation", "SubprocessBackend",
    "ToolBackend", "ToolCall", "ToolLoop", "Turn", "build_action_schema",
    "build_trajectory", "create_provider", "provider_names", "register_provider",
    "render_catalogue", "strict_parameters", "system_prompt", "validate",
    "write_trajectory",
]
