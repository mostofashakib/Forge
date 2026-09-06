"""Harbor entry point.

    harbor run -p example_tasks/slack -a agent.harbor_agent:WorkspaceAgent \
               -m ollama/qwen3.6:35b

The only module in this package that imports Harbor, and the only one that has
to: everything it does is translate between Harbor's agent protocol and the
plain loop in :mod:`agent.loop`.

The loop runs on the host and the tool calls run in the container. That is not a
compromise -- it is what lets a model on this machine drive a task whose
environment has no network at all, and it keeps the boundary the environment
cares about intact: the agent still reaches the workspace only through the
``slack`` client, over ``agent.sock``, as the ``agent`` user Harbor set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from agent.backends import HarborBackend
from agent.config import AgentConfig
from agent.loop import ToolLoop
from agent.prompts import system_prompt
from agent.trajectory import build_trajectory, write_trajectory

AGENT_NAME = "workspace-adapter"
AGENT_VERSION = "1.0.0"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class WorkspaceAgent(BaseAgent):
    """Drive the workspace with whichever provider ``-m`` names.

    ``-m`` is the whole of the switch: ``ollama/qwen3.6:35b`` (the default),
    ``openrouter/anthropic/claude-opus-5``, ``anthropic/claude-opus-5``,
    ``openai/gpt-5``, or a bare model name for Ollama. Anything else is an
    ``--ak key=value``; the accepted keys are the fields of
    :class:`agent.config.AgentConfig`.
    """

    SUPPORTS_ATIF = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *args: Any,
        agent_timeout_sec: float | None = None,
        **kwargs: Any,
    ) -> None:
        # Harbor hands every --ak through as a string, and hands this
        # constructor a good deal it does not own. Take only the keys
        # AgentConfig declares and leave the rest to BaseAgent, so an unknown
        # --ak fails here with the name in it rather than deep in a request.
        config_fields = {field.name for field in AgentConfig.__dataclass_fields__.values()}
        overrides = {key: kwargs.pop(key) for key in list(kwargs) if key in config_fields}
        super().__init__(logs_dir=logs_dir, model_name=model_name, *args, **kwargs)
        self._agent_timeout_sec = agent_timeout_sec
        self._config = self._build_config(model_name, overrides)
        self._result: Any = None

    def _build_config(self, model_name: str | None, overrides: dict[str, Any]) -> AgentConfig:
        typed: dict[str, Any] = {}
        for key, value in overrides.items():
            field = AgentConfig.__dataclass_fields__[key]
            annotation = str(field.type)
            if "bool" in annotation:
                typed[key] = _as_bool(value)
            elif "int" in annotation:
                typed[key] = int(value)
            elif "float" in annotation:
                typed[key] = float(value)
            else:
                typed[key] = value
        # A run that is nearly out of its Harbor budget should stop itself and
        # leave a trajectory, rather than be killed mid-turn with nothing
        # written down.
        if self._agent_timeout_sec and "deadline_sec" not in typed:
            typed["deadline_sec"] = max(30.0, float(self._agent_timeout_sec) - 30.0)
        return AgentConfig.from_env(model=model_name, **typed)

    @staticmethod
    @override
    def name() -> str:
        return AGENT_NAME

    @override
    def version(self) -> str:
        return AGENT_VERSION

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install.

        The client the agent uses is already in the image, and the credential --
        when the chosen provider needs one at all -- stays on the host, where
        the model call is made. The container is not given a key it has no use
        for.
        """
        return None

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        provider = self._config.build_provider()
        backend = HarborBackend(environment)
        self.logger.info(
            "workspace-adapter: %s via %s", provider.spec, backend.describe()["backend"]
        )
        loop = ToolLoop(
            provider,
            backend,
            system=system_prompt(self._config.prompt_id),
            max_turns=self._config.max_turns,
            max_tool_output_chars=self._config.max_tool_output_chars,
            repeat_limit=self._config.repeat_limit,
            deadline_sec=self._config.deadline_sec,
            logger=self.logger,
        )
        try:
            tools = await backend.list_tools()
        except Exception:  # noqa: BLE001 - recorded below via the loop's own path
            tools = []
        result = await loop.run(instruction)
        self._result = result

        context.n_input_tokens = result.prompt_tokens or None
        context.n_output_tokens = result.completion_tokens or None
        context.metadata = {
            **provider.describe(),
            **backend.describe(),
            **result.summary(),
        }

        write_trajectory(
            self.logs_dir / "trajectory.json",
            build_trajectory(
                result,
                agent_name=AGENT_NAME,
                agent_version=AGENT_VERSION,
                model_name=provider.spec,
                tool_definitions=tools or None,
                session_id=self.session_id,
                extra=provider.describe(),
            ),
        )
        self.logger.info("workspace-adapter finished: %s", result.summary())
