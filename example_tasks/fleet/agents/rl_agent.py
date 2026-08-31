"""External agents for the Slack and Task Manager services.
These agents run outside the task environment, call Ollama locally, and invoke
service tools through the Harbor environment process.

Module layout: model I/O lives in fleet.agents.model_adapters, run recording in
fleet.agents.run_recorder; this module holds the agent orchestration only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from fleet.agents.model_adapters import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DegenerateGenerationError,
    LiteLLMAdapter,
    ModelAdapter,
    ModelGenerationTimeout,
    OllamaAdapter,
    create_adapter,
    looks_degenerate,
)
from fleet.agents.run_recorder import RunRecorder, timestamp
from fleet.agents.task_profiles import (
    SLACK_DEFAULT_GUIDANCE,
    SLACK_TASK_PROFILES,
    TASK_MANAGER_DEFAULT_GUIDANCE,
    find_profile,
)
from fleet.environments.slack.schema import SLACK_TOOL_SCHEMA, ToolSchema
from fleet.environments.task_manager.schema import TASK_MANAGER_TOOL_SCHEMA

__all__ = [
    "RLAgent",
    "RunOutcome",
    "SlackExternalAgent",
    "TaskManagerExternalAgent",
    "ModelAdapter",
    "OllamaAdapter",
    "LiteLLMAdapter",
    "ModelGenerationTimeout",
    "DegenerateGenerationError",
    "looks_degenerate",
    "timestamp",
]


@dataclass
class RunOutcome:
    """How the action loop ended. At most one of the reasons is set; both are
    None for a normal finish (final answer or step budget exhausted)."""

    final_answer: str = ""
    stuck_reason: str | None = None
    timeout_reason: str | None = None


class RLAgent(BaseAgent, ABC):
    """Base external Harbor agent that drives service tools with JSON actions."""

    SUPPORTS_ATIF = True
    default_model = DEFAULT_MODEL
    default_ollama_host = DEFAULT_OLLAMA_HOST
    max_steps = 40
    required_tools_before_final: frozenset[str] = frozenset()

    def get_required_tools(self, instruction: str) -> frozenset[str]:
        return self.required_tools_before_final

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        ollama_host: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.ollama_host = (ollama_host or self.default_ollama_host).rstrip("/")
        self.adapter = create_adapter(self.model_name or self.default_model, self.ollama_host)

    def version(self) -> str | None:
        return "0.1.0"

    def _resolve_paths(self) -> tuple[Path, Path, Path]:
        root_dir = self.logs_dir
        while root_dir.parent != root_dir:
            if (root_dir / "slack_task_1").is_dir():
                break
            root_dir = root_dir.parent
        python_exe = root_dir / ".venv" / "bin" / "python"
        verify_script = root_dir / "tests" / "verify_task.py"
        return root_dir, python_exe, verify_script

    def get_verifier_spec(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        await self._reset_to_seed(environment, "setup reset")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # Complete fresh start from the initial seed state during every run.
        await self._reset_to_seed(environment, "run initial reset")
        workspace_root, python_exe, verify_script = self._resolve_paths()
        verifier_spec = self.get_verifier_spec()
        recorder = RunRecorder(
            self.logs_dir,
            instruction,
            self.model_name or self.default_model,
            verifier_spec=verifier_spec,
            python_exe=python_exe,
            verify_script=verify_script,
            workspace_root=workspace_root,
        )
        initial_state = await self.read_state(environment)
        recorder.record_state("Initial", initial_state)
        reset_determinism_check = await self._reset_determinism_check(environment, initial_state)

        tool_history: list[dict[str, Any]] = []
        outcome = await self._run_action_loop(instruction, environment, initial_state, tool_history, recorder)
        recorder.flush()

        final_state = await self.read_state(environment)
        recorder.record_state("Final", final_state)
        trajectory_path, transcript_path = recorder.write_trajectory(
            service_name=self.service_name(),
            agent_name=self.name(),
            agent_version=self.version(),
            extra={
                "service": self.service_name(),
                "tool_schema": self.tool_schema(),
                "initial_state_snapshot": initial_state,
                "final_state_snapshot": final_state,
                "final_answer": outcome.final_answer,
                "reset_determinism_check": reset_determinism_check,
                "model_stuck": outcome.stuck_reason,
                "generation_timeout": outcome.timeout_reason,
            },
        )
        await self.maybe_upload_logs(environment, trajectory_path, transcript_path)
        context.metadata = {
            "service": self.service_name(),
            "model": self.model_name or self.default_model,
            "ollama_host": self.ollama_host,
            "trajectory": str(trajectory_path),
            "transcript": str(transcript_path),
            "tool_calls": [item["tool_name"] for item in tool_history if item["tool_name"] != "agent.feedback"],
            "final_answer": outcome.final_answer,
            "model_stuck": outcome.stuck_reason,
            "generation_timeout": outcome.timeout_reason,
        }

    async def _reset_to_seed(self, environment: BaseEnvironment, label: str) -> None:
        reset_command = self.reset_command()
        if not reset_command:
            return
        result = await environment.exec(reset_command, cwd="/app", user="root", timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(f"{self.name()} {label} failed: {result.stderr or result.stdout}")
        await environment.exec("chown -R agent:agent /app", cwd="/app", user="root", timeout_sec=10)

    async def _reset_determinism_check(
        self, environment: BaseEnvironment, initial_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Reset again and require the state to reproduce byte-for-byte."""
        initial_state_json = json.dumps(initial_state, sort_keys=True)
        initial_hash = hashlib.sha256(initial_state_json.encode("utf-8")).hexdigest()

        await self._reset_to_seed(environment, "reset determinism check")
        reset_state = await self.read_state(environment)
        reset_state_json = json.dumps(reset_state, sort_keys=True)
        reset_hash = hashlib.sha256(reset_state_json.encode("utf-8")).hexdigest()

        passed = initial_state_json == reset_state_json
        if not passed:
            raise RuntimeError(
                f"Reset determinism check failed at agent run start: initial_hash={initial_hash}, reset_hash={reset_hash}"
            )
        return {"passed": passed, "initial_hash": initial_hash, "reset_hash": reset_hash}

    async def _run_action_loop(
        self,
        instruction: str,
        environment: BaseEnvironment,
        initial_state: dict[str, Any],
        tool_history: list[dict[str, Any]],
        recorder: RunRecorder,
    ) -> RunOutcome:
        outcome = RunOutcome()
        for step_index in range(self.max_steps):
            recorder.flush()
            action = await asyncio.to_thread(
                self.next_action,
                instruction,
                initial_state,
                tool_history,
            )
            if action["type"] == "model_stuck":
                outcome.stuck_reason = action["error"]
                break
            if action["type"] == "generation_timeout":
                outcome.timeout_reason = action["error"]
                break
            if action["type"] == "invalid_action":
                # The decoder is schema-constrained, so malformed output means
                # either model failure or a constrained-decoding bug in the
                # serving stack; neither is recoverable by retrying the same
                # request, so end early and keep the raw output for diagnosis.
                outcome.stuck_reason = f"invalid_action: {action['error']}; raw={action.get('raw', '')!r}"
                break
            if action["type"] == "final_answer":
                if self._request_required_tools(instruction, tool_history):
                    continue
                outcome.final_answer = str(action.get("answer", "")).strip()
                recorder.record_thinking(action.get("thought", ""))
                recorder.record_final_answer(outcome.final_answer)
                break
            await self._execute_tool_step(step_index, action, environment, tool_history, recorder)

        if outcome.stuck_reason is not None:
            recorder.record_note(f"[AGENT] Model stuck: {outcome.stuck_reason} -- ending run early")
        if outcome.timeout_reason is not None:
            recorder.record_note(
                f"[AGENT] Generation timeout: {outcome.timeout_reason} -- ending run early "
                "(time budget exhausted; may be configuration, not model failure)"
            )
        return outcome

    def _request_required_tools(self, instruction: str, tool_history: list[dict[str, Any]]) -> bool:
        """If required read tools were skipped, push feedback and report True."""
        missing_tools = sorted(
            self.get_required_tools(instruction) - {item["tool_name"] for item in tool_history}
        )
        if not missing_tools:
            return False
        tool_history.append(
            {
                "tool_name": "agent.feedback",
                "input": {"missing_required_tools": missing_tools},
                "observation": {
                    "error": "missing_required_tools",
                    "message": f"Call these tools before final_answer: {missing_tools}",
                },
            }
        )
        return True

    async def _execute_tool_step(
        self,
        step_index: int,
        action: dict[str, Any],
        environment: BaseEnvironment,
        tool_history: list[dict[str, Any]],
        recorder: RunRecorder,
    ) -> None:
        tool_name = str(action["tool_name"])
        tool_input = dict(action.get("input", {}))
        thought = action.get("thought", "")
        observation = await self.invoke_tool(environment, tool_name, tool_input)
        post_state = await self.read_state(environment)
        tool_history.append({"tool_name": tool_name, "input": tool_input, "observation": observation})
        recorder.record_thinking(thought)
        recorder.record_tool_step(
            step_index + 1,
            f"call-{len(tool_history)}",
            tool_name,
            tool_input,
            observation,
            post_state,
        )

    def next_action(
        self,
        instruction: str,
        state: dict[str, Any],
        tool_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = self.build_prompt(instruction, state, tool_history)
        model = self.model_name or self.default_model
        try:
            raw_response = self.adapter.generate(prompt, model, format_schema=self.action_format_schema())
        except DegenerateGenerationError as exc:
            # Content-based evidence: the model is provably looping.
            return {"type": "model_stuck", "error": str(exc)}
        except ModelGenerationTimeout as exc:
            # Time-based only: the model may simply need longer than the
            # configured budget, so this is reported as a timeout, not as a
            # stuck model.
            return {"type": "generation_timeout", "error": str(exc)}
        except Exception as exc:
            raise RuntimeError(f"Error during model generation with {model}: {exc}") from exc

        raw_response = raw_response.strip()
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            return {"type": "invalid_action", "error": f"not valid JSON: {exc}", "raw": raw_response[:500]}
        action = self.validate_action(parsed)
        if action is None:
            return {
                "type": "invalid_action",
                "error": "action does not match the tool_call or final_answer shape",
                "raw": raw_response[:500],
            }
        return action

    def validate_action(self, action: Any) -> dict[str, Any] | None:
        """Accept only the two action shapes the protocol defines."""
        if not isinstance(action, dict):
            return None
        if action.get("type") == "tool_call":
            tool_name = action.get("tool_name")
            tool_input = action.get("input")
            known_tools = {tool["name"] for tool in self.tool_schema()}
            if isinstance(tool_name, str) and tool_name in known_tools and isinstance(tool_input, dict):
                return {"type": "tool_call", "tool_name": tool_name, "input": tool_input, "thought": action.get("thought", "")}
            return None
        if action.get("type") == "final_answer" and isinstance(action.get("answer"), str):
            return {"type": "final_answer", "answer": action["answer"], "thought": action.get("thought", "")}
        return None

    def action_format_schema(self) -> dict[str, Any]:
        """JSON schema enforced by the model's constrained decoder. The enums
        make degenerate values (made-up action types or tool names) impossible
        to emit, rather than merely invalid after the fact."""
        tool_names = sorted(tool["name"] for tool in self.tool_schema())
        return {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "type": {"enum": ["tool_call"]},
                        "thought": {"type": "string"},
                        "tool_name": {"enum": tool_names},
                        "input": {"type": "object"},
                    },
                    "required": ["type", "thought", "tool_name", "input"],
                },
                {
                    "type": "object",
                    "properties": {
                        "type": {"enum": ["final_answer"]},
                        "thought": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                    "required": ["type", "thought", "answer"],
                },
            ]
        }

    def build_prompt(self, instruction: str, state: dict[str, Any], tool_history: list[dict[str, Any]]) -> str:
        return "\n".join(
            [
                "You are a Harbor external agent operating a deterministic service environment.",
                "Return exactly one JSON object. Do not include markdown or commentary.",
                'For a tool call: {"type":"tool_call","thought":"a detailed explanation of your plan and reasoning","tool_name":"...","input":{...}}',
                'For the final answer: {"type":"final_answer","thought":"a final review of execution","answer":"..."}',
                "Only call tools listed in the tool schema.",
                "The state digest is a compact summary (IDs, names, counts); use read tools to fetch full details such as message bodies or task fields.",
                self.service_guidance(instruction),
                f"Instruction: {instruction}",
                f"Initial state digest: {json.dumps(self.state_digest(state), sort_keys=True)}",
                f"Tool schema: {json.dumps(self.tool_schema(), sort_keys=True)}",
                f"Tool history: {json.dumps(tool_history, sort_keys=True)}",
            ]
        )

    def state_digest(self, state: dict[str, Any]) -> dict[str, Any]:
        """Compact prompt-facing view of the service state. Subclasses reduce it
        to ID/name lookup tables; anything omitted must be reachable through a
        read tool."""
        return state

    async def read_state(self, environment: BaseEnvironment) -> dict[str, Any]:
        result = await environment.exec(self.state_command(), cwd="/app", timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(f"{self.name()} state failed: {result.stderr or result.stdout}")
        return json.loads(result.stdout or "{}")

    async def invoke_tool(self, environment: BaseEnvironment, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        command = self.tool_command(tool_name, tool_input)
        result = await environment.exec(command, cwd="/app", timeout_sec=30)
        if result.return_code != 0:
            return {"error": "tool_failed", "stdout": result.stdout, "stderr": result.stderr, "return_code": result.return_code}
        return json.loads(result.stdout or "{}")

    async def maybe_upload_logs(self, environment: BaseEnvironment, *paths: Path) -> None:
        if environment.capabilities.mounted:
            return
        for path in paths:
            await environment.upload_file(path, f"/logs/agent/{path.name}")

    @abstractmethod
    def service_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def tool_schema(self) -> list[ToolSchema]:
        raise NotImplementedError

    @abstractmethod
    def state_command(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def reset_command(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def tool_command(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        raise NotImplementedError

    def service_guidance(self, instruction: str) -> str:
        return "Use tools to inspect or mutate service state before answering."


class SlackExternalAgent(RLAgent):
    """External Harbor agent for the SQLite Slack service."""

    @staticmethod
    def name() -> str:
        return "slack-ollama-external-agent"

    def service_name(self) -> str:
        return "slack"

    def get_verifier_spec(self) -> str | None:
        parent_name = self.logs_dir.parent.name
        if "slack_task_1" in parent_name:
            return "check:slack_task_1_verifier"
        elif "slack_task_2" in parent_name:
            return "check:slack_task_2_verifier"
        return None

    def tool_schema(self) -> list[ToolSchema]:
        return SLACK_TOOL_SCHEMA

    def state_command(self) -> str:
        return "python3 /app/slack_env.py state"

    def reset_command(self) -> str:
        return "python3 /app/slack_env.py teardown"

    def tool_command(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        payload_json = json.dumps(tool_input)
        return f"python3 /app/slack_env.py execute_tool {shlex.quote(tool_name)} {shlex.quote(payload_json)}"

    def state_digest(self, state: dict[str, Any]) -> dict[str, Any]:
        # Users and chats have no read tool, so keep their lookup fields;
        # messages and reactions are reachable via get_channel_messages and
        # search_messages and reduce to counts.
        return {
            "users": [
                {"user_id": u["user_id"], "display_name": u["display_name"], "handle": u["handle"]}
                for u in state.get("users", [])
            ],
            "channels": [
                {"channel_id": c["channel_id"], "name": c["name"], "is_private": c["is_private"]}
                for c in state.get("channels", [])
            ],
            "chats": [
                {
                    "chat_id": c["chat_id"],
                    "name": c["name"],
                    "type": c["type"],
                    "participant_ids": [p["user_id"] for p in c.get("participants", [])],
                }
                for c in state.get("chats", [])
            ],
            "message_count": len(state.get("messages", [])),
            "reaction_count": len(state.get("reactions", [])),
        }

    def service_guidance(self, instruction: str) -> str:
        profile = find_profile(SLACK_TASK_PROFILES, instruction)
        if profile is not None and profile.guidance:
            return profile.guidance
        return SLACK_DEFAULT_GUIDANCE


class TaskManagerExternalAgent(RLAgent):
    """External Harbor agent for Task Manager service tasks."""

    @staticmethod
    def name() -> str:
        return "task-manager-ollama-external-agent"

    def service_name(self) -> str:
        return "task_manager"

    def get_verifier_spec(self) -> str | None:
        parent_name = self.logs_dir.parent.name
        if "task_manager_task_1" in parent_name:
            return "check:task_manager_task_1_verifier"
        return None

    def tool_schema(self) -> list[ToolSchema]:
        return TASK_MANAGER_TOOL_SCHEMA

    def state_command(self) -> str:
        return "python3 /app/task_manager_env.py state"

    def reset_command(self) -> str:
        return "python3 /app/task_manager_env.py teardown"

    def tool_command(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        payload_json = json.dumps(tool_input)
        return f"python3 /app/task_manager_env.py execute_tool {shlex.quote(tool_name)} {shlex.quote(payload_json)}"

    def state_digest(self, state: dict[str, Any]) -> dict[str, Any]:
        # Users have no read tool, so keep their lookup fields. Tasks keep only
        # routing fields (full labels/descriptions via list_tasks/get_task);
        # milestones and projects via get_project/list_projects.
        return {
            "users": [
                {"user_id": u["user_id"], "display_name": u["display_name"], "handle": u["handle"]}
                for u in state.get("users", [])
            ],
            "projects": [
                {"project_id": p["project_id"], "name": p["name"], "archived": p["archived"]}
                for p in state.get("projects", [])
            ],
            "milestones": [
                {"milestone_id": m["milestone_id"], "title": m["title"], "project_id": m["project_id"]}
                for m in state.get("milestones", [])
            ],
            "tasks": [
                {
                    "task_id": t["task_id"],
                    "title": t["title"],
                    "assignee_id": t["assignee_id"],
                    "milestone_id": t["milestone_id"],
                    "project_id": t["project_id"],
                    "status": t["status"],
                }
                for t in state.get("tasks", [])
            ],
            "dependencies": [
                {"task_id": d["task_id"], "depends_on_task_id": d["depends_on_task_id"]}
                for d in state.get("dependencies", [])
            ],
            "assignment_count": len(state.get("assignments", [])),
            "audit_event_count": len(state.get("audit_events", [])),
        }

    def service_guidance(self, instruction: str) -> str:
        return TASK_MANAGER_DEFAULT_GUIDANCE
