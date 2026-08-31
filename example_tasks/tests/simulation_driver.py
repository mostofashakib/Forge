"""Command line driver for deterministic simulations."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from fleet.agents.model_adapters import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    ModelGenerationTimeout,
    create_adapter,
)
from fleet.core.base import BaseEnvironment
from fleet.core.models import VerificationResult
from fleet.core.serialization import canonical_json
from fleet.environments.slack.environment import SlackEnvironment
from fleet.environments.slack.reference_agents import first_channel_reference
from fleet.environments.slack.schema import SLACK_TOOL_SCHEMA
from fleet.environments.task_manager.environment import TaskManagerEnvironment
from fleet.environments.task_manager.schema import TASK_MANAGER_TOOL_SCHEMA
from fleet.core.atif import atif_trajectory_to_dict
from fleet.core.atif import write_trajectory

DEFAULT_AGENT_PROVIDER = "ollama"
DEFAULT_AGENT_MODEL = DEFAULT_MODEL

INCIDENT_TASK = "Find the most recent message from @alice in #incidents and return the channel it references."
TASK_MANAGER_TASK = "Move the task 'Draft benchmark plan' from PENDING to IN_PROGRESS."


@dataclass(frozen=True)
class DriverStep:
    thought: str
    tool_name: str
    input_payload: dict[str, Any]


@dataclass(frozen=True)
class DriverTask:
    task_id: str
    environment_name: str
    instruction: str
    acting_user_id: str
    steps: list[DriverStep] = field(default_factory=list)
    expected_answer: str | None = None
    final_answer: str | None = None


@dataclass(frozen=True)
class TaskPlaybook:
    """Everything the driver knows about one builtin task, kept out of the
    generic driver loop: scripted reference steps, the answer-format contract,
    and how to derive the final answer. `prompt_guidance` must describe the
    answer format only — never the solution path (tools, order, or IDs), or
    real-agent runs stop measuring the model."""

    prompt_guidance: str
    scripted_steps: Callable[[BaseEnvironment], list[DriverStep]]
    derive_answer: Callable[[dict[str, Any], dict[str, Any]], str]


def _slack_incident_steps(environment: BaseEnvironment) -> list[DriverStep]:
    return [
        DriverStep("I need to search for alice's messages in #incidents.", "search_messages", {"query": "alice incidents"}),
        DriverStep("I need the complete incident channel history to confirm recency.", "get_channel_messages", {"channel_id": "C003"}),
    ]


def _slack_incident_answer(working_context: dict[str, Any], final_state: dict[str, Any]) -> str:
    # Real-agent runs no longer force a tool order, so the first step is not
    # guaranteed to be a search result; report no answer instead of crashing.
    search_output = working_context.get("step_1_output")
    if not isinstance(search_output, dict) or not search_output.get("messages"):
        return ""
    most_recent = search_output["messages"][0]
    reference = first_channel_reference(most_recent["body"])
    return f"Alice's most recent message references {reference}"


def _benchmark_plan_steps(environment: BaseEnvironment) -> list[DriverStep]:
    task_id = find_task_id_by_title(environment.export_state(), "Draft benchmark plan")
    return [
        DriverStep("I need to inspect the current task list.", "list_tasks", {}),
        DriverStep(
            "I found the benchmark plan task and need to move it to IN_PROGRESS.",
            "update_task",
            {"task_id": task_id, "status": "IN_PROGRESS"},
        ),
    ]


def _benchmark_plan_answer(working_context: dict[str, Any], final_state: dict[str, Any]) -> str:
    task = next(
        item
        for item in final_state["tasks"]
        if item["title"] == "Draft benchmark plan"
    )
    return f"Draft benchmark plan is {task['status']}"


TASK_PLAYBOOKS: dict[str, TaskPlaybook] = {
    "slack_task_1": TaskPlaybook(
        prompt_guidance=(
            "The final answer must be exactly: Alice's most recent message references <channel>, "
            "with <channel> being the referenced channel name including the leading #."
        ),
        scripted_steps=_slack_incident_steps,
        derive_answer=_slack_incident_answer,
    ),
    "task_manager_move_benchmark_plan": TaskPlaybook(
        prompt_guidance=(
            "The final answer must be exactly: Draft benchmark plan is <STATUS>, "
            "with <STATUS> being the task's final uppercase status."
        ),
        scripted_steps=_benchmark_plan_steps,
        derive_answer=_benchmark_plan_answer,
    ),
}


class DebugPrinter:
    def __init__(
        self,
        enabled: bool,
        transcript_path: str | Path | None = None,
        transcript_append: bool = False,
    ) -> None:
        self.enabled = enabled
        self.transcript_path = Path(transcript_path) if transcript_path else None
        self.transcript_append = transcript_append
        self.lines: list[str] = []

    def line(self, text: str) -> None:
        if self.enabled:
            print(text)
        if self.transcript_path:
            self.lines.append(text)

    def json_block(self, label: str, payload: dict[str, Any]) -> None:
        self.line(label)
        self.line(json.dumps(payload, sort_keys=True, indent=2))

    def write(self) -> None:
        if not self.transcript_path:
            return
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self.lines) + "\n"
        if self.transcript_append:
            with self.transcript_path.open("a", encoding="utf-8") as handle:
                handle.write(content)
            return
        self.transcript_path.write_text(content, encoding="utf-8")


class AgentGenerationError(RuntimeError):
    pass


class JsonToolAgent:
    """Provider-agnostic JSON tool agent. The model string picks the backend
    via create_adapter: local names go to Ollama, provider-prefixed names
    (openai/..., anthropic/...) go through litellm."""

    def __init__(self, model: str, host: str = DEFAULT_OLLAMA_HOST, timeout_sec: float = 120.0) -> None:
        self.model = model
        self.adapter = create_adapter(model, host.rstrip("/"), stream_stall_timeout_sec=timeout_sec)

    def next_action(
        self,
        task: DriverTask,
        state: dict[str, Any],
        history: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = build_agent_prompt(task, state, history, available_tools)
        try:
            raw_response = self.adapter.generate(prompt, self.model)
        except ModelGenerationTimeout as exc:
            raise AgentGenerationError(f"Model {self.model!r} timed out.") from exc
        except RuntimeError as exc:
            raise AgentGenerationError(str(exc)) from exc

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise AgentGenerationError(f"Model returned non-JSON response: {raw_response!r}") from exc
        if not isinstance(parsed, dict):
            raise AgentGenerationError(f"Model returned JSON that is not an object: {parsed!r}")
        return parsed


def run_incident_reference_simulation(seed: int, output_path: str | Path, verbose: bool = False) -> dict[str, Any]:
    result = run_task_list(
        tasks=[builtin_slack_incident_task()],
        seed=seed,
        output_path=output_path,
        verbose=verbose,
        agent_provider=DEFAULT_AGENT_PROVIDER,
        agent_model=DEFAULT_AGENT_MODEL,
        real_agent=False,
        transcript_path=Path(output_path).with_suffix(".txt"),
    )
    return result["runs"][0]["trajectory"]


def run_task_list(
    tasks: list[DriverTask],
    seed: int,
    output_path: str | Path,
    verbose: bool,
    agent_provider: str,
    agent_model: str,
    real_agent: bool = False,
    transcript_path: str | Path | None = None,
    transcript_append: bool = False,
) -> dict[str, Any]:
    debug = DebugPrinter(verbose, transcript_path, transcript_append)
    runs = []
    agent = JsonToolAgent(agent_model) if real_agent else None
    try:
        for task in tasks:
            runs.append(run_single_task(task, seed, debug, agent_provider, agent_model, agent))

        bundle = build_output_bundle(runs, agent_provider, agent_model)
        write_trajectory(output_path, bundle if len(runs) != 1 else runs[0]["trajectory"])
        debug.line(f"[HARBOR] trajectory={output_path}")
        if transcript_path:
            debug.line(f"[TRANSCRIPT] output={transcript_path}")
        return bundle
    finally:
        debug.write()


def run_single_task(
    task: DriverTask,
    seed: int,
    debug: DebugPrinter,
    agent_provider: str,
    agent_model: str,
    agent: JsonToolAgent | None,
) -> dict[str, Any]:
    environment = create_environment(task, seed)
    initial_state = environment.export_state()
    initial_state_json = canonical_json(initial_state)
    acting_user_id = resolve_acting_user_id(task.acting_user_id, initial_state)

    debug.line("")
    debug.line(f"Task: {task.instruction}")
    debug.line(f"[AGENT] provider={agent_provider} model={agent_model}")
    debug.json_block("[STATE initial]", initial_state)

    environment.instrumentation.record_harbor_event(
        "setup",
        setup_payload(task.environment_name, initial_state),
        environment.clock.now_ms(),
    )
    environment.instrumentation.record_harbor_event(
        "agent_config",
        {"provider": agent_provider, "model": agent_model},
        environment.clock.now_ms(),
    )

    step_results = []
    working_context: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    steps = resolve_real_agent_steps(task, environment, agent, history) if agent is not None else resolve_steps(task, environment, working_context)
    for index, step in enumerate(steps, start=1):
        before_state = environment.export_state()
        before_hash = environment.instrumentation.state_hash(before_state)
        debug.line(f"[AGENT step={index}] Thinking: {step.thought}")
        debug.line(f"[TOOL step={index}] {step.tool_name}({json.dumps(step.input_payload, sort_keys=True)})")
        environment.instrumentation.record_harbor_event(
            "agent_thought",
            {"step": index, "text": step.thought},
            environment.clock.now_ms(),
        )

        observation = environment.execute_tool(step.tool_name, step.input_payload, acting_user_id)
        after_state = environment.export_state()
        after_hash = environment.instrumentation.state_hash(after_state)
        output = observation.payload.get("output")
        debug.json_block(f"[RESULT step={index}]", output if isinstance(output, dict) else {"output": output})
        debug.json_block(
            f"[CHANGE step={index}]",
            summarize_change(before_state, after_state, before_hash, after_hash),
        )
        debug.json_block(f"[STATE after_step={index}]", after_state)
        step_results.append(observation.payload)
        working_context[f"step_{index}_output"] = output
        history.append(
            {
                "thought": step.thought,
                "tool_name": step.tool_name,
                "input_payload": step.input_payload,
                "observation": observation.payload,
            }
        )

    answer = derive_real_agent_answer(task, environment, agent, history) if agent is not None else derive_answer(task, working_context, environment.export_state())
    environment.instrumentation.record_harbor_event(
        "agent_answer",
        {"answer": answer},
        environment.clock.now_ms(),
    )
    debug.line(f"[AGENT] Answer: {answer!r}")

    verification = verify_answer(answer, task.expected_answer)
    environment.instrumentation.record_verifier_output(asdict(verification))
    debug.line(f"[ANSWER CHECK] {'MATCH' if verification.passed else 'MISMATCH'} - {verification.message}")

    final_state = environment.export_state()
    debug.json_block("[STATE final]", final_state)
    legacy_trajectory = environment.export_harbor_trajectory()
    legacy_trajectory["agent"] = {"provider": agent_provider, "model": agent_model}
    legacy_trajectory["debug"] = {"step_results": step_results}

    reset_state = environment.reset()
    reset_state_json = canonical_json(reset_state)
    reset_check = {
        "passed": initial_state_json == reset_state_json,
        "initial_hash": environment.last_reset_determinism_check["initial_hash"],
        "reset_hash": environment.last_reset_determinism_check["reset_hash"],
        "environment_check": environment.last_reset_determinism_check,
    }
    if not reset_check["passed"]:
        raise RuntimeError("Reset determinism check failed.")
    replay_environment = create_environment(task, seed)
    if canonical_json(replay_environment.export_state()) != initial_state_json:
        raise RuntimeError("Fresh seeded workspace determinism check failed.")

    # Trajectory replay determinism: replaying the agent's own recorded tool
    # sequence on a fresh seed must reproduce the final state byte-for-byte.
    # No particular order is enforced — whatever order the agent chose, the
    # environment must respond identically given the same trajectory.
    for step in steps:
        replay_environment.execute_tool(step.tool_name, step.input_payload, acting_user_id)
    replay_final_json = canonical_json(replay_environment.export_state())
    reset_check["trajectory_replay"] = {
        "passed": replay_final_json == canonical_json(final_state),
        "steps": len(steps),
    }
    if not reset_check["trajectory_replay"]["passed"]:
        raise RuntimeError("Trajectory replay determinism check failed.")
    debug.json_block("[RESET deterministic_check]", reset_check)

    legacy_trajectory["reset_determinism_check"] = reset_check
    atif_trajectory = build_atif_trajectory(
        task=task,
        seed=seed,
        agent_provider=agent_provider,
        agent_model=agent_model,
        legacy_trajectory=legacy_trajectory,
    )
    return {
        "task_id": task.task_id,
        "environment_name": task.environment_name,
        "trajectory": atif_trajectory,
    }


@dataclass(frozen=True)
class EnvironmentEntry:
    factory: Callable[..., BaseEnvironment]
    tool_schema: list[dict[str, Any]]
    setup_summary: Callable[[dict[str, Any]], dict[str, Any]]


ENVIRONMENT_REGISTRY: dict[str, EnvironmentEntry] = {
    "slack": EnvironmentEntry(
        factory=SlackEnvironment,
        tool_schema=SLACK_TOOL_SCHEMA,
        setup_summary=lambda state: {
            "description": "Seeded workspace",
            "users": len(state["users"]),
            "channels": len(state["channels"]),
            "messages": len(state["messages"]),
        },
    ),
    "task_manager": EnvironmentEntry(
        factory=TaskManagerEnvironment,
        tool_schema=TASK_MANAGER_TOOL_SCHEMA,
        setup_summary=lambda state: {
            "description": "Seeded task manager",
            "users": len(state["users"]),
            "tasks": len(state["tasks"]),
            "assignments": len(state["assignments"]),
        },
    ),
}


def create_environment(task: DriverTask, seed: int) -> BaseEnvironment:
    entry = ENVIRONMENT_REGISTRY.get(task.environment_name)
    if entry is None:
        raise ValueError(f"Unknown environment: {task.environment_name}")
    return entry.factory(seed=seed, task_id=task.task_id, instruction=task.instruction)


def build_output_bundle(runs: list[dict[str, Any]], agent_provider: str, agent_model: str) -> dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.7",
        "agent": {
            "name": agent_provider,
            "version": None,
            "model_name": agent_model,
        },
        "trajectories": [run["trajectory"] for run in runs],
        "runs": runs,
        "extra": {
            "bundle_type": "multi_task",
            "runs": [
                {
                    "task_id": run["task_id"],
                    "environment_name": run["environment_name"],
                    "trajectory_id": run["trajectory"]["trajectory_id"],
                }
                for run in runs
            ],
        },
    }


def build_atif_trajectory(
    task: DriverTask,
    seed: int,
    agent_provider: str,
    agent_model: str,
    legacy_trajectory: dict[str, Any],
) -> dict[str, Any]:
    initial_timestamp = legacy_trajectory["events"][0]["virtual_timestamp"] if legacy_trajectory["events"] else 0
    steps: list[dict[str, Any]] = [
        {
            "step_id": 1,
            "timestamp": iso_timestamp(initial_timestamp),
            "source": "user",
            "message": task.instruction,
        }
    ]

    thought_by_step = {
        event["payload"]["step"]: event["payload"]["text"]
        for event in legacy_trajectory["events"]
        if event["event_type"] == "agent_thought"
    }
    tool_outputs_by_call_id = {
        output["call_id"]: output
        for output in legacy_trajectory["artifacts"]["tool_outputs"]
    }
    for index, call in enumerate(legacy_trajectory["artifacts"]["tool_calls"], start=1):
        tool_output = tool_outputs_by_call_id.get(call["call_id"], {})
        observation_content = tool_output.get("error") or tool_output.get("output") or {}
        steps.append(
            {
                "step_id": len(steps) + 1,
                "timestamp": iso_timestamp(call["virtual_timestamp"]),
                "source": "agent",
                "model_name": agent_model,
                "message": thought_by_step.get(index, f"Calling {call['tool_name']}."),
                "reasoning_content": thought_by_step.get(index, ""),
                "tool_calls": [
                    {
                        "tool_call_id": call["call_id"],
                        "function_name": call["tool_name"],
                        "arguments": call["input_payload"],
                        "extra": {
                            "acting_user_id": call["acting_user_id"],
                            "virtual_timestamp": call["virtual_timestamp"],
                        },
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": call["call_id"],
                            "content": json.dumps(observation_content, sort_keys=True),
                            "extra": {
                                "state_changed": tool_output.get("state_changed", False),
                                "error": tool_output.get("error"),
                                "virtual_timestamp": tool_output.get("virtual_timestamp"),
                            },
                        }
                    ]
                },
                "llm_call_count": 1,
            }
        )

    answer_event = next(
        (event for event in legacy_trajectory["events"] if event["event_type"] == "agent_answer"),
        None,
    )
    if answer_event is not None:
        steps.append(
            {
                "step_id": len(steps) + 1,
                "timestamp": iso_timestamp(answer_event["virtual_timestamp"]),
                "source": "agent",
                "model_name": agent_model,
                "message": answer_event["payload"]["answer"],
                "llm_call_count": 1,
            }
        )

    trajectory_id = f"{task.task_id}-seed-{seed}"
    return atif_trajectory_to_dict(
        trajectory_id=trajectory_id,
        session_id=trajectory_id,
        agent_name=agent_provider,
        agent_version=None,
        model_name=agent_model,
        steps=steps,
        final_metrics={
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
            "total_cost_usd": 0.0,
            "total_steps": len(steps),
        },
        extra={
            "agent": {
                "environment_name": task.environment_name,
                "acting_user_id": task.acting_user_id,
            },
            "task_id": task.task_id,
            "environment_name": task.environment_name,
            "seed": seed,
            "instruction": task.instruction,
            "expected_answer": task.expected_answer,
            "reset_determinism_check": legacy_trajectory["reset_determinism_check"],
            "legacy": legacy_trajectory,
        },
    )


def legacy_trajectory_view(trajectory: dict[str, Any]) -> dict[str, Any]:
    return trajectory.get("extra", {}).get("legacy", trajectory)


def legacy_artifacts(trajectory: dict[str, Any]) -> dict[str, Any]:
    return legacy_trajectory_view(trajectory)["artifacts"]


def legacy_events(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    return legacy_trajectory_view(trajectory)["events"]


def reset_determinism_check(trajectory: dict[str, Any]) -> dict[str, Any]:
    return trajectory.get("extra", {}).get(
        "reset_determinism_check",
        legacy_trajectory_view(trajectory).get("reset_determinism_check", {}),
    )


def iso_timestamp(virtual_timestamp_ms: int) -> str:
    return datetime.fromtimestamp(virtual_timestamp_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_real_agent_steps(
    task: DriverTask,
    environment: BaseEnvironment,
    agent: JsonToolAgent | None,
    history: list[dict[str, Any]],
) -> list[DriverStep]:
    if agent is None:
        return []
    steps: list[DriverStep] = []
    planning_history = deepcopy(history)
    seen_actions: set[str] = set()
    for _ in range(6):
        action = agent.next_action(task, environment.export_state(), planning_history, available_tools(task.environment_name))
        if "final_answer" in action:
            break
        tool_name = str(action.get("tool_name", ""))
        input_payload = action.get("input_payload", {})
        thought = str(action.get("thought", ""))
        if not tool_name:
            planning_history.append(
                {
                    "feedback": "invalid_action",
                    "message": "Return a tool call with tool_name and input_payload, or final_answer when done.",
                    "invalid_action": action,
                }
            )
            continue
        if not isinstance(input_payload, dict):
            raise AgentGenerationError(f"Agent input_payload must be an object: {action!r}")
        action_key = canonical_json({"tool_name": tool_name, "input_payload": input_payload})
        if action_key in seen_actions:
            planning_history.append(
                {
                    "feedback": "duplicate_tool_call",
                    "message": "This exact call already ran; choose a different action or provide final_answer.",
                    "duplicate": {"tool_name": tool_name, "input_payload": input_payload},
                }
            )
            continue
        seen_actions.add(action_key)
        step = DriverStep(thought=thought, tool_name=tool_name, input_payload=input_payload)
        steps.append(step)
        observation = environment.execute_tool(step.tool_name, step.input_payload, resolve_acting_user_id(task.acting_user_id, environment.export_state()))
        planning_history.append(
            {
                "thought": step.thought,
                "tool_name": step.tool_name,
                "input_payload": step.input_payload,
                "observation": observation.payload,
            }
        )
        if observation.error is not None:
            break
    environment.reset()
    return steps


def resolve_steps(
    task: DriverTask,
    environment: BaseEnvironment,
    working_context: dict[str, Any],
) -> list[DriverStep]:
    if task.steps:
        return task.steps
    playbook = TASK_PLAYBOOKS.get(task.task_id)
    return playbook.scripted_steps(environment) if playbook else []


def derive_real_agent_answer(
    task: DriverTask,
    environment: BaseEnvironment,
    agent: JsonToolAgent | None,
    history: list[dict[str, Any]],
) -> str:
    if agent is None:
        return ""
    response = agent.next_action(task, environment.export_state(), history, available_tools(task.environment_name))
    if "final_answer" not in response:
        return derive_answer(task, {f"step_{index}_output": item["observation"].get("output") for index, item in enumerate(history, start=1)}, environment.export_state())
    return str(response["final_answer"])


def derive_answer(task: DriverTask, working_context: dict[str, Any], final_state: dict[str, Any]) -> str:
    playbook = TASK_PLAYBOOKS.get(task.task_id)
    if playbook is not None:
        return playbook.derive_answer(working_context, final_state)
    return task.final_answer or ""


def available_tools(environment_name: str) -> list[dict[str, Any]]:
    entry = ENVIRONMENT_REGISTRY.get(environment_name)
    return entry.tool_schema if entry else []


def build_agent_prompt(
    task: DriverTask,
    state: dict[str, Any],
    history: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    playbook = TASK_PLAYBOOKS.get(task.task_id)
    lines = [
        "You are an agent interacting with a deterministic evaluation environment.",
        "Return exactly one compact JSON object and no markdown.",
        "To call a tool, return: {\"thought\":\"...\",\"tool_name\":\"...\",\"input_payload\":{...}}",
        "When done, return: {\"thought\":\"...\",\"final_answer\":\"...\"}",
        "Keep thought under 15 words.",
        "Use only the listed tools. Do not invent IDs; read them from state or observations.",
        "Do not answer directly from state. You must call tools before returning final_answer.",
    ]
    if playbook is not None and playbook.prompt_guidance:
        lines.append(playbook.prompt_guidance)
    lines.extend(
        [
            f"Instruction: {task.instruction}",
            f"Environment: {task.environment_name}",
            f"Available tools: {json.dumps(tools, sort_keys=True)}",
            f"Current state: {json.dumps(state, sort_keys=True)}",
            f"History: {json.dumps(history, sort_keys=True)}",
        ]
    )
    return "\n".join(lines)


def verify_answer(answer: str, expected_answer: str | None) -> VerificationResult:
    if expected_answer is None:
        return VerificationResult(1, "state", True, "No expected answer configured.", {"actual": answer})
    passed = answer == expected_answer
    return VerificationResult(
        score=1 if passed else 0,
        layer="state",
        passed=passed,
        message=(
            f"Agent returned expected answer: {expected_answer}"
            if passed
            else f"Agent returned {answer!r}; expected {expected_answer!r}"
        ),
        details={"expected": expected_answer, "actual": answer},
    )


def summarize_change(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    before_hash: str,
    after_hash: str,
) -> dict[str, Any]:
    changed_collections = []
    collection_counts = {}
    for key in sorted(set(before_state) | set(after_state)):
        before_collection = before_state.get(key, [])
        after_collection = after_state.get(key, [])
        before_count = len(before_collection) if isinstance(before_collection, (dict, list)) else 0
        after_count = len(after_collection) if isinstance(after_collection, (dict, list)) else 0
        if before_collection != after_collection:
            changed_collections.append(key)
        collection_counts[key] = {"before": before_count, "after": after_count}
    return {
        "state_changed": before_hash != after_hash,
        "before_hash": before_hash,
        "after_hash": after_hash,
        "changed_collections": changed_collections,
        "collection_counts": collection_counts,
    }


def setup_payload(environment_name: str, state: dict[str, Any]) -> dict[str, Any]:
    entry = ENVIRONMENT_REGISTRY.get(environment_name)
    if entry is None:
        return {"description": "Seeded environment"}
    return entry.setup_summary(state)


def find_task_id_by_title(state: dict[str, Any], title: str) -> str:
    for task in state["tasks"]:
        if task["title"] == title:
            return task["task_id"]
    raise ValueError(f"Task not found: {title}")


def resolve_acting_user_id(acting_user_id: str, state: dict[str, Any]) -> str:
    if acting_user_id == "__admin__":
        for user in state["users"]:
            if user["role"] == "admin":
                return user["user_id"]
        raise ValueError("No admin user found.")
    return acting_user_id


def builtin_slack_incident_task() -> DriverTask:
    return DriverTask(
        task_id="slack_task_1",
        environment_name="slack",
        instruction=INCIDENT_TASK,
        acting_user_id="U002",
        expected_answer="Alice's most recent message references #platform-outages",
    )


def builtin_task_manager_task() -> DriverTask:
    return DriverTask(
        task_id="task_manager_move_benchmark_plan",
        environment_name="task_manager",
        instruction=TASK_MANAGER_TASK,
        acting_user_id="__admin__",
        expected_answer="Draft benchmark plan is IN_PROGRESS",
    )


def load_tasks(path: str | None, environment_name: str) -> list[DriverTask]:
    if path is None:
        if environment_name == "slack":
            return [builtin_slack_incident_task()]
        if environment_name == "task_manager":
            return [builtin_task_manager_task()]
        return [builtin_slack_incident_task(), builtin_task_manager_task()]

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Task file must contain a JSON list.")
    return [task_from_dict(item) for item in data]


def task_from_dict(data: dict[str, Any]) -> DriverTask:
    raw_steps = data.get("steps", [])
    steps = [
        DriverStep(
            thought=str(step.get("thought", "")),
            tool_name=str(step["tool_name"]),
            input_payload=deepcopy(step.get("input_payload", {})),
        )
        for step in raw_steps
    ]
    return DriverTask(
        task_id=str(data["task_id"]),
        environment_name=str(data["environment_name"]),
        instruction=str(data["instruction"]),
        acting_user_id=str(data["acting_user_id"]),
        steps=steps,
        expected_answer=data.get("expected_answer"),
        final_answer=data.get("final_answer"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic agent evaluation simulations.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--environment", choices=["slack", "task_manager", "all"], default="all")
    parser.add_argument("--tasks", help="Path to a JSON list of task definitions.")
    parser.add_argument("--output", default="/tmp/fleet/trajectory.json")
    parser.add_argument("--agent-provider", default=DEFAULT_AGENT_PROVIDER)
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--quiet", action="store_true", help="Disable verbose print debugging.")
    parser.add_argument("--real-agent", action="store_true", help="Use local Ollama model instead of scripted steps.")
    parser.add_argument("--transcript", help="Path for formatted text debug output. Defaults to output path with .txt.")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks, args.environment)
    transcript_path = args.transcript or str(Path(args.output).with_suffix(".txt"))
    run_task_list(
        tasks=tasks,
        seed=args.seed,
        output_path=args.output,
        verbose=not args.quiet,
        agent_provider=args.agent_provider,
        agent_model=args.agent_model,
        real_agent=args.real_agent,
        transcript_path=transcript_path,
    )


if __name__ == "__main__":
    main()
