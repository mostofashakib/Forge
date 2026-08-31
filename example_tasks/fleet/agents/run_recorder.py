"""Transcript and ATIF trajectory recording for external agent runs."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


from fleet.core.atif import atif_trajectory_to_dict


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def format_observation(tool_name: str, observation: dict[str, Any]) -> str:
    if not isinstance(observation, dict):
        return str(observation)
    
    if "error" in observation:
        return f"Error: {observation['error']} ({observation.get('message', '')})"
        
    # Slack specific formatting
    if tool_name == "search_messages":
        count = observation.get("count", 0)
        return f"Found {count} messages"
    if tool_name == "get_channel_messages":
        count = observation.get("count") or len(observation.get("messages", []))
        return f"Returned {count} messages"
    if tool_name == "list_channels":
        channels = observation.get("channels", [])
        return f"Returned {len(channels)} channels"
    if tool_name == "change_user_display_name":
        user = observation.get("user", {})
        display_name = user.get("display_name", "")
        return f"Updated display name to '{display_name}'"
    if tool_name == "create_group":
        chat = observation.get("chat", {})
        chat_id = observation.get("id") or chat.get("chat_id")
        return f"Created group chat {chat_id}"
    if tool_name == "send_group_message":
        msg = observation.get("message", {})
        msg_id = observation.get("id") or msg.get("message_id")
        return f"Sent message {msg_id}"
    if tool_name == "change_group_name":
        chat = observation.get("chat", {})
        name = chat.get("name")
        return f"Renamed group chat to '{name}'"
    if tool_name == "post_message":
        msg = observation.get("message", {})
        msg_id = observation.get("id") or msg.get("message_id")
        return f"Posted message {msg_id}"
    if tool_name == "slack.reply_to_thread":
        msg = observation.get("message", {})
        msg_id = observation.get("id") or msg.get("message_id")
        return f"Replied to thread, message_id={msg_id}"
    if tool_name == "add_reaction":
        reaction = observation.get("reaction", {})
        emoji = reaction.get("emoji")
        return f"Added reaction '{emoji}'"
        
    # Task manager specific formatting
    if tool_name in ("get_task", "create_task", "update_task", "assign_task"):
        task = observation.get("task", {})
        task_id = observation.get("id") or task.get("task_id") or task.get("id")
        assignee = task.get("assignee_id") or task.get("assignee")
        labels = task.get("labels", [])
        status = task.get("status")
        parts = []
        if task_id:
            parts.append(f"Task {task_id}")
        if assignee:
            parts.append(f"assigned to {assignee}")
        if labels:
            parts.append(f"labels={labels}")
        if status:
            parts.append(f"status={status}")
        return " ".join(parts) if parts else str(observation)

    if tool_name == "list_tasks":
        tasks = observation.get("tasks", [])
        return f"Returned {len(tasks)} tasks"
    if tool_name == "list_users":
        users = observation.get("users", [])
        return f"Returned {len(users)} users"
    if tool_name == "list_projects":
        projects = observation.get("projects", [])
        return f"Returned {len(projects)} projects"

    if len(json.dumps(observation)) < 80:
        return json.dumps(observation)
    
    parts = []
    for k, v in observation.items():
        if isinstance(v, list):
            parts.append(f"{len(v)} {k}")
        elif k in ("id", "count", "status", "name"):
            parts.append(f"{k}={v}")
    if parts:
        return "Returned " + ", ".join(parts)
        
    return json.dumps(observation, sort_keys=True)


class RunRecorder:
    """Accumulates the human-readable transcript and the ATIF trajectory steps
    for one agent run, and persists them. The transcript is flushed
    incrementally so an interrupted run still leaves evidence on disk."""

    def __init__(
        self,
        logs_dir: Path,
        instruction: str,
        model_name: str,
        verifier_spec: str | None = None,
        python_exe: Path | None = None,
        verify_script: Path | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.verifier_spec = verifier_spec
        self.python_exe = python_exe
        self.verify_script = verify_script
        self.workspace_root = workspace_root

        self.trajectory_path = logs_dir / "trajectory.json"
        self.transcript_path = logs_dir / "trajectory.txt"
        self.oracle_path = logs_dir / "oracle.txt"
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Verbose formatting for trajectory.txt (exactly as original)
        self.transcript_lines: list[str] = [
            f'Task: "{instruction}"',
            f"[AGENT] Model: {model_name}",
            "[SETUP] External Harbor agent initialized",
        ]
        # Clean formatting for oracle.txt (matching user requested agent log template)
        self.oracle_lines: list[str] = [
            f'Task: "{instruction}"',
            "[SETUP] External Harbor agent initialized",
        ]
        self.trajectory_steps: list[dict[str, Any]] = [
            {"step_id": 1, "timestamp": timestamp(), "source": "user", "message": instruction}
        ]

    def get_eval_status(self, final_state: dict[str, Any], trajectory: dict[str, Any]) -> str:
        if not self.verifier_spec or not self.python_exe or not self.verify_script or not self.workspace_root:
            return ""

        import os
        import subprocess
        import tempfile

        if self.python_exe.exists() and self.verify_script.exists() and self.trajectory_path.exists():
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_report = Path(temp_dir) / "report.json"
                    temp_reward = Path(temp_dir) / "reward.txt"
                    
                    cmd = [
                        str(self.python_exe),
                        str(self.verify_script),
                        "--trajectory", str(self.trajectory_path),
                        "--spec", self.verifier_spec,
                        "--report", str(temp_report),
                        "--reward", str(temp_reward),
                    ]
                    
                    env = dict(os.environ)
                    task_name = self.verifier_spec.split(":")[1].replace("_verifier", "").replace("_test", "")
                    env["PYTHONPATH"] = f"{self.workspace_root}/{task_name}/tests:{self.workspace_root}"
                    
                    subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
                    
                    if temp_report.exists():
                        report_data = json.loads(temp_report.read_text(encoding="utf-8"))
                        passed = report_data.get("passed", False)
                        results = report_data.get("results", [])
                        failed_msgs = [r.get("message") for r in results if not r.get("passed") and r.get("message")]
                        if failed_msgs:
                            details_str = " — " + "; ".join(failed_msgs)
                        else:
                            details_str = " — All checks passed" if passed else " — Verification failed"
                        status_symbol = "✅ PASS" if passed else "❌ FAIL"
                        return f"[EVAL] {status_symbol}{details_str}"
            except Exception as exc:
                return f"[EVAL] ❌ FAIL — Subprocess verifier error: {str(exc)}"

        return ""

    def flush(self) -> None:
        self.transcript_path.write_text("\n".join(self.transcript_lines) + "\n", encoding="utf-8")
        self.oracle_path.write_text("\n".join(self.oracle_lines) + "\n", encoding="utf-8")

    def record_state(self, label: str, state: dict[str, Any]) -> None:
        # Verbose transcript_lines (original trajectory.txt)
        self.transcript_lines.append(f"[STATE] {label}")
        self.transcript_lines.append(json.dumps(state, sort_keys=True, indent=2))
        
        # Clean oracle_lines (oracle.txt)
        if label == "Initial":
            counts = []
            for key in ["users", "channels", "projects", "milestones", "tasks", "messages", "assignments", "dependencies"]:
                val = state.get(key)
                if isinstance(val, list):
                    counts.append(f"{len(val)} {key}")
            counts_str = ", ".join(counts)
            self.oracle_lines = [line for line in self.oracle_lines if not line.startswith("[SETUP]")]
            self.oracle_lines.append(f"[SETUP] Seeded workspace: {counts_str}")

    def record_thinking(self, thought: str) -> None:
        if thought:
            self.oracle_lines.append(f"[AGENT] Thinking: {thought}")

    def record_tool_step(
        self,
        step_number: int,
        call_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        observation: dict[str, Any],
        post_state: dict[str, Any],
    ) -> None:
        # Verbose transcript_lines
        self.transcript_lines.append(f"[AGENT] Step {step_number}: calling {tool_name}")
        self.transcript_lines.append(f"[TOOL] {tool_name}({json.dumps(tool_input, sort_keys=True)})")
        
        # Clean oracle_lines
        self.oracle_lines.append(f"[TOOL] {tool_name}({json.dumps(tool_input, sort_keys=True)})")
        result_summary = format_observation(tool_name, observation)
        self.oracle_lines.append(f"[RESULT] {result_summary}")
        
        self.trajectory_steps.append(
            {
                "step_id": len(self.trajectory_steps) + 1,
                "source": "agent",
                "timestamp": timestamp(),
                "model_name": self.model_name,
                "message": f"Calling {tool_name}.",
                "reasoning_content": f"Calling {tool_name}.",
                "tool_calls": [
                    {
                        "tool_call_id": call_id,
                        "function_name": tool_name,
                        "arguments": tool_input,
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": call_id,
                            "content": json.dumps(observation, sort_keys=True),
                        }
                    ]
                },
            }
        )
        # Verbose transcript_lines
        self.transcript_lines.append("[RESULT]")
        self.transcript_lines.append(json.dumps(observation, sort_keys=True, indent=2))
        self.record_state("After interaction", post_state)

    def record_final_answer(self, answer: str) -> None:
        self.trajectory_steps.append(
            {
                "step_id": len(self.trajectory_steps) + 1,
                "source": "agent",
                "timestamp": timestamp(),
                "model_name": self.model_name,
                "message": answer,
            }
        )
        self.transcript_lines.append(f'[AGENT] Answer: "{answer}"')
        self.oracle_lines.append(f'[AGENT] Answer: "{answer}"')

    def record_note(self, note: str) -> None:
        self.transcript_lines.append(note)
        # If model got stuck/timeout, add to oracle as well
        if "Model stuck" in note or "timeout" in note:
            self.oracle_lines.append(note)

    def write_trajectory(
        self,
        *,
        service_name: str,
        agent_name: str,
        agent_version: str | None,
        extra: dict[str, Any],
    ) -> tuple[Path, Path]:
        trajectory_id = f"{service_name}-{uuid.uuid4()}"
        trajectory = atif_trajectory_to_dict(
            trajectory_id=trajectory_id,
            session_id=trajectory_id,
            agent_name=agent_name,
            agent_version=agent_version,
            model_name=self.model_name,
            steps=self.trajectory_steps,
            final_metrics={
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cached_tokens": 0,
                "total_cost_usd": 0.0,
                "total_steps": len(self.trajectory_steps),
            },
            extra=extra,
        )
        self.trajectory_path.write_text(json.dumps(trajectory, sort_keys=True, indent=2), encoding="utf-8")
        
        # Calculate eval status for oracle.txt
        final_state = extra.get("final_state_snapshot", {})
        eval_line = self.get_eval_status(final_state, trajectory)
        if eval_line:
            self.oracle_lines.append(eval_line)
            
        self.flush()
        return self.trajectory_path, self.transcript_path
