"""Deterministic trajectory instrumentation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any

from fleet.core.determinism import DeterministicIdGenerator
from fleet.core.models import (
    ErrorState,
    HarborEvent,
    Observation,
    StateTransition,
    ToolCall,
    ToolResult,
    Trajectory,
)
from fleet.core.serialization import canonical_json, to_plain_data


class InstrumentationSink:
    def __init__(self, task_id: str, environment_name: str, seed: int, instruction: str) -> None:
        self._ids = DeterministicIdGenerator(seed, f"{environment_name}:instrumentation")
        self.trajectory = Trajectory(
            task_id=task_id,
            environment_name=environment_name,
            seed=seed,
            instruction=instruction,
        )

    def state_hash(self, state: dict[str, Any]) -> str:
        return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()

    def record_tool_call(
        self,
        tool_name: str,
        input_payload: dict[str, Any],
        acting_user_id: str,
        virtual_timestamp: int,
    ) -> ToolCall:
        call = ToolCall(
            call_id=self._ids.next("call"),
            tool_name=tool_name,
            input_payload=to_plain_data(input_payload),
            acting_user_id=acting_user_id,
            virtual_timestamp=virtual_timestamp,
        )
        self.trajectory.tool_calls.append(call)
        self.record_harbor_event(
            "tool_call",
            {
                "tool_name": tool_name,
                "arguments": to_plain_data(input_payload),
                "acting_user_id": acting_user_id,
                "call_id": call.call_id,
            },
            virtual_timestamp,
        )
        return call

    def record_tool_result(
        self,
        call_id: str,
        tool_name: str,
        output: dict[str, Any] | None,
        error: ErrorState | None,
        state_changed: bool,
        virtual_timestamp: int,
    ) -> ToolResult:
        result = ToolResult(
            call_id=call_id,
            tool_name=tool_name,
            output=to_plain_data(output) if output is not None else None,
            error=error,
            state_changed=state_changed,
            virtual_timestamp=virtual_timestamp,
        )
        self.trajectory.tool_outputs.append(result)
        if error is not None:
            self.trajectory.errors.append(error)
        self.record_harbor_event(
            "tool_result",
            {
                "tool_name": tool_name,
                "call_id": call_id,
                "output": to_plain_data(output) if output is not None else None,
                "error": to_plain_data(error) if error is not None else None,
                "state_changed": state_changed,
            },
            virtual_timestamp,
        )
        return result

    def record_observation(
        self,
        environment_name: str,
        payload: dict[str, Any],
        virtual_timestamp: int,
        error: ErrorState | None = None,
    ) -> Observation:
        observation = Observation(
            observation_id=self._ids.next("obs"),
            environment_name=environment_name,
            virtual_timestamp=virtual_timestamp,
            payload=to_plain_data(payload),
            error=error,
        )
        self.trajectory.observations.append(observation)
        return observation

    def record_transition(
        self,
        tool_name: str,
        summary: str,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        virtual_timestamp: int,
    ) -> None:
        before_hash = self.state_hash(before_state)
        after_hash = self.state_hash(after_state)
        if before_hash == after_hash:
            return
        self.trajectory.state_transitions.append(
            StateTransition(
                transition_id=self._ids.next("transition"),
                tool_name=tool_name,
                summary=summary,
                before_hash=before_hash,
                after_hash=after_hash,
                virtual_timestamp=virtual_timestamp,
            )
        )

    def record_verifier_output(self, output: Any) -> None:
        plain_output = to_plain_data(output)
        self.trajectory.verifier_outputs.append(plain_output)
        self.record_harbor_event("eval", plain_output, self._last_timestamp())

    def record_harbor_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        virtual_timestamp: int,
    ) -> HarborEvent:
        event = HarborEvent(
            event_id=self._ids.next("event"),
            event_type=event_type,
            virtual_timestamp=virtual_timestamp,
            payload=to_plain_data(payload),
        )
        self.trajectory.harbor_events.append(event)
        return event

    def export(self, final_state: dict[str, Any]) -> dict[str, Any]:
        self.trajectory.final_state_snapshot = to_plain_data(final_state)
        return to_plain_data(asdict(self.trajectory))

    def export_harbor(self, final_state: dict[str, Any]) -> dict[str, Any]:
        self.trajectory.final_state_snapshot = to_plain_data(final_state)
        return {
            "schema_version": "1.3",
            "task_id": self.trajectory.task_id,
            "environment_name": self.trajectory.environment_name,
            "seed": self.trajectory.seed,
            "instruction": self.trajectory.instruction,
            "events": to_plain_data(self.trajectory.harbor_events),
            "artifacts": {
                "observations": to_plain_data(self.trajectory.observations),
                "tool_calls": to_plain_data(self.trajectory.tool_calls),
                "tool_outputs": to_plain_data(self.trajectory.tool_outputs),
                "error_states": to_plain_data(self.trajectory.errors),
                "state_transitions": to_plain_data(self.trajectory.state_transitions),
                "verifier_outputs": to_plain_data(self.trajectory.verifier_outputs),
                "final_state_snapshot": to_plain_data(final_state),
            },
        }

    def _last_timestamp(self) -> int:
        event = self.trajectory.harbor_events[-1] if self.trajectory.harbor_events else None
        return event.virtual_timestamp if event is not None else 0
