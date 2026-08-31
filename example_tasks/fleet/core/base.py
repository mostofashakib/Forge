"""Abstract interfaces for deterministic evaluation environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from fleet.core.determinism import DeterministicIdGenerator, VirtualClock
from fleet.core.instrumentation import InstrumentationSink
from fleet.core.models import Action, ErrorState, Observation, VerificationResult
from fleet.core.serialization import to_plain_data
from fleet.core.serialization import canonical_json


class BaseTool(ABC):
    name: str
    schema_version = "1.0"

    @abstractmethod
    def run(self, environment: "BaseEnvironment", action: Action) -> dict[str, Any]:
        raise NotImplementedError


class BaseAgent(ABC):
    provider_name: str

    @abstractmethod
    def next_action(self, observation: Observation) -> Action:
        raise NotImplementedError


class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, final_state: dict[str, Any], trajectory: dict[str, Any]) -> list[VerificationResult]:
        raise NotImplementedError


class BaseEnvironment(ABC):
    environment_name: str

    def __init__(self, seed: int, task_id: str = "local", instruction: str = "") -> None:
        self.seed = seed
        self.task_id = task_id
        self.instruction = instruction
        self.clock = VirtualClock()
        self.ids = DeterministicIdGenerator(seed, self.environment_name)
        self.instrumentation = InstrumentationSink(task_id, self.environment_name, seed, instruction)
        self.tools: dict[str, BaseTool] = {}
        self._initial_state = self._build_initial_state()
        self._initial_state_json = canonical_json(self._state_to_plain(self._initial_state))
        self.last_reset_determinism_check: dict[str, Any] | None = None
        self._restore_initial_state()
        self._register_tools()

    @abstractmethod
    def _build_initial_state(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _register_tools(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def _restore_initial_state(self) -> None:
        """Put live state back to the seeded initial state. The default copies
        the in-process value; storage-backed environments override this to
        re-seed their backing store instead."""
        self.state = deepcopy(self._initial_state)

    def reset(self) -> dict[str, Any]:
        self.clock.reset()
        self.ids.reset()
        self.instrumentation = InstrumentationSink(
            self.task_id,
            self.environment_name,
            self.seed,
            self.instruction,
        )
        self._restore_initial_state()
        reset_state = self.export_state()
        reset_state_json = canonical_json(reset_state)
        self.last_reset_determinism_check = {
            "passed": reset_state_json == self._initial_state_json,
            "initial_hash": self.instrumentation.state_hash(self._state_to_plain(self._initial_state)),
            "reset_hash": self.instrumentation.state_hash(reset_state),
        }
        if not self.last_reset_determinism_check["passed"]:
            raise DeterminismError("Reset did not reproduce the initial state byte-for-byte.")
        return reset_state

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool

    def execute_tool(self, tool_name: str, input_payload: dict[str, Any], acting_user_id: str) -> Observation:
        started_at = self.clock.tick()
        before_state = self.export_state()
        action = Action(tool_name=tool_name, input_payload=input_payload, acting_user_id=acting_user_id)
        call = self.instrumentation.record_tool_call(tool_name, input_payload, acting_user_id, started_at)

        tool = self.tools.get(tool_name)
        if tool is None:
            error = self.error(
                "tool_not_found",
                "invalid_tool",
                f"Tool '{tool_name}' is not registered.",
                tool_name,
                input_payload,
                retryable=False,
                state_changed=False,
            )
            return self._finish_call(call.call_id, tool_name, None, error, before_state)

        try:
            output = tool.run(self, action)
            return self._finish_call(call.call_id, tool_name, output, None, before_state)
        except EnvironmentToolError as exc:
            return self._finish_call(call.call_id, tool_name, None, exc.error, before_state)

    def error(
        self,
        error_code: str,
        error_type: str,
        message: str,
        tool_name: str,
        input_payload: dict[str, Any],
        retryable: bool,
        state_changed: bool,
    ) -> ErrorState:
        return ErrorState(
            error_code=error_code,
            error_type=error_type,
            message=message,
            tool_name=tool_name,
            input_payload=to_plain_data(input_payload),
            retryable=retryable,
            virtual_timestamp=self.clock.now_ms(),
            state_changed=state_changed,
        )

    def fail(
        self,
        error_code: str,
        error_type: str,
        message: str,
        tool_name: str,
        input_payload: dict[str, Any],
        retryable: bool = False,
        state_changed: bool = False,
    ) -> None:
        raise EnvironmentToolError(
            self.error(
                error_code,
                error_type,
                message,
                tool_name,
                input_payload,
                retryable,
                state_changed,
            )
        )

    def _finish_call(
        self,
        call_id: str,
        tool_name: str,
        output: dict[str, Any] | None,
        error: ErrorState | None,
        before_state: dict[str, Any],
    ) -> Observation:
        ended_at = self.clock.tick()
        after_state = self.export_state()
        state_changed = before_state != after_state
        self.instrumentation.record_transition(
            tool_name,
            self._transition_summary(tool_name, output, error, state_changed),
            before_state,
            after_state,
            ended_at,
        )
        self.instrumentation.record_tool_result(
            call_id,
            tool_name,
            output,
            error,
            state_changed,
            ended_at,
        )
        payload = {"tool_name": tool_name, "output": output, "state_changed": state_changed}
        if error is not None:
            payload["error"] = to_plain_data(error)
        return self.instrumentation.record_observation(
            self.environment_name,
            payload,
            ended_at,
            error=error,
        )

    def _transition_summary(
        self,
        tool_name: str,
        output: dict[str, Any] | None,
        error: ErrorState | None,
        state_changed: bool,
    ) -> str:
        if error is not None:
            return f"{tool_name} failed with {error.error_code}"
        if not state_changed:
            return f"{tool_name} completed without state change"
        entity_id = output.get("id") if output else None
        return f"{tool_name} mutated state" + (f" for {entity_id}" if entity_id else "")

    def export_trajectory(self) -> dict[str, Any]:
        return self.instrumentation.export(self.export_state())

    def export_harbor_trajectory(self) -> dict[str, Any]:
        return self.instrumentation.export_harbor(self.export_state())

    def _state_to_plain(self, state: Any) -> dict[str, Any]:
        original_state = getattr(self, "state", None)
        self.state = state
        try:
            return self.export_state()
        finally:
            self.state = original_state


class EnvironmentToolError(Exception):
    def __init__(self, error: ErrorState) -> None:
        super().__init__(error.message)
        self.error = error


class DeterminismError(Exception):
    pass
