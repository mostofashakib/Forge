from __future__ import annotations

import json

import pytest

from forge.runtime.agent_logger import AgentRunLogger, LogEntry, run_logged_episode


def _fake_clock():
    counter = {"n": 0}

    def clock() -> str:
        value = f"t{counter['n']}"
        counter["n"] += 1
        return value

    return clock


# ---------------------------------------------------------------------------
# AgentRunLogger core
# ---------------------------------------------------------------------------

def test_logger_records_correlated_ordered_entries():
    logger = AgentRunLogger(run_id="ep1", clock=_fake_clock())
    logger.start_run(metadata={"seed": 7})
    logger.set_step(0)
    logger.log_llm_call(prompt="p0", tool_call={"type": "a"}, response="r0")
    logger.log_action(action={"type": "a"}, result={"ok": True}, reward=1.0)
    logger.log_state_change(before={"x": 0}, after={"x": 1})
    logger.end_run(status="completed")

    entries = logger.entries
    assert [e.kind for e in entries] == [
        "run_start", "llm_call", "action", "state_change", "run_end",
    ]
    # Correlated by run id, ordered by a strictly increasing seq.
    assert all(e.run_id == "ep1" for e in entries)
    assert [e.seq for e in entries] == [0, 1, 2, 3, 4]
    # The three in-step entries carry step 0; run start/end are not step-scoped.
    assert [e.step for e in entries] == [None, 0, 0, 0, None]
    # Timestamps come from the injected clock in emission order.
    assert [e.ts for e in entries] == ["t0", "t1", "t2", "t3", "t4"]

    llm = entries[1]
    assert llm.payload["prompt"] == "p0"
    assert llm.payload["tool_call"] == {"type": "a"}
    assert llm.payload["response"] == "r0"


def test_end_run_records_failure_status_and_error():
    logger = AgentRunLogger(run_id="ep2")
    logger.start_run()
    logger.set_step(0)
    logger.log_action(action={"type": "a"}, result={"ok": False})
    logger.end_run(status="failed", error="boom")

    trace = logger.trace()
    assert trace["run_id"] == "ep2"
    assert trace["status"] == "failed"
    assert trace["error"] == "boom"
    assert len(trace["entries"]) == 3


def test_trace_status_is_incomplete_before_end_run():
    # A run that never reached end_run must still expose a coherent, queryable
    # partial trace rather than raising.
    logger = AgentRunLogger(run_id="ep3")
    logger.start_run()
    logger.set_step(0)
    logger.log_llm_call(prompt="p", tool_call={"type": "a"}, response="r")

    trace = logger.trace()
    assert trace["status"] == "incomplete"
    assert [e["kind"] for e in trace["entries"]] == ["run_start", "llm_call"]


def test_to_jsonl_roundtrips_every_entry():
    logger = AgentRunLogger(run_id="ep4", clock=_fake_clock())
    logger.start_run()
    logger.set_step(0)
    logger.log_action(action={"type": "a"}, result={"ok": True}, reward=0.5)
    logger.end_run()

    lines = logger.to_jsonl().splitlines()
    assert len(lines) == len(logger.entries)
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["kind"] == "run_start"
    assert all(row["run_id"] == "ep4" for row in parsed)
    # Round-trips back into LogEntry.
    assert [LogEntry(**row).kind for row in parsed] == [e.kind for e in logger.entries]


def test_entries_for_step_filters_by_step():
    logger = AgentRunLogger(run_id="ep5")
    logger.start_run()
    logger.set_step(0)
    logger.log_action(action={"type": "a"}, result={})
    logger.set_step(1)
    logger.log_action(action={"type": "b"}, result={})

    step1 = logger.entries_for_step(1)
    assert len(step1) == 1
    assert step1[0].payload["action"] == {"type": "b"}


def test_log_helpers_default_to_current_step():
    logger = AgentRunLogger(run_id="ep6")
    logger.start_run()
    logger.set_step(3)
    logger.log_llm_call(prompt="p", tool_call={"type": "a"}, response="r")
    assert logger.entries[-1].step == 3
    # An explicit step overrides the current one.
    logger.log_action(action={"type": "a"}, result={}, step=9)
    assert logger.entries[-1].step == 9


# ---------------------------------------------------------------------------
# run_logged_episode driver
# ---------------------------------------------------------------------------

class _CounterEnv:
    """Minimal gym-like env: state x increments each step, terminates at `stop`."""

    def __init__(self, stop: int = 3, fail_at: int | None = None) -> None:
        self._x = 0
        self._stop = stop
        self._fail_at = fail_at
        self.action_types = frozenset({"inc"})

    def reset(self, seed=None, options=None):
        self._x = 0
        return {"x": self._x}, {"seed": seed}

    def step(self, action):
        self._x += 1
        if self._fail_at is not None and self._x == self._fail_at:
            raise RuntimeError("env exploded")
        terminated = self._x >= self._stop
        return {"x": self._x}, 1.0, terminated, False, {"status_code": 200}


class _StubAgent:
    def __init__(self) -> None:
        self.logger = None

    def act(self, obs, action_types):
        action = {"type": "inc"}
        if self.logger is not None:
            self.logger.log_llm_call(
                prompt=json.dumps(obs), tool_call=action, response="stub-response"
            )
        return action


def test_run_logged_episode_captures_full_correlated_trace():
    logger = AgentRunLogger(run_id="run1")
    run_logged_episode(_CounterEnv(stop=3), _StubAgent(), logger, seed=42, max_steps=10)

    trace = logger.trace()
    assert trace["status"] == "completed"
    # run_start carries the seed.
    assert trace["entries"][0]["payload"]["seed"] == 42
    # Three steps, each with an llm_call (from the agent) + action + state_change.
    for step in (0, 1, 2):
        kinds = [e.kind for e in logger.entries_for_step(step)]
        assert kinds == ["llm_call", "action", "state_change"], (step, kinds)
    # State change on the first step reflects the env transition.
    sc = [e for e in logger.entries_for_step(0) if e.kind == "state_change"][0]
    assert sc.payload["before"] == {"x": 0}
    assert sc.payload["after"] == {"x": 1}


def test_run_logged_episode_keeps_partial_trace_on_failure():
    logger = AgentRunLogger(run_id="run2")
    with pytest.raises(RuntimeError, match="env exploded"):
        run_logged_episode(
            _CounterEnv(stop=5, fail_at=2), _StubAgent(), logger, seed=1, max_steps=10
        )

    trace = logger.trace()
    assert trace["status"] == "failed"
    assert "env exploded" in trace["error"]
    # Step 0 completed fully before the failure and is still queryable.
    assert [e.kind for e in logger.entries_for_step(0)] == [
        "llm_call", "action", "state_change",
    ]
    # The failing step recorded its llm_call before the env raised.
    assert any(e.kind == "llm_call" for e in logger.entries_for_step(1))


def test_run_logged_episode_respects_max_steps():
    logger = AgentRunLogger(run_id="run3")
    run_logged_episode(_CounterEnv(stop=100), _StubAgent(), logger, seed=0, max_steps=2)
    steps = {e.step for e in logger.entries if e.step is not None}
    assert steps == {0, 1}
    assert logger.trace()["status"] == "completed"
