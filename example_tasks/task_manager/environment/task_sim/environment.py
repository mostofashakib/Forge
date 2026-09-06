"""RL-facing lifecycle and state contract for the task-tracker world."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from task_sim.clock import VIRTUAL_CLOCK
from task_sim.identity import LOGGED_IN_USER, TrackerUserCredentials
from task_sim.progress import workspace_progress
from task_sim.prompts import PromptInterface, get_prompt_interface
from task_sim.scenario import Scenario
from task_sim.seed import REASSIGNMENT_SCENARIO
from task_sim.service import (
    DEFAULT_DB_PATH,
    DEFAULT_SNAPSHOT_PATH,
    execute_tool,
    export_state,
    seed_database,
)
from task_sim.sqlite_common import ToolError, connect, storage_errors
from task_sim.tool_definitions import get_tool_definitions

CONTRACT_VERSION = "7.0"
DEFAULT_MAX_TURNS = 100
DEFAULT_PROMPT_ID = "tracker_operator"
SESSION_COOKIE_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class TaskHandoverEnvironment:
    """
    Harbor supplies isolation at the container level. This adapter supplies the
    model-world contract within that container: lifecycle, prompt, tools, opaque
    session identity, tool transitions, and durable turn history.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        snapshot_path: str | Path = DEFAULT_SNAPSHOT_PATH,
        prompt_interface: PromptInterface | None = None,
        credentials: TrackerUserCredentials = LOGGED_IN_USER,
        scenario: Scenario = REASSIGNMENT_SCENARIO,
    ) -> None:
        self.db_path = Path(db_path)
        self.snapshot_path = Path(snapshot_path)
        self.prompt_interface = prompt_interface or get_prompt_interface()
        self.credentials = credentials
        # Which task this environment is hosting. The world is the same either
        # way; hand it a different Scenario and it hosts a different one.
        self.scenario = scenario

    def setup(self) -> dict[str, Any]:
        """Initialize durable storage once without erasing an existing episode."""
        if not self.db_path.exists():
            self.setup_state(seed=0)
        self._ensure_session_schema()
        return {
            "contract_version": CONTRACT_VERSION,
            "persistent": True,
            "prompt_count": len(self.prompts()),
            "tool_count": len(get_tool_definitions()),
        }

    def setup_state(self, seed: int = 0) -> None:
        """Drop the workspace and rebuild it, so every episode starts identical.

        `seed` selects a variant of the initial state. This scenario is fully
        deterministic and defines only variant 0; a scenario with randomised
        fixtures would branch here.
        """
        if seed != 0:
            raise ValueError(f"scenario defines only deterministic seed 0, not {seed}")
        seed_database(self.db_path, self.snapshot_path, self.scenario)
        self._ensure_session_schema()

    def reset(
        self,
        *,
        session_cookie: str = "default",
        seed: int = 0,
        user_instruction: str = "",
        prompt_id: str = DEFAULT_PROMPT_ID,
        prompt_context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> dict[str, Any]:
        """Start a fresh episode, then create its first model session."""
        self.setup_state(seed=seed)
        session = self.seed_session(
            session_cookie=session_cookie,
            user_instruction=user_instruction,
            prompt_id=prompt_id,
            prompt_context=prompt_context,
            metadata=metadata,
            max_turns=max_turns,
        )
        return {
            "episode": {"seed": seed, "state": "ready", "max_turns": max_turns},
            "session": session,
            "prompt": self.render_prompt(prompt_id, user_instruction, prompt_context),
            "observation": {"workspace": "tasks", "actor_id": self.credentials.user_id},
        }

    def seed_session(
        self,
        *,
        session_cookie: str = "default",
        user_instruction: str = "",
        prompt_id: str = DEFAULT_PROMPT_ID,
        prompt_context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> dict[str, Any]:
        """Create/reset conversation state and return its opaque cookie."""
        if max_turns < 1:
            raise ValueError("An episode must allow at least one turn.")
        self.setup()
        self._validate_session_cookie(session_cookie)
        prompt = self.render_prompt(prompt_id, user_instruction, prompt_context)
        actor_id = self.credentials.user_id
        with connect(self.db_path) as connection:
            if connection.execute(
                "SELECT 1 FROM users WHERE user_id = ?", (actor_id,)
            ).fetchone() is None:
                raise ValueError(f"Unknown actor_id: {actor_id}")
            created_ms = VIRTUAL_CLOCK.now(connection)
            connection.execute(
                "DELETE FROM conversation_history WHERE session_cookie = ?", (session_cookie,)
            )
            connection.execute(
                """
                INSERT INTO rl_sessions (
                    session_cookie, actor_id, created_ms, turn_count,
                    prompt_id, prompt_context_json, user_instruction, metadata_json,
                    max_turns, last_progress, finished
                ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, 0.0, 0)
                ON CONFLICT(session_cookie) DO UPDATE SET
                    actor_id = excluded.actor_id,
                    created_ms = excluded.created_ms,
                    turn_count = 0,
                    prompt_id = excluded.prompt_id,
                    prompt_context_json = excluded.prompt_context_json,
                    user_instruction = excluded.user_instruction,
                    metadata_json = excluded.metadata_json,
                    max_turns = excluded.max_turns,
                    last_progress = 0.0,
                    finished = 0
                """,
                (
                    session_cookie,
                    actor_id,
                    created_ms,
                    prompt_id,
                    json.dumps(prompt_context or {}, sort_keys=True),
                    user_instruction,
                    json.dumps(metadata or {}, sort_keys=True),
                    max_turns,
                ),
            )
            self._append_history(
                connection, session_cookie, 0, "system", "message", None, prompt["system"]
            )
            self._append_history(
                connection, session_cookie, 0, "user", "message", None, user_instruction
            )
            connection.commit()
        return {
            "session_cookie": session_cookie,
            "actor_id": actor_id,
            "turn": 0,
            "max_turns": max_turns,
        }

    def prompts(self) -> list[dict[str, str]]:
        return self.prompt_interface.list()

    def render_prompt(
        self,
        prompt_id: str,
        user_instruction: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Render one named prompt and attach the current tool contract."""
        prompt = self.prompt_interface.render(
            prompt_id, user_instruction=user_instruction, context=context
        )
        prompt["tools"] = get_tool_definitions()
        return prompt

    def tool_definitions(self) -> list[dict[str, Any]]:
        return get_tool_definitions()

    def step(
        self, session_cookie: str, tool_name: str, input_payload: dict[str, Any]
    ) -> dict[str, Any]:
        """One action. Storage faults stop the rollout; tool errors do not.

        The distinction is the whole point of separating the two: a rule the
        policy broke is an observation it can learn from and comes back inside
        the step, while a corrupt or locked database is a condition no action
        can improve. Reporting the second as an error observation would train
        the policy to avoid a tool that was never the problem.

        The boundary is here rather than around `execute_tool` alone because
        this method writes its own history rows, and a step that recorded the
        action and then lost the database has still failed for a storage reason.
        """
        with storage_errors():
            return self._step(session_cookie, tool_name, input_payload)

    def _step(
        self, session_cookie: str, tool_name: str, input_payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply one model action and persist both sides of the transition.

        Returns the observation plus the RL signals a trainer needs: `reward` is
        the change in progress caused by this action, `progress` is the absolute
        score so far, and `terminated`/`truncated` say whether the rollout is
        over by goal or by budget.
        """
        session = self._get_session(session_cookie)
        max_turns = int(session["max_turns"])
        turn = int(session["turn_count"]) + 1
        previous_progress = float(session["last_progress"])

        if int(session["finished"]):
            # Acting after a rollout ends is a trainer bug, not a tool failure.
            # Reported in the same shape as any other step: a caller that reads
            # `side_effects` or `milestones_met` every turn must not crash on
            # the one turn that tells it the rollout is over.
            scored = workspace_progress(export_state(self.db_path))
            return {
                "session_cookie": session_cookie,
                "turn": int(session["turn_count"]),
                "ok": False,
                "error": {
                    "code": "episode_finished",
                    "type": "invalid_state",
                    "message": "This episode has already terminated; reset before stepping again.",
                },
                "reward": 0.0,
                "progress": previous_progress,
                "milestones_met": scored["milestones_met"],
                "side_effects": scored["side_effects"],
                "penalty_total": scored["penalty_total"],
                "terminated": int(session["turn_count"]) < max_turns,
                "truncated": int(session["turn_count"]) >= max_turns,
                "done": True,
                "turns_remaining": 0,
            }

        with connect(self.db_path) as connection:
            self._append_history(
                connection, session_cookie, turn, "assistant", "tool_call",
                tool_name, input_payload,
            )
            connection.commit()
        try:
            result = execute_tool(
                self.db_path, tool_name, input_payload, actor_id=str(session["actor_id"])
            )
            observation: dict[str, Any] = {"ok": True, "result": result}
        except ToolError as exc:
            observation = {
                "ok": False,
                "error": {
                    "code": exc.error_code,
                    "type": exc.error_type,
                    "message": str(exc),
                },
            }
        with connect(self.db_path) as connection:
            self._append_history(
                connection, session_cookie, turn, "tool", "tool_result",
                tool_name, observation,
            )
            connection.commit()

        # Score after the action, and only on what the world can observe: which
        # events the agent's work released, and whether a report was filed at
        # all. Whether the reported set is *right* is the verifier's call --
        # five ids are a short string a policy could assert without reading
        # anything, so shaping on their correctness would reward the assertion
        # rather than the work behind it.
        scored = workspace_progress(export_state(self.db_path))
        progress = float(scored["progress"])
        reward = round(progress - previous_progress, 6)
        terminated = bool(scored["complete"])
        truncated = (not terminated) and turn >= max_turns

        with connect(self.db_path) as connection:
            connection.execute(
                "UPDATE rl_sessions SET turn_count = ?, last_progress = ?, finished = ? "
                "WHERE session_cookie = ?",
                (turn, progress, int(terminated or truncated), session_cookie),
            )
            connection.commit()

        return {
            "session_cookie": session_cookie,
            "turn": turn,
            **observation,
            "reward": reward,
            "progress": progress,
            "milestones_met": scored["milestones_met"],
            "side_effects": scored["side_effects"],
            "penalty_total": scored["penalty_total"],
            "terminated": terminated,
            "truncated": truncated,
            "done": terminated or truncated,
            "turns_remaining": max(0, max_turns - turn),
        }

    def history(self, session_cookie: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Turn history, oldest first.

        `limit` returns only the most recent `limit` events, still in order. A
        long rollout's transcript grows without bound, so any caller reading it
        every turn should pass one.
        """
        self._get_session(session_cookie)
        if limit is not None and limit < 0:
            raise ValueError("history limit cannot be negative.")
        with connect(self.db_path) as connection:
            if limit is None:
                rows = connection.execute(
                    "SELECT event_ms, turn_index, role, kind, name, content_json "
                    "FROM conversation_history WHERE session_cookie = ? ORDER BY event_id",
                    (session_cookie,),
                ).fetchall()
            else:
                # Take the newest `limit` rows, then restore chronological order.
                rows = list(reversed(connection.execute(
                    "SELECT event_ms, turn_index, role, kind, name, content_json "
                    "FROM conversation_history WHERE session_cookie = ? "
                    "ORDER BY event_id DESC LIMIT ?",
                    (session_cookie, limit),
                ).fetchall()))
        return [
            {
                "turn": row["turn_index"],
                "ms": row["event_ms"],
                "role": row["role"],
                "kind": row["kind"],
                "name": row["name"],
                "content": json.loads(row["content_json"]),
            }
            for row in rows
        ]

    def history_count(self, session_cookie: str) -> int:
        with connect(self.db_path) as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM conversation_history WHERE session_cookie = ?",
                (session_cookie,),
            ).fetchone()[0])

    def state(self, session_cookie: str, history_limit: int | None = None) -> dict[str, Any]:
        """Model-visible session state and previously observed history.

        `history_limit` windows the transcript to its most recent events;
        `history_total` always reports the full count so a caller can see what
        was elided. Hidden workspace tables are intentionally absent: the only
        tracker data in this response is data already returned by tool calls in
        this session's history. Verifiers use the protected admin interface when
        they need authoritative world state.
        """
        session = self._get_session(session_cookie)
        return {
            "session": {
                "session_cookie": session["session_cookie"],
                "actor_id": session["actor_id"],
                "turn": session["turn_count"],
                "max_turns": session["max_turns"],
                "progress": session["last_progress"],
                "done": bool(session["finished"]),
                "prompt_id": session["prompt_id"],
                "prompt_context": json.loads(session["prompt_context_json"]),
                "metadata": json.loads(session["metadata_json"]),
            },
            "history": self.history(session_cookie, limit=history_limit),
            "history_total": self.history_count(session_cookie),
        }

    def _ensure_session_schema(self) -> None:
        with connect(self.db_path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS rl_sessions (
                    session_cookie      TEXT PRIMARY KEY,
                    actor_id            TEXT NOT NULL,
                    created_ms          INTEGER NOT NULL,
                    turn_count          INTEGER NOT NULL,
                    prompt_id           TEXT NOT NULL,
                    prompt_context_json TEXT NOT NULL,
                    user_instruction    TEXT NOT NULL,
                    metadata_json       TEXT NOT NULL,
                    max_turns           INTEGER NOT NULL DEFAULT 100,
                    last_progress       REAL NOT NULL DEFAULT 0.0,
                    finished            INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS conversation_history (
                    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_ms       INTEGER NOT NULL,
                    session_cookie TEXT NOT NULL,
                    turn_index     INTEGER NOT NULL,
                    role           TEXT NOT NULL,
                    kind           TEXT NOT NULL,
                    name           TEXT,
                    content_json   TEXT NOT NULL,
                    FOREIGN KEY (session_cookie) REFERENCES rl_sessions(session_cookie)
                        ON DELETE CASCADE
                );
                """
            )
            connection.commit()

    def _get_session(self, session_cookie: str) -> sqlite3.Row:
        self.setup()
        self._validate_session_cookie(session_cookie)
        with connect(self.db_path) as connection:
            session = connection.execute(
                "SELECT * FROM rl_sessions WHERE session_cookie = ?", (session_cookie,)
            ).fetchone()
        if session is None:
            raise KeyError(f"Unknown session cookie: {session_cookie}")
        return session

    @staticmethod
    def _validate_session_cookie(session_cookie: str) -> None:
        if not SESSION_COOKIE_PATTERN.fullmatch(session_cookie):
            raise ValueError("Session cookie must match [A-Za-z0-9._-]{1,128}.")

    @staticmethod
    def _append_history(
        connection: sqlite3.Connection,
        session_cookie: str,
        turn: int,
        role: str,
        kind: str,
        name: str | None,
        content: Any,
    ) -> None:
        # Transcript bookkeeping observes world time without moving it; the row's
        # position in the conversation comes from event_id, not from the clock.
        # Ticking here is what would let read-only tool calls advance the world
        # and perturb the timestamps of later mutations.
        event_ms = VIRTUAL_CLOCK.now(connection)
        connection.execute(
            "INSERT INTO conversation_history ("
            "  event_ms, session_cookie, turn_index, role, kind, name, content_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event_ms, session_cookie, turn, role, kind, name,
                json.dumps(content, sort_keys=True),
            ),
        )
