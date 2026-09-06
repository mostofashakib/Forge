"""Persistent deterministic time for the task tracker.

The clock is a single counter in SQLite. A successful mutating tool call moves
it one step; a scheduled scenario event moves it to that event's own instant on
the seed calendar. Reads observe it without moving it, and rejected calls roll
back with the rest of their transaction, so the counter is a pure function of
the sequence of world mutations and released events. An agent that browses more
before acting still writes byte-identical timestamps.

Time is Unix milliseconds throughout, because that is what the tracker's own
`*_at_ms` columns have always held: the clock became authoritative without the
schema or any tool result changing shape.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

# Step 0 is 2023-11-14 22:13:20 UTC, the epoch the seed corpus was authored
# against, and one step is one second. Every seeded timestamp is therefore an
# absolute wall-clock instant rather than an offset nobody can read.
START_MS = 1_700_000_000_000
STEP_MS = 1_000
ONE_DAY_MS = 86_400_000

_SUPPORTS_RETURNING = sqlite3.sqlite_version_info >= (3, 35, 0)


def moment(day: int, hour: int = 0, minute: int = 0, second: int = 0) -> int:
    """The step for a wall-clock time, `day` days after the seed epoch."""
    if day < 0 or not (0 <= hour < 24) or not (0 <= minute < 60) or not (0 <= second < 60):
        raise ValueError(f"not a valid seed moment: day={day} {hour}:{minute}:{second}")
    return day * 86_400 + hour * 3_600 + minute * 60 + second


@dataclass(frozen=True, slots=True)
class VirtualClock:
    start_ms: int = START_MS
    step_ms: int = STEP_MS

    def at(self, step: int) -> int:
        if step < 0:
            raise ValueError("Virtual-clock steps cannot be negative.")
        return self.start_ms + step * self.step_ms

    def initialize(self, connection: sqlite3.Connection, step: int = 0) -> None:
        connection.execute(
            "INSERT INTO virtual_clock (clock_id, current_ms) VALUES (1, ?)",
            (self.at(step),),
        )

    def now(self, connection: sqlite3.Connection) -> int:
        """Read the current instant. Never moves the clock."""
        row = connection.execute(
            "SELECT current_ms FROM virtual_clock WHERE clock_id = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("Virtual clock has not been initialized.")
        return int(row[0])

    def advance(self, connection: sqlite3.Connection, steps: int = 1) -> int:
        """Move time forward and return the instant it now reads.

        Callers are mutating operations and scenario-event activation, each
        advancing exactly once. The `virtual_clock_monotonic` trigger rejects
        any non-increasing write, so the counter can never stall or run
        backwards.
        """
        if steps < 1:
            raise ValueError("Virtual clock must advance by at least one step.")
        delta = steps * self.step_ms
        if _SUPPORTS_RETURNING:
            row = connection.execute(
                "UPDATE virtual_clock SET current_ms = current_ms + ? "
                "WHERE clock_id = 1 RETURNING current_ms",
                (delta,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Virtual clock has not been initialized.")
            return int(row[0])
        connection.execute(
            "UPDATE virtual_clock SET current_ms = current_ms + ? WHERE clock_id = 1",
            (delta,),
        )
        return self.now(connection)

    def advance_to(self, connection: sqlite3.Connection, step: int) -> int:
        """Advance to a deterministic scheduled instant, never backwards.

        Most actions advance by one step. A scenario event that declares a
        `scheduled_step` uses this method instead, so something the world does
        on its own clock lands on the fixture's virtual calendar rather than one
        second after whatever the agent last did.
        """
        target_ms = self.at(step)
        current_ms = self.now(connection)
        if target_ms <= current_ms:
            return self.advance(connection)
        connection.execute(
            "UPDATE virtual_clock SET current_ms = ? WHERE clock_id = 1",
            (target_ms,),
        )
        return target_ms

    def action_instant(self, connection: sqlite3.Connection) -> int:
        """The instant this tool call writes at, advancing once per call.

        One action is one point in time however many rows it touches, so the
        first mutation in a call moves the clock and every later write in the
        same call reuses the value cached on the connection.
        """
        cached = getattr(connection, "action_ms", None)
        if cached is not None:
            return int(cached)
        instant = self.advance(connection)
        try:
            connection.action_ms = instant  # type: ignore[attr-defined]
        except AttributeError:
            pass
        return instant


VIRTUAL_CLOCK = VirtualClock()
