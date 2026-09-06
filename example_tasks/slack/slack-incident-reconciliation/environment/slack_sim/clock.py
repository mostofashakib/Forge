"""Persistent deterministic time for the Slack simulator.

The clock is a single counter in SQLite. A successful mutating tool call moves
it one step; a scheduled scenario event moves it to that event's own instant on
the seed calendar. Reads observe it without moving it, and rejected calls roll
back with the rest of their transaction, so the counter is a pure function of
the sequence of world mutations and released events. An agent that browses more
before acting still writes byte-identical timestamps.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

# The seeded workspace runs on a real calendar so that relative references in
# the messages ("Thursday", "yesterday", "last week") resolve the way a reader
# expects. Step 0 is Monday 2026-08-03 00:00:00 America/Los_Angeles (PDT, UTC-7)
# and one step is one second, so a step is simply "seconds into the seed
# calendar" and every seeded timestamp is an absolute wall-clock instant.
START_US = 1_785_740_400_000_000
STEP_US = 1_000_000

SECONDS_PER_DAY = 86_400


def moment(day: int, hour: int, minute: int = 0, second: int = 0) -> int:
    """The step for a wall-clock time, `day` days after the seed epoch.

    Day 0 is Monday 2026-08-03, so day 14 is Monday 2026-08-17 and day 16 is
    Wednesday 2026-08-19 -- the benchmark's current date.
    """
    if day < 0 or not (0 <= hour < 24) or not (0 <= minute < 60) or not (0 <= second < 60):
        raise ValueError(f"not a valid seed moment: day={day} {hour}:{minute}:{second}")
    return day * SECONDS_PER_DAY + hour * 3_600 + minute * 60 + second

_SUPPORTS_RETURNING = sqlite3.sqlite_version_info >= (3, 35, 0)


def slack_ts(microseconds: int) -> str:
    """Format virtual microseconds as Slack's fixed-width epoch-like ts."""
    seconds, remainder = divmod(microseconds, 1_000_000)
    return f"{seconds}.{remainder:06d}"


def parse_slack_ts(value: str) -> int:
    seconds, fraction = value.split(".", 1)
    return int(seconds) * 1_000_000 + int(fraction.ljust(6, "0")[:6])


@dataclass(frozen=True, slots=True)
class VirtualClock:
    start_us: int = START_US
    step_us: int = STEP_US

    def at(self, step: int) -> str:
        if step < 0:
            raise ValueError("Virtual-clock steps cannot be negative.")
        return slack_ts(self.start_us + step * self.step_us)

    def initialize(self, connection: sqlite3.Connection, step: int) -> None:
        connection.execute(
            "INSERT INTO virtual_clock (clock_id, current_us) VALUES (1, ?)",
            (self.start_us + step * self.step_us,),
        )

    def now(self, connection: sqlite3.Connection) -> str:
        """Read the current instant. Never moves the clock."""
        row = connection.execute(
            "SELECT current_us FROM virtual_clock WHERE clock_id = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("Virtual clock has not been initialized.")
        return slack_ts(int(row[0]))

    def advance(self, connection: sqlite3.Connection, steps: int = 1) -> str:
        """Move time forward and return the instant it now reads.

        Callers are mutating operations and scenario-event activation, each
        advancing exactly once. The `virtual_clock_monotonic` trigger rejects
        any non-increasing write, so the counter can never stall or run
        backwards.
        """
        if steps < 1:
            raise ValueError("Virtual clock must advance by at least one step.")
        delta = steps * self.step_us
        if _SUPPORTS_RETURNING:
            row = connection.execute(
                "UPDATE virtual_clock SET current_us = current_us + ? "
                "WHERE clock_id = 1 RETURNING current_us",
                (delta,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Virtual clock has not been initialized.")
            return slack_ts(int(row[0]))
        connection.execute(
            "UPDATE virtual_clock SET current_us = current_us + ? WHERE clock_id = 1",
            (delta,),
        )
        return self.now(connection)

    def advance_to(self, connection: sqlite3.Connection, step: int) -> str:
        """Advance to a deterministic scheduled instant, never backwards.

        Most actions advance by one second. A scenario event that declares a
        `scheduled_step` uses this method instead, so something the world does
        on its own clock lands on the fixture's virtual calendar rather than
        one second after whatever the agent last did.
        """
        target_us = self.start_us + step * self.step_us
        row = connection.execute(
            "SELECT current_us FROM virtual_clock WHERE clock_id = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("Virtual clock has not been initialized.")
        current_us = int(row[0])
        if target_us <= current_us:
            return self.advance(connection)
        connection.execute(
            "UPDATE virtual_clock SET current_us = ? WHERE clock_id = 1",
            (target_us,),
        )
        return slack_ts(target_us)


VIRTUAL_CLOCK = VirtualClock()
