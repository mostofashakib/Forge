"""SQLite plumbing shared by the Harbor task services."""

from __future__ import annotations

import os
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class WorldError(RuntimeError):
    """Anything the Slack world raises on purpose, with a code attached.

    The socket server has exactly one way to describe a failure to a caller --
    `{"code": ..., "type": ..., "message": ...}` -- so the code and the type
    live on the exception rather than at the place that catches it. Every
    subclass therefore answers `error_code` and `error_type` as class
    attributes, and the server maps any `WorldError` uniformly.

    What this replaces is a single `except Exception` that reported everything
    as `internal_error` with a stringified class name. A locked database, a
    mistyped tool name and a real bug arrived identically, which meant the one
    question a reader of a failed call actually has -- is this my fault, the
    world's, or the harness's -- could not be answered from the response.
    """

    error_code = "world_error"
    error_type = "internal_error"


class ToolError(WorldError):
    """A tool-level failure with a stable machine-readable code.

    The services raise this for every business-rule rejection so that both
    consumers see the same taxonomy: the CLI surfaces the message text, and
    the driver environments convert code/type into a deterministic ErrorState.

    This is the one branch of the family that is *about the request*. An agent
    that receives it learns something actionable: ask differently. That is why
    the storage and internal branches below are deliberately not subclasses --
    code that means to forgive a rule violation must not also swallow a corrupt
    database.
    """

    error_code = "tool_error"
    error_type = "validation_error"

    def __init__(self, error_code: str, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.error_type = error_type


class UnknownToolError(ToolError):
    """A tool name that is not in the surface.

    Raised from the handler map in `service.execute_tool` and reported by the
    server's own name check. Both say `unknown_tool`: an agent that trips the
    one and then the other is making a single mistake and should be told so
    once, in one vocabulary.
    """

    error_code = "unknown_tool"
    error_type = "validation_error"

    def __init__(self, tool_name: str) -> None:
        super().__init__(self.error_code, self.error_type, f"Unknown tool: {tool_name}")
        self.tool_name = tool_name


class StorageError(WorldError):
    """SQLite could not do its job: the file is locked, corrupt or unreachable.

    Not a `ToolError`, because nothing the caller asks for differently will
    help. Kept distinct from `InternalError` because the remedy is different
    too -- contention and a damaged file are operational conditions with
    obvious next steps, where an unexpected exception is a bug to be read.
    """

    error_code = "storage_error"
    error_type = "internal_error"


class InternalError(WorldError):
    """The residue: something went wrong that no other class describes.

    Every one of these is a defect. The class exists so that the catch-all has
    a name and a code of its own rather than being the default destination for
    failures that were always distinguishable.
    """

    error_code = "internal_error"
    error_type = "internal_error"


@contextmanager
def storage_errors() -> Iterator[None]:
    """Translate SQLite's exceptions into `StorageError` at one boundary.

    Wrapping here rather than at every cursor keeps the handlers readable and,
    more importantly, keeps the translation honest: a `sqlite3.IntegrityError`
    from a constraint the schema declares is still a storage fault, not a rule
    the agent broke -- the rules are enforced in Python, above this line.
    """
    try:
        yield
    except sqlite3.Error as exc:
        raise StorageError(f"{type(exc).__name__}: {exc}") from exc


# A contended writer waits this long for the lock instead of failing outright,
# which is what turned incidental lock contention into flaky "database is
# locked" errors.
BUSY_TIMEOUT_SECONDS = 5.0


class SimConnection(sqlite3.Connection):
    """A connection scoped to exactly one tool call.

    Handlers open one of these per call, so it is the natural place to cache
    that call's action instant: the first mutation advances the virtual clock
    and every later write in the same call reuses the value. One action is one
    point in time, however many rows it touches.

    Closing on exit matters as much as committing. The base class commits or
    rolls back but leaves the handle open, so every tool call used to leak a
    connection that went on holding SQLite locks until garbage collection.
    """

    action_ts: str | None = None

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def connect(db_path: Path) -> SimConnection:
    connection = sqlite3.connect(
        db_path, timeout=BUSY_TIMEOUT_SECONDS, factory=SimConnection
    )
    connection.row_factory = sqlite3.Row
    return connection


def query_rows(connection: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(sql, params).fetchall()]


def remove_pycaches() -> None:
    """Drop bytecode caches so container resets leave no stray filesystem state."""
    root = Path("/app")
    if not root.is_dir():
        root = Path(__file__).resolve().parents[2]
    for r, dirs, files in os.walk(root, topdown=False):
        for name in dirs:
            if name == "__pycache__":
                try:
                    shutil.rmtree(os.path.join(r, name))
                except Exception:
                    pass
        for name in files:
            if name.endswith((".pyc", ".pyo")):
                try:
                    os.unlink(os.path.join(r, name))
                except Exception:
                    pass
