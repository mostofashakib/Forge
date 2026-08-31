"""SQLite plumbing shared by the Harbor task services."""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any


class ToolError(RuntimeError):
    """A tool-level failure with a stable machine-readable code.

    The services raise this for every business-rule rejection so that both
    consumers see the same taxonomy: the CLI surfaces the message text, and
    the driver environments convert code/type into a deterministic ErrorState.
    """

    def __init__(self, error_code: str, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.error_type = error_type


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
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
