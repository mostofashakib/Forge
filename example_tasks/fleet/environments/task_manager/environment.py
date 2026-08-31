"""Deterministic task manager environment backed by the SQLite service.

All state lives in the service's SQLite database; this class only adapts the
service to the BaseEnvironment interface for the simulation driver.
"""

from __future__ import annotations

from fleet.environments import sqlite_environment
from fleet.environments.task_manager import sqlite_service


class TaskManagerEnvironment(sqlite_environment.SqliteBackedEnvironment):
    environment_name = "task_manager"
    db_filename = "task_manager.db"
    tool_names = sqlite_service.TOOL_NAMES

    _seed = staticmethod(sqlite_service.seed_database)
    _export = staticmethod(sqlite_service.export_state)
    _execute = staticmethod(sqlite_service.execute_tool)
