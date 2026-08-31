"""Deterministic Slack environment backed by the SQLite service.

All state lives in the service's SQLite database; this class only adapts the
service to the BaseEnvironment interface for the simulation driver.
"""

from __future__ import annotations

from fleet.environments import sqlite_environment
from fleet.environments.slack import sqlite_service


class SlackEnvironment(sqlite_environment.SqliteBackedEnvironment):
    environment_name = "slack"
    db_filename = "slack.db"
    tool_names = sqlite_service.TOOL_NAMES

    _seed = staticmethod(sqlite_service.seed_database)
    _export = staticmethod(sqlite_service.export_state)
    _execute = staticmethod(sqlite_service.execute_tool)
