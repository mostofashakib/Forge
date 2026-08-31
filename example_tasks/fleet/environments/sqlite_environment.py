"""Driver-environment adapter over the SQLite services.

SQLite is the single source of truth for environment state: the seed
generators write rows into the database and every tool mutates it directly.
This adapter gives the simulation driver the BaseEnvironment interface
(instrumentation, reset and determinism checks, Observations) on top of a
service-owned temporary database file, so the driver exercises exactly the
same storage and tool semantics as the Harbor containers.
"""

from __future__ import annotations

import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Any, Callable, ClassVar

from fleet.core.base import BaseEnvironment, BaseTool
from fleet.core.models import Action
from fleet.environments.sqlite_common import ToolError


class ServiceTool(BaseTool):
    """Delegates one named tool to the environment's SQLite service."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, environment: "SqliteBackedEnvironment", action: Action) -> dict[str, Any]:
        return environment.call_service(self.name, action)


class SqliteBackedEnvironment(BaseEnvironment):
    """Base for environments whose state lives in a SQLite database file.

    Subclasses bind the service module's functions as staticmethods:
    ``_seed(db_path, snapshot_path)``, ``_export(db_path)`` and
    ``_execute(db_path, tool_name, input_payload, actor_id)``.
    """

    db_filename: ClassVar[str]
    tool_names: ClassVar[tuple[str, ...]]
    _seed: ClassVar[Callable[[Path, Path], None]]
    _export: ClassVar[Callable[[Path], dict[str, Any]]]
    _execute: ClassVar[Callable[[Path, str, dict[str, Any], str], dict[str, Any]]]

    def _build_initial_state(self) -> Path:
        workspace = Path(tempfile.mkdtemp(prefix=f"fleet-{self.environment_name}-"))
        weakref.finalize(self, shutil.rmtree, str(workspace), ignore_errors=True)
        self._snapshot_path = workspace / "seed_snapshot.sql"
        db_path = workspace / self.db_filename
        self._seed(db_path, self._snapshot_path)
        return db_path

    def _restore_initial_state(self) -> None:
        self._seed(self._initial_state, self._snapshot_path)
        self.state = self._initial_state

    def _register_tools(self) -> None:
        for name in self.tool_names:
            self.register_tool(ServiceTool(name))

    def export_state(self) -> dict[str, Any]:
        return self._export(self.state)

    def call_service(self, tool_name: str, action: Action) -> dict[str, Any]:
        try:
            return self._execute(self.state, tool_name, action.input_payload, action.acting_user_id)
        except ToolError as exc:
            self.fail(exc.error_code, exc.error_type, str(exc), tool_name, action.input_payload)
