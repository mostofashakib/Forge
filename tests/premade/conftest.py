from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from sqlalchemy import event

PREMADE_ROOT = Path(__file__).resolve().parents[2] / "docker" / "premade"


@pytest.fixture
def load_premade(tmp_path, monkeypatch):
    """Import a premade app fresh, with its SQLite file inside tmp_path."""
    loaded = []

    def _load(name: str):
        monkeypatch.chdir(tmp_path)
        spec = importlib.util.spec_from_file_location(
            f"premade_{name}_{tmp_path.name}", PREMADE_ROOT / name / "app.py"
        )
        module = importlib.util.module_from_spec(spec)
        # Registered so pydantic can resolve the app's postponed annotations.
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
        return module

    yield _load
    for module in loaded:
        module.engine.dispose()


def count_selects(engine, action) -> int:
    selects: list[str] = []

    def _record(_conn, _cursor, statement, *_args):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    event.listen(engine, "before_cursor_execute", _record)
    try:
        action()
    finally:
        event.remove(engine, "before_cursor_execute", _record)
    return len(selects)
