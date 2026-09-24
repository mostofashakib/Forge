from __future__ import annotations

from sqlalchemy import text

from backend.app import database


def _fresh_engine(monkeypatch, url: str):
    monkeypatch.setenv("FORGE_DB_URL", url)
    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_SessionLocal", None)
    return database.get_engine()


def test_sqlite_engine_uses_wal_and_waits_for_locks(tmp_path, monkeypatch):
    engine = _fresh_engine(monkeypatch, f"sqlite:///{tmp_path}/forge.db")
    try:
        with engine.connect() as conn:
            assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
            assert conn.execute(text("PRAGMA busy_timeout")).scalar() >= 5_000
    finally:
        engine.dispose()



def test_in_memory_database_is_not_forced_into_wal(monkeypatch):
    engine = _fresh_engine(monkeypatch, "sqlite:///:memory:")
    try:
        with engine.connect() as conn:
            assert conn.execute(text("PRAGMA journal_mode")).scalar() != "wal"
            assert conn.execute(text("SELECT 1")).scalar() == 1
    finally:
        engine.dispose()
