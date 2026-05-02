from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

_engine = None
_SessionLocal = None


class Base(DeclarativeBase):
    pass


def get_engine():
    global _engine
    if _engine is None:
        url = os.environ.get("FORGE_DB_URL", "sqlite:///./forge.db")
        _engine = create_engine(url, connect_args={"check_same_thread": False})
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def _try_alter(conn, sql: str) -> None:
    try:
        conn.execute(__import__("sqlalchemy").text(sql))
        conn.commit()
    except Exception:
        pass  # column already exists — safe to ignore


def init_db() -> None:
    from backend.app import models  # noqa: F401
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        _try_alter(conn, "ALTER TABLE sandbox_environments ADD COLUMN env_type TEXT DEFAULT 'general'")


def get_db():
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
