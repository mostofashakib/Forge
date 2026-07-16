from __future__ import annotations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from forge.settings import database_url

_engine = None
_SessionLocal = None


class Base(DeclarativeBase):
    pass


def get_engine():
    global _engine
    if _engine is None:
        url = database_url()
        _engine = create_engine(url, connect_args={"check_same_thread": False})
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def init_db() -> None:
    from backend.app import models
    _ = models
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        existing_columns = {
            column["name"]
            for column in inspect(conn).get_columns("sandbox_environments")
        }
        migrations = {
            "env_type": "ALTER TABLE sandbox_environments ADD COLUMN env_type TEXT DEFAULT 'general'",
            "state_schema": "ALTER TABLE sandbox_environments ADD COLUMN state_schema TEXT",
            "validation_missing_fields": (
                "ALTER TABLE sandbox_environments "
                "ADD COLUMN validation_missing_fields TEXT"
            ),
        }
        for column_name, statement in migrations.items():
            if column_name not in existing_columns:
                conn.execute(text(statement))
        conn.commit()


def get_db():
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
