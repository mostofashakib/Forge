"""The Forge protocol every premade app serves, and the deterministic ledger behind it.

Episodes must replay identically, so an app never reads the wall clock or
mints random ids. Time is a counter that moves one second per event, and ids
count up per prefix. Both live in SQLite, so they reset, snapshot, and
survive a restart together with the rows they describe.

The build copies this file next to each app, so apps import it as a sibling
module. Tables are declared on the app's own Base to keep each app's schema
in its own metadata.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import Session

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
BASELINE_SLOT = "baseline"


@dataclass(frozen=True)
class ProtocolTables:
    saved_state: type
    action_log: type
    counter: type


def protocol_tables(base: type) -> ProtocolTables:
    """Declare the snapshot, action log, and counter tables on an app's Base."""

    class SavedState(base):
        __tablename__ = "saved_states"
        slot = Column(String, primary_key=True)
        data = Column(Text, nullable=False)  # JSON blob

    class ActionLog(base):
        __tablename__ = "action_log"
        id = Column(String, primary_key=True)
        action_type = Column(String, nullable=False)
        target_id = Column(String, nullable=True)
        payload = Column(Text, default="{}")
        timestamp = Column(String, nullable=False)

    class ForgeCounter(base):
        """The virtual clock and the id counters, stored with the data they number."""
        __tablename__ = "forge_counters"
        name = Column(String, primary_key=True)
        value = Column(Integer, nullable=False)

    return ProtocolTables(saved_state=SavedState, action_log=ActionLog, counter=ForgeCounter)


class Ledger:
    """Virtual time, ids, and the action log, all read from the app's database."""

    def __init__(self, tables: ProtocolTables, epoch: datetime) -> None:
        self._tables = tables
        self._epoch = epoch

    def bump(self, db: Session, name: str) -> int:
        counter = db.get(self._tables.counter, name)
        if counter is None:
            counter = self._tables.counter(name=name, value=0)
            db.add(counter)
            db.flush()  # later lookups in this session must find it
        counter.value += 1
        return counter.value

    def elapsed(self, db: Session) -> int:
        """Seconds of virtual time so far, without advancing the clock."""
        counter = db.get(self._tables.counter, "clock")
        return counter.value if counter is not None else 0

    def at(self, seconds: int) -> str:
        return (self._epoch + timedelta(seconds=seconds)).strftime(TIMESTAMP_FORMAT)

    def now(self, db: Session) -> str:
        return self.at(self.bump(db, "clock"))

    def next_id(self, db: Session, prefix: str) -> str:
        # The underscore keeps minted ids apart from seed ids like "e001".
        return f"{prefix}_{self.bump(db, f'id:{prefix}'):04d}"

    def log_action(self, db: Session, action_type: str, target_id: str | None = None, payload: dict | None = None) -> None:
        db.add(self._tables.action_log(
            id=self.next_id(db, "a"),
            action_type=action_type,
            target_id=target_id,
            payload=json.dumps(payload or {}),
            timestamp=self.now(db),
        ))


def dump_ledger(db: Session, tables: ProtocolTables) -> dict:
    """The action log and counters, in the shape `restore_ledger` takes."""
    log = tables.action_log
    return {
        "action_log": [
            {"id": a.id, "action_type": a.action_type, "target_id": a.target_id,
             "payload": a.payload, "timestamp": a.timestamp}
            for a in db.query(log).order_by(log.timestamp).all()
        ],
        "forge_counters": {
            c.name: c.value for c in db.query(tables.counter).order_by(tables.counter.name).all()
        },
    }


def restore_ledger(db: Session, tables: ProtocolTables, data: dict) -> None:
    db.query(tables.action_log).delete()
    # A snapshot carries the clock and counters, so restoring it rewinds them
    # too. State JSON without them leaves them running, which keeps new ids
    # clear of the rows already there.
    if "forge_counters" in data:
        db.query(tables.counter).delete()
        for name, value in data["forge_counters"].items():
            db.add(tables.counter(name=name, value=value))
    for a in data.get("action_log", []):
        db.add(tables.action_log(
            id=a["id"], action_type=a["action_type"], target_id=a.get("target_id"),
            payload=a.get("payload", "{}"), timestamp=a["timestamp"],
        ))


def save_snapshot(db: Session, tables: ProtocolTables, slot: str, data: dict) -> None:
    saved = db.get(tables.saved_state, slot)
    if saved:
        saved.data = json.dumps(data)
    else:
        db.add(tables.saved_state(slot=slot, data=json.dumps(data)))
    db.commit()


class SnapshotRequest(BaseModel):
    slot: str


def forge_router(
    session_factory: Callable[[], Session],
    tables: ProtocolTables,
    *,
    state: Callable[[Session], dict],
    dump: Callable[[Session], dict],
    restore: Callable[[Session, dict], None],
    domain_tables: Sequence[type],
    seed: Callable[[], None],
) -> APIRouter:
    """The /forge/* routes. Reset wipes `domain_tables` and the ledger, then reseeds."""
    router = APIRouter(prefix="/forge")

    @router.get("/health")
    def health():
        return {"status": "ok"}

    @router.get("/state")
    def forge_state():
        with session_factory() as db:
            return state(db)

    @router.get("/dump")
    def forge_dump():
        """The full restorable state, in the shape /forge/restore-state takes."""
        with session_factory() as db:
            return dump(db)

    @router.post("/reset")
    def forge_reset():
        with session_factory() as db:
            for table in (*domain_tables, tables.action_log, tables.saved_state, tables.counter):
                db.query(table).delete()
            db.commit()
        seed()
        with session_factory() as db:
            return {"status": "reset", "state": state(db)}

    @router.post("/snapshot")
    def forge_snapshot(req: SnapshotRequest):
        with session_factory() as db:
            save_snapshot(db, tables, req.slot, dump(db))
        return {"status": "snapshot_saved", "slot": req.slot}

    @router.post("/restore/{slot}")
    def forge_restore(slot: str):
        with session_factory() as db:
            saved = db.get(tables.saved_state, slot)
            if not saved:
                raise HTTPException(status_code=404, detail=f"Slot '{slot}' not found")
            restore(db, json.loads(saved.data))
            return {"status": "restored", "slot": slot, "state": state(db)}

    @router.post("/restore-state")
    def forge_restore_state(data: dict):
        with session_factory() as db:
            restore(db, data)
            return {"status": "restored", "state": state(db)}

    return router
