from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from backend.app.database import Base
from backend.app.models import SandboxEnvironment


def test_sandbox_environment_create_and_query():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        sandbox = SandboxEnvironment(
            id="ticket_env",
            status="building",
            ttl_days=30,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
            policy_requirements="no deletes",
            reward_requirements=None,
        )
        db.add(sandbox)
        db.commit()
        fetched = db.get(SandboxEnvironment, "ticket_env")
        assert fetched.status == "building"
        assert fetched.ttl_days == 30
        assert fetched.policy_requirements == "no deletes"
        assert fetched.reward_requirements is None
        assert fetched.container_id is None
        assert fetched.container_port is None
        # Negative: querying an id that was never inserted returns nothing,
        # rather than silently resolving to the row above.
        assert not db.get(SandboxEnvironment, "unknown_env")
        assert "unknown_env" not in {row.id for row in db.query(SandboxEnvironment).all()}
    Base.metadata.drop_all(engine)
