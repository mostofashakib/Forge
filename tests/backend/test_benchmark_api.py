import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from backend.app.database import Base
from backend.app.models import BenchmarkRun


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def test_benchmark_run_model(db):
    run = BenchmarkRun(
        id="bm_test0001",
        status="queued",
        domains="email,project_mgmt",
        depth=5,
        seeds=5,
        output_dir="benchmark_results",
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    fetched = db.get(BenchmarkRun, "bm_test0001")
    assert fetched.status == "queued"
    assert fetched.domains == "email,project_mgmt"
    assert fetched.completed_at is None
    assert fetched.error is None
    assert fetched.report_json is None
