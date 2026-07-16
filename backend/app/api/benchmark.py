from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import BenchmarkRun
from forge.paths import confined_relative_path
from forge.settings import redis_url

logger = logging.getLogger(__name__)

# Lazy module-level reference so tests can patch backend.app.api.benchmark.run_benchmark_task.
# The actual import is deferred to avoid circular-import issues at package load time.
try:
    from backend.app.worker.tasks import run_benchmark_task  # noqa: F401
except Exception:  # pragma: no cover
    run_benchmark_task = None  # type: ignore[assignment]
router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


class CreateBenchmarkRunRequest(BaseModel):
    domains: list[str] = Field(default_factory=lambda: ["email", "project_mgmt"], min_length=1)
    depth: int = Field(default=5, ge=1, le=5)
    seeds: int = Field(default=5, ge=1, le=100)
    output_dir: str = Field(default="benchmark_results", min_length=1, max_length=255)


@router.post("/runs", status_code=202)
def create_benchmark_run(body: CreateBenchmarkRunRequest, db: Session = Depends(get_db)):
    try:
        output_dir = confined_relative_path(Path.cwd(), body.output_dir)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    run_id = f"bm_{uuid.uuid4().hex[:12]}"
    run = BenchmarkRun(
        id=run_id,
        status="queued",
        domains=",".join(body.domains),
        depth=body.depth,
        seeds=body.seeds,
        output_dir=str(output_dir),
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    logger.info("[benchmark] queued run %s — domains=%s", run_id, body.domains)

    run_benchmark_task.delay(
        run_id=run_id,
        domains=body.domains,
        depth=body.depth,
        seeds=body.seeds,
        output_dir=str(output_dir),
    )
    return {"run_id": run_id}


@router.get("/runs")
def list_benchmark_runs(db: Session = Depends(get_db)):
    runs = db.query(BenchmarkRun).order_by(BenchmarkRun.created_at.desc()).all()
    return [_run_to_dict(r) for r in runs]


@router.get("/runs/{run_id}")
def get_benchmark_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(BenchmarkRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="BenchmarkRun not found")
    return _run_to_dict(run)


@router.get("/runs/{run_id}/report")
def get_benchmark_report(run_id: str, db: Session = Depends(get_db)):
    run = db.get(BenchmarkRun, run_id)
    if run is None or run.report_json is None:
        raise HTTPException(status_code=404, detail="Report not available yet")
    return json.loads(run.report_json)


@router.get("/runs/{run_id}/report/download")
def download_benchmark_csv(run_id: str, db: Session = Depends(get_db)):
    import csv
    import io
    from fastapi.responses import StreamingResponse

    run = db.get(BenchmarkRun, run_id)
    if run is None or run.report_json is None:
        raise HTTPException(status_code=404, detail="Report not available yet")

    metrics = json.loads(run.report_json)
    output = io.StringIO()
    fieldnames = ["env_name", "state_coverage_score", "reward_density",
                  "dead_end_rate", "action_diversity", "num_episodes", "num_steps"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(metrics)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="env_quality_{run_id}.csv"'},
    )


@router.websocket("/ws/progress/{run_id}")
async def benchmark_progress_ws(websocket: WebSocket, run_id: str, db: Session = Depends(get_db)):
    """Stream benchmark run progress from Celery worker via Redis pub/sub."""
    import redis
    await websocket.accept()

    run = db.get(BenchmarkRun, run_id)
    if run is None:
        await websocket.send_json({"error": "run not found"})
        await websocket.close()
        return
    if run.status == "done":
        await websocket.send_json({"done": True})
        await websocket.close()
        return
    if run.status == "failed":
        await websocket.send_json({"error": run.error or "run failed"})
        await websocket.close()
        return

    redis_connection_url = redis_url()
    channel = f"forge:benchmark:{run_id}"
    try:
        r = redis.asyncio.from_url(redis_connection_url)
        pubsub = r.pubsub()
        await pubsub.subscribe(channel)
        logger.info("[ws:benchmark] subscribed to %s", channel)
    except Exception:
        logger.exception("[ws:benchmark] FAILED to connect to Redis")
        await websocket.close(code=1011)
        return

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            data = json.loads(message["data"])
            await websocket.send_json(data)
            if data.get("done") or data.get("error"):
                break
    except WebSocketDisconnect:
        logger.info("[ws:benchmark] client disconnected — run_id=%s", run_id)
    except Exception:
        logger.exception("[ws:benchmark] unexpected error for %s", run_id)
    finally:
        await pubsub.unsubscribe(channel)
        await r.aclose()
        try:
            await websocket.close()
        except RuntimeError:
            pass


def _run_to_dict(run: BenchmarkRun) -> dict:
    return {
        "id": run.id,
        "status": run.status,
        "domains": run.domains,
        "depth": run.depth,
        "seeds": run.seeds,
        "output_dir": run.output_dir,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "error": run.error,
    }
