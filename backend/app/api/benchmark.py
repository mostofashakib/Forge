from __future__ import annotations
import json
import logging
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, WebSocket
from pydantic import BaseModel, Field, model_validator
from typing import Literal
from sqlalchemy.orm import Session

from backend.app.api._pubsub_relay import stream_channel
from backend.app.database import get_db, get_session_factory
from backend.app.models import BenchmarkRun
from forge.paths import confined_relative_path
from forge.settings import redis_url
from forge.benchmark.metrics import TaskTrial, generate_benchmark_graphs

logger = logging.getLogger(__name__)

# Lazy module-level reference so tests can patch backend.app.api.benchmark.run_benchmark_task.
# The actual import is deferred to avoid circular-import issues at package load time.
try:
    from backend.app.worker.benchmark_tasks import run_benchmark_task, run_evaluation_task, run_transfer_task  # noqa: F401
except Exception:  # pragma: no cover
    run_benchmark_task = None  # type: ignore[assignment]
    run_evaluation_task = None  # type: ignore[assignment]
    run_transfer_task = None  # type: ignore[assignment]
router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


class CreateBenchmarkRunRequest(BaseModel):
    domains: list[str] = Field(..., min_length=1)
    depth: int = Field(default=5, ge=1, le=5)
    seeds: int = Field(default=5, ge=1, le=100)
    output_dir: str = Field(default="benchmark_results", min_length=1, max_length=255)
    inference_mode: Literal["auto", "local_gpu", "api_gateway"] = "auto"


class CreateTransferRequest(BaseModel):
    # None trains the base model the experiment declares.
    base_model: str | None = Field(default=None, min_length=1)
    data_dir: str = Field(default="benchmark_results/data", min_length=1)
    output_dir: str = Field(default="benchmark_results/transfer", min_length=1)
    # Experiment YAML whose train/held-out split the transfer run follows.
    eval_suite: str = Field(default="experiments/internal_heldout.yaml", min_length=1)
    max_steps: int = Field(default=500, ge=1, le=5000)
    seeds: int = Field(default=3, ge=1, le=20)
    inference_mode: Literal["auto", "local_gpu", "api_gateway"] = "auto"


class CreateEvaluationRequest(BaseModel):
    engine: Literal["forge", "harbor"] = "forge"
    checkpoint: str | None = "policy_checkpoint"
    experiment: str | None = "experiments/internal_heldout.yaml"
    seed: int | None = None
    runs_dir: str = Field(default="runs", min_length=1, max_length=255)
    # The example task's own adapter agent, on a local model. Both halves of
    # the default matter: the agent resolves `provider/model` itself, so
    # changing provider here is one string and needs no code, and Ollama needs
    # no account -- a default that reaches for a hosted key nobody has set is a
    # default that fails on first use.
    harbor_task_path: str | None = "example_tasks/task_manager"
    harbor_agent: str | None = "agent.harbor_agent:TrackerAgent"
    harbor_model: str | None = "ollama/qwen3.6:35b"
    inference_mode: Literal["auto", "local_gpu", "api_gateway"] = "auto"

    @model_validator(mode="after")
    def validate_engine_fields(self):
        required = (
            ("checkpoint", "experiment")
            if self.engine == "forge"
            else ("harbor_task_path", "harbor_agent", "harbor_model")
        )
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ValueError(f"{self.engine} evaluation requires: {', '.join(missing)}")
        return self


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

    if run_benchmark_task:
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
    runs = (
        db.query(BenchmarkRun)
        .filter(BenchmarkRun.kind == "benchmark")
        .order_by(BenchmarkRun.created_at.desc())
        .all()
    )
    return [_run_to_dict(r) for r in runs]


@router.get("/evals/capabilities")
def evaluation_capabilities():
    bundled_harbor = Path.cwd() / "example_tasks" / ".venv" / "bin" / "harbor"
    from forge.contracts.gpu import compute_status
    status = compute_status()
    return {
        "engines": {
            "forge": {"available": True},
            "harbor": {
                "available": bool(shutil.which("harbor") or bundled_harbor.is_file()),
                "setup_hint": "Run `uv tool install harbor` to enable Harbor.",
            },
        },
        "compute": {
            "hardware": status["hardware"],
            "local_gpu_available": status["default_mode"] == "local_gpu",
            "api_gateway": status["api_gateway"],
            "supported_modes": ["auto", "local_gpu", "api_gateway"],
        },
    }


@router.post("/evals", status_code=202)
def create_evaluation(body: CreateEvaluationRequest, db: Session = Depends(get_db)):
    root = Path.cwd()
    config = body.model_dump()
    path_fields = (
        ("checkpoint", "experiment", "runs_dir")
        if body.engine == "forge"
        else ("harbor_task_path",)
    )
    try:
        for field_name in path_fields:
            value = config.get(field_name)
            if value:
                config[field_name] = str(confined_relative_path(root, value))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    required_existing = (
        ("checkpoint", "experiment")
        if body.engine == "forge"
        else ("harbor_task_path",)
    )
    for field_name in required_existing:
        if not Path(config[field_name]).exists():
            raise HTTPException(status_code=422, detail=f"{field_name} does not exist")

    run_id = f"eval_{uuid.uuid4().hex[:12]}"
    run = BenchmarkRun(
        id=run_id,
        status="queued",
        domains="heldout" if body.engine == "forge" else str(body.harbor_task_path),
        depth=body.seed or 0,
        seeds=1,
        output_dir=config.get("runs_dir") or "jobs",
        created_at=datetime.now(timezone.utc),
        kind="evaluation",
        engine=body.engine,
        config_json=json.dumps(config),
    )
    db.add(run)
    db.commit()
    run_evaluation_task.delay(run_id=run_id, engine=body.engine, config=config)
    return {"run_id": run_id}


@router.get("/evals/{run_id}")
def get_evaluation(run_id: str, db: Session = Depends(get_db)):
    run = db.get(BenchmarkRun, run_id)
    if run is None or run.kind != "evaluation":
        raise HTTPException(status_code=404, detail="Evaluation not found")
    payload = _run_to_dict(run)
    payload["result"] = json.loads(run.report_json) if run.report_json else None
    return payload


@router.post("/transfer", status_code=202)
def create_transfer_run(body: CreateTransferRequest, db: Session = Depends(get_db)):
    config = body.model_dump()
    try:
        for field_name in ("eval_suite", "data_dir", "output_dir"):
            config[field_name] = str(confined_relative_path(Path.cwd(), config[field_name]))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not Path(config["eval_suite"]).is_file():
        raise HTTPException(status_code=422, detail="eval_suite does not exist")

    run_id = f"transfer_{uuid.uuid4().hex[:12]}"
    run = BenchmarkRun(
        id=run_id,
        status="queued",
        domains=body.eval_suite,
        depth=body.max_steps,
        seeds=body.seeds,
        output_dir=config["output_dir"],
        created_at=datetime.now(timezone.utc),
        kind="transfer",
        engine="forge-transfer",
        config_json=json.dumps(config),
    )
    db.add(run)
    db.commit()
    logger.info("[benchmark] queued transfer run %s — model=%s suite=%s", run_id, body.base_model, body.eval_suite)

    if run_transfer_task:
        run_transfer_task.delay(run_id=run_id, config=config)
    return {"run_id": run_id}


@router.get("/transfer/{run_id}")
def get_transfer_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(BenchmarkRun, run_id)
    if run is None or run.kind != "transfer":
        raise HTTPException(status_code=404, detail="Transfer run not found")
    payload = _run_to_dict(run)
    payload["result"] = json.loads(run.report_json) if run.report_json else None
    return payload


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


class GenerateGraphsRequest(BaseModel):
    trials: list[TaskTrial]
    max_k: int | None = 10
    graph_type: Literal["all", "pass_curve", "agreement_bar", "risk_breakdown"] = "all"


@router.get("/runs/{run_id}/graphs")
def get_benchmark_graphs(
    run_id: str,
    graph_type: Literal["all", "pass_curve", "agreement_bar", "risk_breakdown"] = "all",
    max_k: int = 10,
    db: Session = Depends(get_db),
):
    """Retrieve benchmark graph options (pass@1, pass@k, pass^k, Cohen's kappa, diagnostics)."""
    run = db.get(BenchmarkRun, run_id)
    if run is None or run.report_json is None:
        raise HTTPException(status_code=404, detail="Benchmark run or report not found")

    raw_data = json.loads(run.report_json)
    trials: list[TaskTrial] = []

    if isinstance(raw_data, dict) and "trials" in raw_data:
        trials = [TaskTrial(**t) for t in raw_data["trials"]]
    elif isinstance(raw_data, list):
        for idx, item in enumerate(raw_data):
            num_episodes = max(1, item.get("num_episodes", 10))
            reward_density = item.get("reward_density", 0.5)
            coverage = item.get("state_coverage_score", 0.5)
            diversity = item.get("action_diversity", 0.5)
            passed = max(0, int(round(reward_density * num_episodes)))
            para_passed = max(0, int(round(coverage * num_episodes)))
            v_passes = [True] * passed + [False] * (num_episodes - passed)
            gt_passes = [True] * para_passed + [False] * (num_episodes - para_passed)

            trials.append(
                TaskTrial(
                    task_id=item.get("env_name", f"env_{idx}"),
                    passed_samples=passed,
                    total_samples=num_episodes,
                    paraphrased_passed_samples=para_passed,
                    paraphrased_total_samples=num_episodes,
                    verifier_passes=v_passes,
                    ground_truth_passes=gt_passes,
                    ngram_overlap=max(0.0, min(1.0, 1.0 - diversity)),
                )
            )
    elif isinstance(raw_data, dict):
        trials.append(
            TaskTrial(
                task_id=raw_data.get("task_path", run_id),
                passed_samples=1 if raw_data.get("status") == "completed" else 0,
                total_samples=1,
            )
        )

    graph_data = generate_benchmark_graphs(trials, max_k=max_k)
    res = graph_data.model_dump()
    if graph_type != "all" and graph_type in res["chart_configs"]:
        res["chart_configs"] = {graph_type: res["chart_configs"][graph_type]}
    return res


@router.post("/graphs")
def create_benchmark_graphs(body: GenerateGraphsRequest) -> dict:
    """Calculate and render benchmark graphs (pass@1, pass@k, pass^k, Cohen's kappa, diagnostics)."""
    graph_data = generate_benchmark_graphs(body.trials, max_k=body.max_k)
    res = graph_data.model_dump()
    if body.graph_type != "all" and body.graph_type in res["chart_configs"]:
        res["chart_configs"] = {body.graph_type: res["chart_configs"][body.graph_type]}
    return res


def _finished_run_message(run_id: str) -> dict | None:
    """The terminal progress message for a run that already ended, else None."""
    with get_session_factory()() as db:
        run = db.get(BenchmarkRun, run_id)
        if run is None:
            return {"error": "run not found"}
        if run.status == "done":
            return {"done": True}
        if run.status == "failed":
            return {"error": run.error or "run failed"}
        return None


@router.websocket("/ws/progress/{run_id}")
async def benchmark_progress_ws(websocket: WebSocket, run_id: str):
    """Stream benchmark run progress from Celery worker via Redis pub/sub."""
    await stream_channel(
        websocket,
        redis_url=redis_url(),
        channel=f"forge:benchmark:{run_id}",
        finished_message=lambda: _finished_run_message(run_id),
        is_final=lambda data: bool(data.get("done") or data.get("error")),
        log_tag="benchmark",
    )


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
        "kind": run.kind,
        "engine": run.engine,
    }
