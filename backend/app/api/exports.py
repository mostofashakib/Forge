from __future__ import annotations
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import ExportJob
from backend.app.services.export_service import run_export

router = APIRouter(prefix="/api/exports", tags=["exports"])


class CreateExportRequest(BaseModel):
    env_name: str
    formats: list[str]


@router.post("/")
def post_export(body: CreateExportRequest, db: Session = Depends(get_db)):
    job_id = f"ex_{secrets.token_hex(4)}"
    job = ExportJob(
        id=job_id,
        env_name=body.env_name,
        formats=json.dumps(body.formats),
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    db.commit()
    run_export(job_id, db)
    return {"export_job_id": job_id}


@router.get("/{export_job_id}")
def get_export_job(export_job_id: str, db: Session = Depends(get_db)):
    job = db.get(ExportJob, export_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ExportJob not found")
    return {
        "id": job.id,
        "env_name": job.env_name,
        "formats": json.loads(job.formats),
        "output_path": job.output_path,
        "status": job.status,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error,
    }


_ALLOWED_FILENAMES = frozenset([
    "trajectories.jsonl",
    "rewards.jsonl",
    "verifier_results.jsonl",
    "sft_pairs.jsonl",
    "preference_pairs.jsonl",
    "grpo_rollouts.parquet",
    "failure_dataset.jsonl",
])


@router.get("/{export_job_id}/download/{filename}")
def download_export_file(
    export_job_id: str, filename: str, db: Session = Depends(get_db)
):
    if filename not in _ALLOWED_FILENAMES:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {filename}")
    job = db.get(ExportJob, export_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ExportJob not found")
    if job.output_path is None:
        raise HTTPException(status_code=404, detail="Export not ready")
    file_path = Path(job.output_path) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    return FileResponse(str(file_path))
