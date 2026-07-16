from __future__ import annotations
import json
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.app.models import ExportJob
from backend.app.services.export_writers import WRITERS
from forge.settings import generated_envs_root

logger = logging.getLogger(__name__)
BASE_DIR = generated_envs_root()


def run_export(export_job_id: str, db: Session) -> None:
    job = db.get(ExportJob, export_job_id)
    if job is None:
        raise ValueError(f"ExportJob {export_job_id} not found")

    job.status = "running"
    db.commit()

    try:
        formats: list[str] = json.loads(job.formats)
        out_dir = BASE_DIR / job.env_name / "exports" / export_job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            writer = WRITERS.get(fmt)
            if writer is None:
                logger.warning("Unknown export format: %s", fmt)
                continue
            writer(job.env_name, db, out_dir)

        job.output_path = str(out_dir)
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as exc:
        logger.exception("ExportJob %s failed: %s", export_job_id, exc)
        job.status = "failed"
        job.error = str(exc)
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        raise
