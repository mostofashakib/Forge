from __future__ import annotations
import json
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.app.models import ExportJob
from backend.app.services.export_writers import WRITERS
from backend.app.services.export_writers.gate import QUARANTINE_FILE, gate_from_env
from backend.app.services.export_writers.manifest import MANIFEST_FILE, dataset_manifest
from forge.safety.types import DataSource
from forge.settings import generated_envs_root

logger = logging.getLogger(__name__)


def run_export(export_job_id: str, db: Session) -> None:
    job = db.get(ExportJob, export_job_id)
    if job is None:
        raise ValueError(f"ExportJob {export_job_id} not found")

    job.status = "running"
    db.commit()

    try:
        formats: list[str] = json.loads(job.formats)
        out_dir = generated_envs_root() / job.env_name / "exports" / export_job_id
        # Built before the folder exists, so a misconfigured gate leaves nothing behind.
        # Rollouts are model-generated: screened for harm, not PII-scrubbed.
        gate = gate_from_env(out_dir, DataSource.MODEL)
        out_dir.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            writer = WRITERS.get(fmt)
            if writer is None:
                logger.warning("Unknown export format: %s", fmt)
                continue
            writer(job.env_name, db, out_dir, gate)

        manifest = dataset_manifest(
            files=sorted(
                f.name for f in out_dir.iterdir()
                if f.is_file() and f.name not in (MANIFEST_FILE, QUARANTINE_FILE)
            ),
            safety=gate.stats(),
            export_job_id=export_job_id,
            env_name=job.env_name,
            formats=formats,
        )
        (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))

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
