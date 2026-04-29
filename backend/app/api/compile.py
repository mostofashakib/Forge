from __future__ import annotations
import json
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.models import CompileJob
from backend.app.schemas import (
    ExtractRequest, ExtractResponse, GenerateResponse, JobStatusResponse,
)
from backend.app.services import extraction_service, compiler_service
from forge.extraction.schemas import CompilerInput

router = APIRouter(prefix="/api/compile")


@router.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest, db: Session = Depends(get_db)):
    compiler_input = extraction_service.run_extraction(
        prompt=request.prompt,
        project_name=request.project_name,
        domain=request.domain,
    )
    job_id = str(uuid.uuid4())
    job = CompileJob(
        id=job_id,
        project_name=request.project_name,
        status="reviewing",
        prompt=request.prompt,
        compiler_input_json=compiler_input.model_dump_json(),
    )
    db.add(job)
    db.commit()
    return ExtractResponse(job_id=job_id, compiler_input=compiler_input)


@router.post("/generate/{job_id}", response_model=GenerateResponse)
def generate(job_id: str, compiler_input: CompilerInput, db: Session = Depends(get_db)):
    job = db.get(CompileJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.compiler_input_json = compiler_input.model_dump_json()
    job.status = "generating"
    db.commit()
    try:
        result = compiler_service.run_compilation(compiler_input)
        if result is None:
            job.status = "complete"
        else:
            pkg_dir, validation = result
            job.status = "complete" if validation.passed else "failed"
            job.output_path = str(pkg_dir)
            if not validation.passed:
                job.error = validation.output[-2000:]
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
    db.commit()
    return GenerateResponse(job_id=job_id, status=job.status)


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(CompileJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    ci = CompilerInput.model_validate_json(job.compiler_input_json) if job.compiler_input_json else None
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        project_name=job.project_name,
        compiler_input=ci,
        output_path=job.output_path,
        error=job.error,
    )
