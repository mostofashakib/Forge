from __future__ import annotations
from pydantic import BaseModel
from forge.extraction.schemas import CompilerInput


class ExtractRequest(BaseModel):
    prompt: str
    project_name: str
    domain: str


class ExtractResponse(BaseModel):
    job_id: str
    compiler_input: CompilerInput


class GenerateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    project_name: str
    compiler_input: CompilerInput | None = None
    output_path: str | None = None
    error: str | None = None
