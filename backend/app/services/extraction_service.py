from __future__ import annotations
from forge.extraction.llm_client import get_client
from forge.extraction.pipeline import ExtractionPipeline
from forge.extraction.schemas import CompilerInput


def run_extraction(prompt: str, project_name: str, domain: str) -> CompilerInput:
    client = get_client(capable=True)
    pipeline = ExtractionPipeline(client)
    return pipeline.run(prompt=prompt, project_name=project_name, domain=domain)
