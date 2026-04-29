from __future__ import annotations
import os
from forge.extraction.llm_client import AnthropicClient
from forge.extraction.pipeline import ExtractionPipeline
from forge.extraction.schemas import CompilerInput


def run_extraction(prompt: str, project_name: str, domain: str) -> CompilerInput:
    model = os.environ.get("FORGE_LLM_MODEL", "claude-sonnet-4-6")
    client = AnthropicClient(model=model)
    pipeline = ExtractionPipeline(client)
    return pipeline.run(prompt=prompt, project_name=project_name, domain=domain)
