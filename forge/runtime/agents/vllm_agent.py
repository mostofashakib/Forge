from __future__ import annotations
import os
from forge.runtime.agents.openai_agent import OpenAIAgent


class vLLMAgent(OpenAIAgent):
    def __init__(self, model: str, client=None) -> None:
        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        super().__init__(model=model, client=client, base_url=base_url)
