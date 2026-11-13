from __future__ import annotations

import os

from forge.runtime.agents.openai_agent import OpenAIAgent


class OllamaAgent(OpenAIAgent):
    """An Ollama model through Ollama's OpenAI-compatible endpoint, which supports native tool calls."""

    def __init__(self, model: str, client=None, **kwargs) -> None:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1"
        super().__init__(model=model, client=client, base_url=base_url, **kwargs)
