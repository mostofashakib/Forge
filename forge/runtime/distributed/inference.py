"""Inference contracts and mock provider for distributed testing."""
from __future__ import annotations

import time

from forge.contracts.inference import (
    InferenceBatchRequest,
    InferenceBatchResponse,
    InferenceProvider,
    InferenceRequest,
    InferenceResponse,
)


class MockInferenceProvider(InferenceProvider):
    """Inference provider implementation executing local or mock inference requests."""

    def __init__(self, default_response: str = "Action completed.") -> None:
        self.default_response = default_response

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        t0 = time.time()
        content = self.default_response
        if request.prompt:
            content = f"Processed prompt: {request.prompt[:60]}"
        elif request.messages:
            last = request.messages[-1].get("content", "")
            content = f"Response to: {last[:60]}"
        latency = (time.time() - t0) * 1000.0
        return InferenceResponse(
            content=content,
            model=request.model,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            latency_ms=latency,
        )

    def generate_batch(self, batch: InferenceBatchRequest) -> InferenceBatchResponse:
        return InferenceBatchResponse(
            responses=[self.generate(req) for req in batch.requests]
        )
