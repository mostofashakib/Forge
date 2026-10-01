"""Inference API contracts for distributed training (Feature Request 8)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Specification of an inference request in distributed training."""

    model: str
    prompt: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    """Authoritative response returned from distributed inference."""

    content: str
    model: str
    finish_reason: str = "stop"
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceBatchRequest(BaseModel):
    """Batch of inference requests distributed across workers."""

    requests: list[InferenceRequest]


class InferenceBatchResponse(BaseModel):
    """Aggregated responses from a distributed inference batch."""

    responses: list[InferenceResponse]


class InferenceProvider(ABC):
    """Interface for distributed training inference providers."""

    @abstractmethod
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Execute a single inference request."""
        ...

    @abstractmethod
    def generate_batch(self, batch: InferenceBatchRequest) -> InferenceBatchResponse:
        """Execute a batch of inference requests concurrently."""
        ...
