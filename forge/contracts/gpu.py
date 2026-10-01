"""Robust GPU and Compute Inference contracts for training and benchmark.

Supports local GPU inference (CUDA, MPS, local accelerators) vs. Cloud API Gateway inference
with automatic fallback, hardware inspection, and precision controls.
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class InferenceMode(str, Enum):
    """Execution targets for model inference during training and benchmark."""

    LOCAL_GPU = "local_gpu"
    API_GATEWAY = "api_gateway"
    AUTO = "auto"


class GPUDeviceSpec(BaseModel):
    """Specification for local GPU acceleration."""

    device_type: Literal["cuda", "mps", "cpu", "auto"] = "auto"
    device_ids: list[int] = Field(default_factory=lambda: [0])
    vram_budget_mb: int | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    precision: Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"] = "bf16"
    memory_fraction: float = Field(default=0.9, ge=0.1, le=1.0)
    local_runner: str = "vllm"  # "vllm", "transformers", "ollama", "torch"

    def resolved_device(self) -> str:
        """Resolve actual hardware target (e.g., cuda:0, mps, cpu)."""
        if self.device_type != "auto":
            return self.device_type

        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                dev_id = self.device_ids[0] if self.device_ids else 0
                return f"cuda:{dev_id}"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    @classmethod
    def probe_hardware(cls) -> dict[str, Any]:
        """Inspect available local hardware accelerators."""
        info: dict[str, Any] = {
            "cuda_available": False,
            "mps_available": False,
            "device_count": 0,
            "devices": [],
        }
        try:
            import torch
            if torch.cuda.is_available():
                info["cuda_available"] = True
                info["device_count"] = torch.cuda.device_count()
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    info["devices"].append({
                        "id": i,
                        "name": props.name,
                        "total_memory_mb": props.total_memory // (1024 * 1024),
                    })
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info["mps_available"] = True
                info["device_count"] = 1
                info["devices"].append({"id": 0, "name": "Apple Silicon (MPS)"})
        except Exception as exc:
            info["error"] = str(exc)

        return info


class APIGatewaySpec(BaseModel):
    """Specification for cloud API gateway inference."""

    endpoint_url: str = Field(
        default_factory=lambda: os.environ.get("FORGE_INFERENCE_GATEWAY_URL", "https://api.openai.com/v1")
    )
    api_key_env: str = "FORGE_INFERENCE_API_KEY"
    api_key: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = Field(default=60.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)
    route_path: str = "/chat/completions"

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env) or os.environ.get("OPENAI_API_KEY")


class GPUInferenceContract(BaseModel):
    """Authoritative contract bridging training & benchmark to local GPU or Cloud Gateway."""

    mode: InferenceMode = InferenceMode.AUTO
    gpu_spec: GPUDeviceSpec = Field(default_factory=GPUDeviceSpec)
    api_gateway_spec: APIGatewaySpec = Field(default_factory=APIGatewaySpec)
    fallback_to_cloud: bool = True

    def effective_mode(self) -> InferenceMode:
        """Resolve AUTO mode based on available local hardware."""
        if self.mode != InferenceMode.AUTO:
            return self.mode

        hardware = GPUDeviceSpec.probe_hardware()
        if hardware.get("cuda_available") or hardware.get("mps_available"):
            return InferenceMode.LOCAL_GPU
        return InferenceMode.API_GATEWAY
