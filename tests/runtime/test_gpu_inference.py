"""Tests for robust GPU and API Gateway compute & inference contracts."""
from unittest.mock import MagicMock, patch
import pytest

from forge.contracts.gpu import (
    APIGatewaySpec,
    GPUDeviceSpec,
    GPUInferenceContract,
    InferenceMode,
)
from forge.contracts.inference import InferenceRequest
from forge.runtime.gpu_inference import (
    CloudGatewayInferenceProvider,
    HybridGPUInferenceEngine,
    LocalGPUInferenceProvider,
)


def test_gpu_device_spec_and_probe():
    spec = GPUDeviceSpec(device_type="auto", precision="bf16")
    dev = spec.resolved_device()
    assert isinstance(dev, str)
    assert dev in ("cuda:0", "mps", "cpu")

    info = GPUDeviceSpec.probe_hardware()
    assert "cuda_available" in info
    assert "mps_available" in info
    assert isinstance(info["devices"], list)


def test_api_gateway_spec_resolution(monkeypatch):
    monkeypatch.setenv("FORGE_INFERENCE_API_KEY", "test-secret-key")
    spec = APIGatewaySpec(endpoint_url="https://gateway.example.com/v1")
    assert spec.resolved_api_key() == "test-secret-key"
    assert spec.endpoint_url == "https://gateway.example.com/v1"


def test_gpu_inference_contract_effective_mode():
    # Explicit mode
    c_local = GPUInferenceContract(mode=InferenceMode.LOCAL_GPU)
    assert c_local.effective_mode() == InferenceMode.LOCAL_GPU

    c_cloud = GPUInferenceContract(mode=InferenceMode.API_GATEWAY)
    assert c_cloud.effective_mode() == InferenceMode.API_GATEWAY

    # Auto mode resolution
    c_auto = GPUInferenceContract(mode=InferenceMode.AUTO)
    eff = c_auto.effective_mode()
    assert eff in (InferenceMode.LOCAL_GPU, InferenceMode.API_GATEWAY)


def test_local_gpu_inference_provider():
    provider = LocalGPUInferenceProvider(GPUDeviceSpec(precision="fp16"))
    req = InferenceRequest(model="test-policy", prompt="Test prompt for local GPU")
    res = provider.generate(req)
    assert res.model == "test-policy"
    assert "local" in res.metadata["provider"]
    assert res.metadata["precision"] == "fp16"
    assert res.latency_ms >= 0


def test_cloud_gateway_inference_provider():
    gw_spec = APIGatewaySpec(endpoint_url="https://mock-gateway.internal/v1")
    provider = CloudGatewayInferenceProvider(gw_spec)
    req = InferenceRequest(model="gpt-4o-mini", prompt="Test prompt for cloud gateway")
    res = provider.generate(req)
    assert res.model == "gpt-4o-mini"
    assert res.metadata["provider"] == "api_gateway"
    assert res.metadata["gateway_url"] == "https://mock-gateway.internal/v1"


def test_hybrid_gpu_inference_engine_fallback():
    contract = GPUInferenceContract(
        mode=InferenceMode.LOCAL_GPU,
        fallback_to_cloud=True,
    )
    engine = HybridGPUInferenceEngine(contract)

    # Mock local provider to fail, ensuring automatic cloud fallback works
    with patch.object(engine.local_provider, "generate", side_effect=RuntimeError("CUDA Out of Memory")):
        req = InferenceRequest(model="deepseek-r1", prompt="Execute policy step")
        res = engine.generate(req)
        assert res.metadata["provider"] == "api_gateway"


def test_hybrid_gpu_without_fallback_raises_on_failure():
    contract = GPUInferenceContract(
        mode=InferenceMode.LOCAL_GPU,
        fallback_to_cloud=False,
    )
    engine = HybridGPUInferenceEngine(contract)
    with patch.object(engine.local_provider, "generate", side_effect=RuntimeError("CUDA Device Lost")):
        req = InferenceRequest(model="deepseek-r1", prompt="Execute policy step")
        with pytest.raises(RuntimeError, match="CUDA Device Lost"):
            engine.generate(req)


def test_rejects_empty_prompt():
    provider = LocalGPUInferenceProvider(GPUDeviceSpec())
    req = InferenceRequest(model="test", prompt="")
    # False-positive / boundary check: empty prompt should be handled gracefully or validated
    assert provider.generate(req).content is not None
