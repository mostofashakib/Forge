"""End-to-End Test Suite for Critical Forge Platform Flows.

Verifies the end-to-end integration and correctness of:
1. Synthetic Data & Task Generation Pipeline (FR2 Contamination, FR3 Adversarial, FR7 Multi-Type)
2. Environment Determinism, Fingerprinting & Snapshot State Replay (FR5 Snapshot, FR6 Determinism)
3. Episode Execution, Reliability Retries & Budget Truncation (FR1 Reliability, FR4 Budget Truncation)
4. Robust GPU Compute & Inference Contracts (Local GPU vs. Cloud API Gateway with Fallback)
5. Benchmark Evaluation, Graph Metrics & Diagnostics (FR9 pass@k, pass^k, kappa, hacking diagnostics)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

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
from forge.taskfactory.contamination import (
    NGramOverlapDetector,
    TaskContaminationVerifier,
)
from forge.validation.robustness import (
    AttackStyle,
    CalibratedSafetyJudge,
    JailbreakRobustnessSuite,
    RiskCategory,
)
from forge.taskfactory.pipeline import PipelineResult
from forge.taskfactory.schemas import (
    TaskDraft,
    Taxonomy,
    TaxonomyCategory,
)
from forge.taskfactory.runner import fingerprint_of
from forge.benchmark.metrics import (
    TaskTrial,
    compute_cohens_kappa,
    compute_pass_at_k,
    compute_pass_pow_k,
    generate_benchmark_graphs,
)
from forge.contracts.termination import BudgetTerminationPolicy
from forge.contracts.types import StepOutcome
from forge.runtime.reliability import execute_reliable_episode


# ==============================================================================
# Critical Flow 1: Task Factory Synthetic Data Generation & Verification Pipeline
# ==============================================================================

class DummyTaskGenerationInference:
    """Mock LLM inference provider for synthetic task factory generation.

    # TODO: This is a dummy inference engine for synthetic task generation.
    # To switch to real inference:
    # 1. Provide a capable LLM API key (e.g. export ANTHROPIC_API_KEY="..." or OPENAI_API_KEY="...").
    # 2. Use LiteLLM or official SDKs:
    #      from litellm import completion
    #      resp = completion(model="anthropic/claude-3-7-sonnet", messages=messages)
    #      return resp.choices[0].message.content
    # 3. Configure FORGE_TASK_VALIDATOR_PROVIDER and FORGE_TASK_VALIDATOR_MODEL in .env.
    """

    def generate_task_taxonomy(self, env_name: str, count: int) -> dict:
        return {
            "categories": [
                {
                    "name": "data_manipulation",
                    "description": "Sorting, filtering, and summarizing records",
                    "exercises": [
                        {
                            "name": f"exercise_{i}",
                            "prompt": f"Inspect the database in {env_name} and clean up record #{i}.",
                            "difficulty": 1,
                        }
                        for i in range(count)
                    ],
                    "difficulties": [1],
                }
            ]
        }


def test_flow_task_factory_synthetic_data_pipeline_e2e(tmp_path):
    """E2E Flow: Generate synthetic tasks with contamination & adversarial robustness verification."""
    # 1. Initialize dummy inference for task generation
    dummy_llm = DummyTaskGenerationInference()
    raw_taxonomy = dummy_llm.generate_task_taxonomy(env_name="e2e_general_env", count=3)

    exercise_names = [ex["name"] for ex in raw_taxonomy["categories"][0]["exercises"]]

    taxonomy = Taxonomy(
        categories=[
            TaxonomyCategory(
                name="data_manipulation",
                description="Sorting, filtering, and summarizing records",
                exercises=exercise_names,
                difficulties=[1],
            )
        ]
    )

    # 2. Verify Contamination Guard (FR2): Check 8-gram overlap against benchmark and cutoff date
    verifier = TaskContaminationVerifier(
        training_corpus=[
            "Standard benchmark evaluation: inspect the database in production and clean up all records."
        ]
    )
    draft = TaskDraft(
        slot=0,
        title="exercise_0",
        objective="Inspect the database in e2e_general_env and clean up record #0.",
        golden=[],
        checks=[],
    )
    audit = verifier.verify_draft(draft)
    assert audit.is_contaminated is False
    assert audit.ngram_result is not None
    assert audit.ngram_result.overlap_ratio < 0.4  # No benchmark leakage

    # 3. Verify Adversarial Robustness Suite (FR3): Multi-turn probes and calibrated safety judge
    adversarial_suite = JailbreakRobustnessSuite()
    # Dummy respondent safely refuses attacks while answering benign prompts
    def dummy_safety_respondent(prompt: str) -> str:
        if any(w in prompt.lower() for w in ["crash", "wire transfer", "dump all user", "bypass"]):
            return "I cannot assist with this request as it violates safety guidelines."
        return "Here is the verified safe instruction for your workflow."

    report = adversarial_suite.evaluate_model(dummy_safety_respondent)
    assert report.passed is True
    assert report.harmful_compliance_rate == 0.0
    assert report.robustness_score == 1.0

    # 4. Multi-Type Synthetic Dataset Serialization (FR7: rl_tasks, preference_pairs, sft)
    pipeline_res = PipelineResult(
        requested=3,
        taxonomy=taxonomy,
        data_type="rl_tasks",
    )
    assert pipeline_res.data_type == "rl_tasks"

    # Verify formatting output
    serialized = {
        "data_type": pipeline_res.data_type,
        "taxonomy": taxonomy.model_dump(),
        "audit": audit.model_dump(),
        "adversarial_passed": report.passed,
    }
    assert serialized["data_type"] == "rl_tasks"
    assert len(serialized["taxonomy"]["categories"][0]["exercises"]) == 3


# ==============================================================================
# Critical Flow 2: Environment Determinism, Fingerprinting & Container Snapshot
# ==============================================================================

def test_flow_environment_determinism_and_snapshot_replay_e2e():
    """E2E Flow: Environment state reset produces stable fingerprints and verifies snapshot cloning."""
    # 1. State determinism verification
    initial_world_state_1 = {
        "users": [{"id": 1, "username": "alice", "balance": 100.0}],
        "inventory": {"item_99": {"stock": 42}},
    }
    initial_world_state_2 = {
        "inventory": {"item_99": {"stock": 42}},
        "users": [{"id": 1, "username": "alice", "balance": 100.0}],
    }

    # Fingerprint must be permutation-invariant for JSON keys
    fp1 = fingerprint_of(initial_world_state_1)
    fp2 = fingerprint_of(initial_world_state_2)
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex string

    # 2. Modifying state alters the deterministic fingerprint
    modified_state = {
        "users": [{"id": 1, "username": "alice", "balance": 90.0}],  # Deducted 10
        "inventory": {"item_99": {"stock": 41}},
    }
    fp3 = fingerprint_of(modified_state)
    assert fp3 != fp1

    # 3. Simulate snapshot restoration back to initial state
    restored_state = initial_world_state_1.copy()
    fp_restored = fingerprint_of(restored_state)
    assert fp_restored == fp1


# ==============================================================================
# Critical Flow 3: Episode Execution, Reliability Retries & Budget Truncation
# ==============================================================================

def test_flow_episode_reliability_and_budget_truncation_e2e():
    """E2E Flow: Retries on transient failure, short episode restart, and marks truncation on budget limit."""
    call_count = 0

    class MockEpisodeResult:
        def __init__(self, status="success", is_truncated=False):
            self.status = status
            self.is_truncated = is_truncated

    def flaky_task_runner(attempt: int, seed: int):
        nonlocal call_count
        call_count += 1
        if attempt == 0:
            # First attempt fails with transient network or container blip
            raise ConnectionResetError("Connection blip to container bridge")
        # Second attempt succeeds
        return MockEpisodeResult(status="success")

    exec_result = execute_reliable_episode(
        env_name="e2e_general_env",
        task_runner=flaky_task_runner,
        seed=42,
    )

    result, attempts = exec_result.result, exec_result.attempts
    assert result.status == "success"
    assert len(attempts) == 2
    assert call_count == 2

    # 2. Budget Guardrail Truncation (FR4: marks ep.status = "truncated")
    budget_policy = BudgetTerminationPolicy(
        max_steps=10,
        max_tokens=1000,
        max_wall_clock_time=5.0,
        max_cost=0.50,
    )

    # Within budget: outcome continues (check returns None)
    outcome_ok = StepOutcome(
        step_index=5,
        score=0.5,
        state_hash="hash_step_5",
        tokens=400,
        wall_clock_time=2.0,
        cost=0.10,
    )
    assert budget_policy.check(outcome_ok) is None

    # Exceeding step limit terminates with truncated=True
    outcome_exceeded = StepOutcome(
        step_index=12,  # Exceeded max_steps=10
        score=0.5,
        state_hash="hash_step_12",
        tokens=400,
        wall_clock_time=2.0,
        cost=0.10,
    )
    term = budget_policy.check(outcome_exceeded)
    assert term is not None
    assert term.truncated is True
    assert term.reason == "max_steps"


# ==============================================================================
# Critical Flow 4: GPU Inference Engine (Local GPU vs. Cloud Gateway & Fallback)
# ==============================================================================

class DummyLocalGPUBackend:
    """Dummy engine standing in for local CUDA / Apple Silicon MPS execution.

    # TODO: This is a dummy inference engine for local GPU inference.
    # To switch to real local GPU inference:
    # 1. Install GPU acceleration libraries:
    #      pip install torch vllm transformers accelerate
    # 2. Load model onto accelerator:
    #      from transformers import AutoModelForCausalLM, AutoTokenizer
    #      tokenizer = AutoTokenizer.from_pretrained(model_id)
    #      model = AutoModelForCausalLM.from_pretrained(
    #          model_id,
    #          device_map="auto",
    #          torch_dtype=torch.bfloat16,
    #      )
    # 3. Generate tokens:
    #      inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    #      outputs = model.generate(**inputs, max_new_tokens=128)
    #      return tokenizer.decode(outputs[0], skip_special_tokens=True)
    """

    def __init__(self, device: str = "mps"):
        self.device = device

    def infer(self, prompt: str) -> str:
        return f"[DUMMY_LOCAL_GPU:{self.device}] Completed action for: {prompt[:40]}"


class DummyCloudAPIGatewayBackend:
    """Dummy engine standing in for Cloud OpenAI-compatible API Gateway execution.

    # TODO: This is a dummy inference client for Cloud API Gateway.
    # To switch to real Cloud API Gateway inference:
    # 1. Configure the Gateway URL and API Key:
    #      import openai
    #      client = openai.OpenAI(
    #          base_url=gateway_url or "https://api.openai.com/v1",
    #          api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"),
    #      )
    # 2. Invoke chat completion:
    #      resp = client.chat.completions.create(
    #          model=model_name,
    #          messages=[{"role": "user", "content": prompt}],
    #          temperature=0.2,
    #      )
    #      return resp.choices[0].message.content
    """

    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def infer(self, prompt: str) -> str:
        return f"[DUMMY_API_GATEWAY:{self.endpoint_url}] Completed action for: {prompt[:40]}"


def test_flow_gpu_inference_contract_and_hybrid_fallback_e2e():
    """E2E Flow: Test GPU inference contract with local GPU provider, cloud gateway provider, and hybrid fallback."""
    # 1. Local GPU Provider execution
    gpu_spec = GPUDeviceSpec(device_type="auto", precision="bf16")
    local_provider = LocalGPUInferenceProvider(gpu_spec)
    req = InferenceRequest(prompt="Write bash command to inspect sqlite schema", model="forge-coder-7b")
    resp_local = local_provider.generate(req)
    assert resp_local.finish_reason == "stop"
    assert resp_local.metadata["provider"] == "local_gpu"
    assert resp_local.metadata["precision"] == "bf16"

    # 2. Cloud API Gateway Provider execution
    gateway_spec = APIGatewaySpec(endpoint_url="https://api.mock-gateway.forge.ai/v1", timeout_s=10)
    gateway_provider = CloudGatewayInferenceProvider(gateway_spec)

    # Use patch to simulate remote HTTP gateway response
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "sqlite3 test.db '.schema'"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        resp_gateway = gateway_provider.generate(req)
        assert resp_gateway.content == "sqlite3 test.db '.schema'"
        assert resp_gateway.metadata["provider"] == "api_gateway"

    # 3. Hybrid GPU Inference Engine Fallback (Local failure -> Gateway fallback)
    contract = GPUInferenceContract(
        mode=InferenceMode.AUTO,
        gpu_spec=GPUDeviceSpec(device_type="cuda"),
        api_gateway_spec=gateway_spec,
    )
    hybrid_engine = HybridGPUInferenceEngine(contract)

    with patch.object(GPUDeviceSpec, "probe_hardware", return_value={"cuda_available": True}), \
         patch.object(LocalGPUInferenceProvider, "generate", side_effect=RuntimeError("CUDA out of memory")), \
         patch("urllib.request.urlopen") as mock_urlopen_fb:
        mock_fb = MagicMock()
        mock_fb.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Fallback gateway action response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }).encode("utf-8")
        mock_fb.__enter__.return_value = mock_fb
        mock_urlopen_fb.return_value = mock_fb

        fallback_resp = hybrid_engine.generate(req)
        assert fallback_resp.content == "Fallback gateway action response"
        assert fallback_resp.metadata.get("fallback_from") == "local_gpu"
        assert fallback_resp.metadata["gateway_url"] == "https://api.mock-gateway.forge.ai/v1"


# ==============================================================================
# Critical Flow 6: Benchmark Evaluation, Metrics & Diagnostic Graphs
# ==============================================================================

class DummyBenchmarkModelEvaluator:
    """Mock evaluator for policy checkpoint evaluation across benchmark tasks.

    # TODO: This is a dummy benchmark evaluator.
    # To switch to real benchmark evaluation:
    # 1. Load the policy checkpoint:
    #      from transformers import pipeline
    #      generator = pipeline("text-generation", model=checkpoint.model_path, device_map="auto")
    # 2. Run episodes in the target environment:
    #      for task in benchmark_suite:
    #          action = generator(task.prompt)
    #          obs, reward, done, info = env.step(action)
    # 3. Record pass/fail per sample and compile TaskTrial records.
    """

    def evaluate(self, task_count: int = 10, samples_per_task: int = 5) -> list[TaskTrial]:
        trials = []
        for i in range(task_count):
            # Model passes 3 out of 5 samples consistently
            trials.append(
                TaskTrial(
                    task_id=f"benchmark_task_{i}",
                    passed_samples=3,
                    total_samples=samples_per_task,
                    verifier_agreement=0.92,
                    memorization_score=0.15,
                    contamination_flag=False,
                )
            )
        return trials


def test_flow_benchmark_evaluation_and_graph_diagnostics_e2e():
    """E2E Flow: Run evaluation, compute pass@1, pass@k, pass^k, kappa, and generate graph series."""
    evaluator = DummyBenchmarkModelEvaluator()
    trials = evaluator.evaluate(task_count=10, samples_per_task=5)

    # 1. Compute pass@k and pass^k metrics
    pass_at_1 = compute_pass_at_k(5, 3, 1)
    pass_at_3 = compute_pass_at_k(5, 3, 3)
    pass_pow_3 = compute_pass_pow_k(5, 3, 3)

    assert pass_at_1 == 0.6
    assert pass_at_3 > pass_at_1  # Pass@k increases with k
    assert pass_pow_3 < pass_at_1  # Pass^k decreases with k

    # 2. Cohen's Kappa Inter-Evaluator Agreement
    judge_a_decisions = [True, True, False, True, False, False, True, True]
    judge_b_decisions = [True, True, False, True, False, False, True, False]
    kappa = compute_cohens_kappa(judge_a_decisions, judge_b_decisions)
    assert 0.7 < kappa < 1.0  # High agreement

    # 3. Generate diagnostic benchmark graphs (FR9)
    graph_data = generate_benchmark_graphs(
        trials=trials,
        max_k=5,
    )

    assert graph_data.k_labels == [1, 2, 3, 4, 5]
    assert len(graph_data.pass_at_k) == 5
    assert len(graph_data.pass_pow_k) == 5

    diagnostics = graph_data.diagnostics
    assert "reward_hacking_risk" in diagnostics
    assert "memorization_risk" in diagnostics
    assert "contamination_risk" in diagnostics
    assert diagnostics["contamination_risk"] == "low"  # Clean suite
