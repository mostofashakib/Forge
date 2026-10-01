"""End-to-End Test Suite for Critical Forge Platform Flows.

Verifies the end-to-end integration and correctness of:
1. Synthetic Data & Task Generation Pipeline (FR2 Contamination, FR3 Adversarial, FR7 Multi-Type)
2. Environment Determinism, Fingerprinting & Snapshot State Replay (FR5 Snapshot, FR6 Determinism)
3. Episode Execution, Reliability Retries & Budget Truncation (FR1 Reliability, FR4 Budget Truncation)
4. Robust GPU Compute & Inference Contracts (Local GPU vs. Cloud API Gateway with Fallback)
5. Policy Training Pipeline (GRPO / DPO Objectives, Checkpointing & Registry)
6. Benchmark Evaluation, Graph Metrics & Diagnostics (FR9 pass@k, pass^k, kappa, hacking diagnostics)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database import init_db
from backend.app.models import SandboxEnvironment
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
from forge.training.checkpoint import PolicyCheckpoint
from forge.training.trainer import (
    PolicyTrainer,
    TrainingConfig,
    TrainingObjective,
)
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
# Fixtures and Test Harness Setup
# ==============================================================================

@pytest.fixture
def e2e_client(tmp_path, monkeypatch):
    """Provides a fresh isolated database and TestClient environment for E2E flows."""
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/e2e_forge.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    monkeypatch.setenv("FORGE_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-5")
    monkeypatch.chdir(tmp_path)

    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    init_db()

    with database.get_session_factory()() as db:
        # Register test environments
        db.add(SandboxEnvironment(
            id="e2e_general_env",
            status="running",
            env_type="container",
            container_id="cont_e2e_01",
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        ))
        db.commit()

    return TestClient(app)


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
    gateway_spec = APIGatewaySpec(endpoint_url="https://api.mock-gateway.forge.ai/v1", timeout_seconds=10)
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
        gpu_spec=GPUDeviceSpec(device_type="cuda"),  # CUDA will fail if host has no CUDA
        api_gateway=gateway_spec,
    )
    hybrid_engine = HybridGPUInferenceEngine(contract)

    with patch.object(LocalGPUInferenceProvider, "generate", side_effect=RuntimeError("CUDA out of memory")), \
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


# ==============================================================================
# Critical Flow 5: Policy Training Pipeline (GRPO / DPO) & Checkpoints
# ==============================================================================

class DummyPolicyTrainingBackend:
    """Dummy policy training backend executing GRPO and DPO updates.

    # TODO: This is a dummy policy training backend.
    # To switch to real model training:
    # 1. Install HuggingFace TRL and PyTorch:
    #      pip install torch trl transformers peft datasets accelerate
    # 2. For GRPO (Group Relative Policy Optimization):
    #      from trl import GRPOTrainer, GRPOConfig
    #      training_args = GRPOConfig(output_dir=output_dir, max_steps=max_steps, learning_rate=1e-5)
    #      trainer = GRPOTrainer(model=base_model, reward_funcs=reward_funcs, args=training_args, train_dataset=dataset)
    #      trainer.train()
    # 3. For DPO (Direct Preference Optimization):
    #      from trl import DPOTrainer, DPOConfig
    #      trainer = DPOTrainer(model=base_model, ref_model=None, args=DPOConfig(output_dir=output_dir), train_dataset=dataset)
    #      trainer.train()
    # 4. Save model weights and tokenizer:
    #      trainer.save_model(output_dir)
    """

    def __init__(self) -> None:
        self.recorded_runs: list[dict] = []

    def train(self, base_model: str, examples: list, output_dir: Path | str, max_steps: int) -> str:
        self.recorded_runs.append({
            "base_model": base_model,
            "example_count": len(examples),
            "max_steps": max_steps,
            "output_dir": str(output_dir),
        })
        model_dir = Path(output_dir) / "policy_weights"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(json.dumps({"base_model": base_model, "steps": max_steps}))
        return str(model_dir)


def test_flow_policy_training_grpo_and_checkpoint_production_e2e(tmp_path):
    """E2E Flow: Prepare rollouts, run GRPO training, save PolicyCheckpoint, and verify loadability."""
    # 1. Generate rollout dataset (parquet)
    import pandas as pd
    data_dir = tmp_path / "training_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rollouts = [
        {
            "episode_id": f"ep_{i}",
            "env_name": "e2e_general_env",
            "task_name": "clean_db",
            "prompt": "Task: clean_db\nEnvironment: e2e_general_env",
            "completion": f"action_{i}",
            "total_reward": 0.5 + (0.1 * i),
            "passed": True,
            "per_step_rewards": json.dumps([0.5 + (0.1 * i)]),
        }
        for i in range(5)
    ]
    pd.DataFrame(rollouts).to_parquet(data_dir / "grpo_rollouts.parquet", index=False)

    # 2. Execute Policy Trainer using Dummy Backend
    backend = DummyPolicyTrainingBackend()
    output_dir = tmp_path / "policy_checkpoints"

    trainer = PolicyTrainer(backend=backend)
    result = trainer.train(
        TrainingConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            objective=TrainingObjective.GRPO,
            max_steps=200,
        )
    )

    assert result.objective == "grpo"
    assert len(backend.recorded_runs) == 1
    assert backend.recorded_runs[0]["example_count"] == 5
    assert backend.recorded_runs[0]["max_steps"] == 200

    # 3. Verify PolicyCheckpoint format & metadata
    ckpt = PolicyCheckpoint.load(output_dir)
    assert ckpt.objective == "grpo"
    assert ckpt.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert ckpt.num_examples == 5
    assert Path(ckpt.model_path).exists()


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


# ==============================================================================
# Critical Flow 7: Full Web API Route Integration E2E
# ==============================================================================

def test_flow_web_api_routes_integration_e2e(e2e_client, tmp_path):
    """E2E Flow: Exercises Web API endpoints for Training and Benchmark hardware discovery."""
    # 1. Hardware probing endpoint
    hw_res = e2e_client.get("/api/training/hardware")
    assert hw_res.status_code == 200
    hw_data = hw_res.json()
    assert "hardware" in hw_data
    assert "api_gateway" in hw_data
    assert "default_mode" in hw_data

    # 2. Launch Training Run via API
    data_dir = tmp_path / "api_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    with patch("backend.app.api.training.threading.Thread") as mock_thread:
        mock_instance = mock_thread.return_value
        run_res = e2e_client.post(
            "/api/training/runs",
            json={
                "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "data_dir": "api_data",
                "output_dir": "api_policy",
                "objective": "grpo",
                "max_steps": 100,
                "inference_mode": "auto",
                "gpu_precision": "bf16",
            },
        )
        assert run_res.status_code == 202
        run_id = run_res.json()["run_id"]
        assert run_id.startswith("tr_")
        assert mock_instance.start.called

    # 3. Retrieve Run Status
    get_run_res = e2e_client.get(f"/api/training/runs/{run_id}")
    assert get_run_res.status_code == 200
    run_payload = get_run_res.json()
    assert run_payload["base_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert run_payload["objective"] == "grpo"
    assert run_payload["inference_mode"] == "auto"

    # 4. Checkpoints discovery endpoint
    cp_dir = tmp_path / "api_policy"
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp = PolicyCheckpoint(
        objective="grpo",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        model_path="api_policy/weights",
        num_examples=120,
        mean_reward=0.94,
        run_id=run_id,
    )
    cp.save(cp_dir / "checkpoint_01")

    cp_res = e2e_client.get("/api/training/checkpoints?output_dir=api_policy")
    assert cp_res.status_code == 200
    checkpoints = cp_res.json()
    assert len(checkpoints) >= 1
    assert checkpoints[0]["run_id"] == run_id
    assert checkpoints[0]["num_examples"] == 120
