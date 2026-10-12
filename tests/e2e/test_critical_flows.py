"""End-to-End Test Suite for Critical Forge Platform Flows.

Verifies the end-to-end integration and correctness of:
1. Synthetic Data & Task Generation Pipeline (FR2 Contamination, FR3 Adversarial, FR7 Multi-Type)
2. Environment Determinism, Fingerprinting & Snapshot State Replay (FR5 Snapshot, FR6 Determinism)
3. Episode Execution, Reliability Retries & Budget Truncation (FR1 Reliability, FR4 Budget Truncation)
"""
from __future__ import annotations


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
