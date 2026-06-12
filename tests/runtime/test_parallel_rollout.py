# tests/runtime/test_parallel_rollout.py
import copy
import threading
from forge.runtime.env_builder import EnvBuilder
from forge.runtime.parallel_rollout import (
    ParallelRolloutRunner,
    RolloutBatch,
    RolloutSpec,
)
from forge.runtime.transition import TransitionResult
from forge.runtime.verification import CheckResult, VerificationResult


class SeededCounterFactory:
    def create(self, ctx, options):
        return {"counter": {"c_0": {"id": "c_0", "value": ctx.rng.randint(0, 1000)}}}


def increment(state, action, ctx):
    new_state = copy.deepcopy(state)
    new_state["counter"]["c_0"]["value"] += 50
    return TransitionResult(state=new_state, events=[{"type": "incremented"}])


def reach_900(state, trajectory, task):
    passed = state["counter"]["c_0"]["value"] >= 900
    return VerificationResult.from_checks(
        "reach_900", [CheckResult(name="reached", passed=passed, score=1.0 if passed else 0.0)]
    )


def make_env_factory(created_envs=None):
    def factory():
        env = (
            EnvBuilder("rollout_env", domain="test", max_steps=5)
            .with_initial_state(SeededCounterFactory())
            .with_transition("increment", increment)
            .with_verifier("reach_900", reach_900)
            .build(verify=False)
        )
        if created_envs is not None:
            created_envs.append(env)
        return env

    return factory


TASK = {"name": "reach_900", "verifier_id": "reach_900", "inputs": {}}


def increment_policy(obs, action_types):
    return {"type": "increment"}


# ---------------------------------------------------------------------------
# Isolation: each rollout gets its own env copy, and parallel == sequential
# ---------------------------------------------------------------------------

def test_each_rollout_gets_its_own_env_instance():
    created = []
    runner = ParallelRolloutRunner(make_env_factory(created), max_workers=4)
    specs = [RolloutSpec(seed=s, task=TASK, policy=increment_policy) for s in range(6)]
    runner.run(specs)
    assert len(created) == 6
    assert len({id(env) for env in created}) == 6


def test_parallel_results_match_sequential_per_seed():
    specs = [RolloutSpec(seed=s, task=TASK, policy=increment_policy) for s in range(8)]

    parallel = ParallelRolloutRunner(make_env_factory(), max_workers=8).run(specs)
    sequential = ParallelRolloutRunner(make_env_factory(), max_workers=1).run(specs)

    by_seed_parallel = {r.seed: (r.total_reward, r.steps, r.outcome) for r in parallel.records}
    by_seed_sequential = {r.seed: (r.total_reward, r.steps, r.outcome) for r in sequential.records}
    assert by_seed_parallel == by_seed_sequential


def test_rollouts_actually_run_concurrently():
    barrier = threading.Barrier(4, timeout=10)

    def blocking_policy(obs, action_types):
        barrier.wait()  # deadlocks unless 4 rollouts run at once
        return {"type": "increment"}

    # no task → no verifier → no early termination, so all 4 stay in lockstep
    specs = [RolloutSpec(seed=s, policy=blocking_policy) for s in range(4)]
    batch = ParallelRolloutRunner(make_env_factory(), max_workers=4).run(specs)
    assert len(batch.records) == 4


# ---------------------------------------------------------------------------
# Diversity: same task, classified outcomes
# ---------------------------------------------------------------------------

def test_same_task_produces_diverse_outcomes():
    # seeds give initial values 0..1000; +50/step, 5 steps → only some succeed
    specs = [RolloutSpec(seed=s, task=TASK, policy=increment_policy) for s in range(30)]
    batch = ParallelRolloutRunner(make_env_factory(), max_workers=8).run(specs)

    assert isinstance(batch, RolloutBatch)
    assert len(batch.records) == 30
    assert batch.outcome_counts.get("success", 0) > 0
    assert batch.outcome_counts.get("failure", 0) + batch.outcome_counts.get("partial_success", 0) > 0


def test_invalid_action_rollouts_classified_as_edge_case():
    def bad_policy(obs, action_types):
        return {"type": "summon_demon"}

    specs = [RolloutSpec(seed=1, task=TASK, policy=bad_policy)]
    batch = ParallelRolloutRunner(make_env_factory(), max_workers=1).run(specs)
    record = batch.records[0]
    assert record.outcome == "edge_case"
    assert record.invalid_actions > 0


def test_crashing_rollout_is_recorded_as_edge_case_not_lost():
    def crashing_policy(obs, action_types):
        raise RuntimeError("policy exploded")

    specs = [
        RolloutSpec(seed=1, task=TASK, policy=crashing_policy),
        RolloutSpec(seed=2, task=TASK, policy=increment_policy),
    ]
    batch = ParallelRolloutRunner(make_env_factory(), max_workers=2).run(specs)
    assert len(batch.records) == 2
    crashed = next(r for r in batch.records if r.seed == 1)
    assert crashed.outcome == "edge_case"
    assert "policy exploded" in (crashed.error or "")
    healthy = next(r for r in batch.records if r.seed == 2)
    assert healthy.error is None


def test_default_policy_is_seeded_random_over_action_types():
    # no policy given → deterministic seeded-random actions, so repeat runs match
    specs = [RolloutSpec(seed=s, task=TASK) for s in range(4)]
    first = ParallelRolloutRunner(make_env_factory(), max_workers=4).run(specs)
    second = ParallelRolloutRunner(make_env_factory(), max_workers=4).run(specs)
    assert [(r.seed, r.total_reward, r.steps) for r in first.records] == [
        (r.seed, r.total_reward, r.steps) for r in second.records
    ]


def test_run_diverse_convenience_builds_seed_range():
    runner = ParallelRolloutRunner(make_env_factory(), max_workers=4)
    batch = runner.run_diverse(task=TASK, num_rollouts=10, seed_start=100, policy=increment_policy)
    assert sorted(r.seed for r in batch.records) == list(range(100, 110))


def test_batch_filtering_and_summary():
    specs = [RolloutSpec(seed=s, task=TASK, policy=increment_policy) for s in range(20)]
    batch = ParallelRolloutRunner(make_env_factory(), max_workers=4).run(specs)
    successes = batch.by_outcome("success")
    assert all(r.outcome == "success" for r in successes)
    assert sum(batch.outcome_counts.values()) == 20
    # records come back ordered by seed regardless of completion order
    assert [r.seed for r in batch.records] == sorted(r.seed for r in batch.records)
