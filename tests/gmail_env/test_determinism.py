# tests/gmail_env/test_determinism.py
import json
from examples.gmail_env.gym_wrapper import build_gmail_env


def test_same_seed_and_actions_produce_identical_trajectory():
    task = {
        "name": "reply_to_customer",
        "verifier_id": "reply_to_customer",
        "inputs": {},  # thread_id filled in after reset
    }

    env1 = build_gmail_env()
    obs1, info1 = env1.reset(seed=42, options={"task": task})
    thread_id = next(iter(obs1["threads"]))
    task_with_id = {**task, "inputs": {"thread_id": thread_id}}

    env1._current_task = task_with_id
    obs1_a, r1, t1, tr1, i1 = env1.step({"type": "reply_email", "thread_id": thread_id, "body": "Hello"})
    hash1 = env1._traj_store._steps[-1].state_hash_after

    env2 = build_gmail_env()
    obs2, info2 = env2.reset(seed=42, options={"task": task})
    env2._current_task = task_with_id
    obs2_a, r2, t2, tr2, i2 = env2.step({"type": "reply_email", "thread_id": thread_id, "body": "Hello"})
    hash2 = env2._traj_store._steps[-1].state_hash_after

    assert hash1 == hash2
    assert r1 == r2
    assert t1 == t2


def test_different_seeds_produce_different_episode_ids():
    env = build_gmail_env()
    _, info1 = env.reset(seed=1)
    _, info2 = env.reset(seed=2)
    assert info1["episode_id"] != info2["episode_id"]


def test_trajectory_exports_valid_jsonl():
    env = build_gmail_env()
    task = {"name": "archive_newsletter", "verifier_id": "archive_newsletter", "inputs": {}}
    obs, info = env.reset(seed=7, options={"task": task})
    email_id = next(iter(obs["emails"]))
    env._current_task = {**task, "inputs": {"email_id": email_id}}
    env.step({"type": "archive_email", "email_id": email_id})

    jsonl = env._traj_store.to_jsonl()
    lines = jsonl.strip().split("\n")
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["episode_id"] == info["episode_id"]
    assert "state_hash_before" in record
    assert "state_hash_after" in record
    assert "diff" in record
    assert "reward" in record


def test_invalid_action_does_not_corrupt_trajectory_hashes():
    env = build_gmail_env()
    obs, info = env.reset(seed=5)
    email_id = next(iter(obs["emails"]))

    env.step({"type": "mark_read", "email_id": email_id})
    hash_after_valid = env._traj_store._steps[-1].state_hash_after

    env.step({"type": "nonexistent_action"})
    hash_after_invalid = env._traj_store._steps[-1].state_hash_after

    assert hash_after_valid == hash_after_invalid


def test_episode_task_completion_produces_positive_reward():
    env = build_gmail_env()
    task = {"name": "reply_to_customer", "verifier_id": "reply_to_customer", "inputs": {}}
    obs, info = env.reset(seed=10, options={"task": task})
    thread_id = next(iter(obs["threads"]))
    env._current_task = {**task, "inputs": {"thread_id": thread_id}}

    _, reward, terminated, _, _ = env.step(
        {"type": "reply_email", "thread_id": thread_id, "body": "Thank you for reaching out."}
    )
    assert terminated is True
    assert reward > 0.0


def test_five_seeds_all_produce_distinct_state_hashes():
    env = build_gmail_env()
    hashes = set()
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        from forge.runtime.state import StateStore
        store = StateStore(obs)
        hashes.add(store.hash())
    assert len(hashes) == 5
