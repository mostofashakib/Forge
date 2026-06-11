# tests/runtime/test_agent_context.py
import json
import pytest
from forge.runtime.agents.agent_context import AgentContext, ContextDiagnosis


def big_obs(step: int) -> dict:
    return {
        "emails": {f"e_{i}": {"id": f"e_{i}", "subject": f"subject {i}", "body": "x" * 200} for i in range(30)},
        "step": step,
    }


def fill(ctx: AgentContext, n: int, hash_fn=lambda i: f"h{i}", reward_fn=lambda i: 0.0):
    for i in range(n):
        ctx.record(
            action={"type": "increment"},
            observation={"counter": i},
            state_hash=hash_fn(i),
            reward=reward_fn(i),
        )


# ---------------------------------------------------------------------------
# Compact digest
# ---------------------------------------------------------------------------

def test_digest_contains_recent_actions_and_latest_observation():
    ctx = AgentContext()
    ctx.record({"type": "open_email"}, {"open": True}, "h1")
    ctx.record({"type": "reply_email"}, {"replied": True}, "h2", reward=1.0)
    digest = ctx.digest()
    assert "open_email" in digest
    assert "reply_email" in digest
    assert "replied" in digest


def test_digest_compacts_large_observations():
    ctx = AgentContext(max_observation_chars=300)
    ctx.record({"type": "list_emails"}, big_obs(0), "h1")
    digest = ctx.digest()
    assert len(digest) < 2000
    assert "emails" in digest  # summarised, not dropped


def test_digest_is_deterministic_for_same_history():
    a, b = AgentContext(), AgentContext()
    for ctx in (a, b):
        ctx.record({"type": "open_email"}, {"z": 1, "a": 2}, "h1")
    assert a.digest() == b.digest()


# ---------------------------------------------------------------------------
# Stuck vs context-limit diagnosis
# ---------------------------------------------------------------------------

def test_fresh_context_is_ok():
    diagnosis = AgentContext().diagnose()
    assert isinstance(diagnosis, ContextDiagnosis)
    assert diagnosis.status == "ok"
    assert diagnosis.stuck is False
    assert diagnosis.context_limit_exceeded is False


def test_unchanged_state_hash_across_window_means_stuck():
    ctx = AgentContext(stuck_window=4)
    fill(ctx, 6, hash_fn=lambda i: "same_hash")
    diagnosis = ctx.diagnose()
    assert diagnosis.stuck is True
    assert diagnosis.status == "stuck"
    assert "no state change" in diagnosis.detail


def test_progressing_state_hashes_are_not_stuck():
    ctx = AgentContext(stuck_window=4)
    fill(ctx, 10)
    assert ctx.diagnose().stuck is False


def test_stuck_needs_full_window():
    ctx = AgentContext(stuck_window=4)
    fill(ctx, 3, hash_fn=lambda i: "same_hash")
    assert ctx.diagnose().stuck is False


def test_token_budget_exceeded_is_context_limit_not_stuck():
    ctx = AgentContext(max_tokens=50, auto_prune=False)
    for i in range(20):
        ctx.record({"type": f"act_{i}"}, {"data": "y" * 100, "i": i}, f"h{i}")
    diagnosis = ctx.diagnose()
    assert diagnosis.context_limit_exceeded is True
    assert diagnosis.stuck is False
    assert diagnosis.status == "context_limit_exceeded"


def test_stuck_and_over_budget_reports_stuck_first():
    ctx = AgentContext(max_tokens=50, stuck_window=3, auto_prune=False)
    for i in range(10):
        ctx.record({"type": "spin"}, {"data": "y" * 100}, "same_hash")
    diagnosis = ctx.diagnose()
    assert diagnosis.stuck is True
    assert diagnosis.context_limit_exceeded is True
    assert diagnosis.status == "stuck"


# ---------------------------------------------------------------------------
# Pruning — no garbage accumulation
# ---------------------------------------------------------------------------

def test_prune_drops_no_progress_entries_outside_recent_window():
    ctx = AgentContext(stuck_window=3, auto_prune=False)
    # 10 wasted steps (no state change, no reward), then 3 productive ones
    fill(ctx, 10, hash_fn=lambda i: "h0")
    for i in range(3):
        ctx.record({"type": "reply"}, {"i": i}, f"progress_{i}", reward=1.0)
    removed = ctx.prune()
    assert removed > 0
    assert len(ctx.entries) < 13
    # productive entries survive
    assert all(e.reward == 1.0 for e in ctx.entries if e.state_hash.startswith("progress"))


def test_prune_keeps_recent_entries_even_if_unproductive():
    ctx = AgentContext(stuck_window=5, auto_prune=False)
    fill(ctx, 4, hash_fn=lambda i: "h0")
    before = len(ctx.entries)
    ctx.prune()
    assert len(ctx.entries) == before  # all within the recent window


def test_prune_drops_error_entries_outside_recent_window():
    ctx = AgentContext(stuck_window=2, auto_prune=False)
    for i in range(5):
        ctx.record({"type": "bad_action"}, {"error": "UNKNOWN_TYPE"}, f"h{i}")
    ctx.record({"type": "reply"}, {"ok": True}, "h_final", reward=1.0)
    ctx.prune()
    old_errors = [e for e in ctx.entries[:-2] if e.is_error]
    assert old_errors == []


def test_auto_prune_keeps_context_within_budget_when_possible():
    ctx = AgentContext(max_tokens=400, stuck_window=2)
    for i in range(50):
        ctx.record({"type": "noop"}, {"filler": "z" * 50}, "h0")
        ctx.record({"type": "step"}, {"i": i}, f"h{i}")
    assert ctx.token_estimate <= 400 or len(ctx.entries) <= ctx.stuck_window + 1


def test_pruned_garbage_never_reappears_in_digest():
    ctx = AgentContext(stuck_window=2, auto_prune=False)
    for i in range(6):
        ctx.record({"type": "spam_refresh"}, {"noise": i}, "h0")
    ctx.record({"type": "reply"}, {"done": True}, "h1", reward=1.0)
    ctx.prune()
    digest = ctx.digest()
    assert digest.count("spam_refresh") <= 2  # only the protected recent window


def test_clear_resets_everything():
    ctx = AgentContext()
    fill(ctx, 5)
    ctx.clear()
    assert ctx.entries == []
    assert ctx.token_estimate == 0
    assert ctx.diagnose().status == "ok"
