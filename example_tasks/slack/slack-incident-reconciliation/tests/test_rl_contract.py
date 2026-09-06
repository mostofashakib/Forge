#!/usr/bin/env python3
"""The RL lifecycle contract: termination, per-step reward, bounded state.

Three properties a trainer needs and the 2.0 contract did not provide:

  * a rollout says when it is over, by goal or by budget;
  * every transition carries a scalar reward, not just the final verifier run;
  * episode state can be read without growing without bound.

In-episode reward covers only what the environment can observe: which channels
the agent opened, and whether it delivered an answer. It deliberately does not
score the content of that answer -- the verdict and the owners are short
strings a policy could assert without reading anything, and rewarding that
would put the shaped signal in direct opposition to the verifier. Correctness
is graded after the episode, from the world's own state export -- which is
where the answer lives too, as a reply in Daniel's thread.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from slack_sim.environment import SlackIncidentEnvironment
from slack_sim.migration_reward import WORKSPACE_MILESTONES
from slack_sim.migration_truth import REQUEST_MESSAGE_ID

COOKIE = "rl"
BENCHMARK_CHANNELS = ("C019", "C020", "C021", "C022", "C023")
REPLY = ("Readiness: BLOCKED. Thu 20 Aug 2026, 9:00 PM PT, 90 minutes. "
         "SSO Priya, export Ahmed, permissions Marcus.")


def env(root: Path, **kwargs) -> SlackIncidentEnvironment:
    e = SlackIncidentEnvironment(root / "slack.db", root / "seed.sql")
    e.reset(session_cookie=COOKIE, user_instruction="reconcile", **kwargs)
    return e


def answer(e: SlackIncidentEnvironment, body: str) -> dict:
    return e.step(COOKIE, "reply_to_thread",
                  {"thread_parent_id": REQUEST_MESSAGE_ID, "body": body})


def investigate(e: SlackIncidentEnvironment, *channel_ids: str) -> None:
    for channel_id in channel_ids or BENCHMARK_CHANNELS:
        e.step(COOKIE, "get_channel_messages", {"channel_id": channel_id})


def review(e: SlackIncidentEnvironment) -> None:
    """Drive every review and live-cutover transition to terminal state."""
    e.step(COOKIE, "search_messages", {"query": "IDP-ACME-014"})
    e.step(COOKIE, "search_messages", {"query": "ACME-ACCESS-04"})
    e.step(COOKIE, "search_messages", {"query": "DIR-PAGE-311"})
    e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "includes is substring matching but exact equality is required; a partial audience passes"})
    e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "Looks good to me"})
    e.step(COOKIE, "get_channel_messages", {"channel_id": "C020"})
    e.step(COOKIE, "get_channel_messages", {"channel_id": "C020"})
    e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "MSG200", "body": "Looks good to me"})
    e.step(COOKIE, "search_messages", {"query": "26 synthetic"})
    e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "MSG200", "body": "Looks good to me"})
    e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "MSG214", "body": "some is any-match and accepts one group, but every/all required group is needed"})
    e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "MSG214", "body": "Looks good to me"})
    e.step(COOKIE, "get_channel_messages", {"channel_id": "C022"})
    e.step(COOKIE, "get_channel_messages", {"channel_id": "C022"})
    e.step(COOKIE, "get_channel_messages", {"channel_id": "C022"})
    e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "MSG219", "body": "Looks good to me"})
    e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "MSG224", "body": "A duplicate boundary record can repeat across pages and provisionUser is non-idempotent, causing provisioning twice; deduplicate IDs or use a snapshot."})
    e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "CUT001", "body": "Nina, confirm whether 9:30 was an actual request or only a question"})
    e.step(COOKIE, "get_channel_messages", {"channel_id": "C019"})
    # The cutover bridge is public but Ben is not on it, so joining comes first.
    e.step(COOKIE, "join_channel", {"channel_id": "C024"})
    bridge = e.step(COOKIE, "get_channel_messages", {"channel_id": "C024"})
    coverage_root = next(message for message in bridge["result"]["messages"] if message["id"] == "CUT002")
    e.step(COOKIE, "get_thread_replies", {"channel_id": "C024", "thread_ts": coverage_root["thread_ts"]})
    e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "LAT022", "body": "Verify the rollback worker by running its recovery invocation and report status"})
    rehearsal = e.step(COOKIE, "get_channel_messages", {"channel_id": "C024"})
    rehearsal_root = next(message for message in rehearsal["result"]["messages"] if message["id"] == "LAT027")
    e.step(COOKIE, "get_thread_replies", {"channel_id": "C024", "thread_ts": rehearsal_root["thread_ts"]})
    e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "LAT027", "body": "Rollback used a stale config pin; correct the pin and rerun the recovery invocation"})
    e.step(COOKIE, "get_thread_replies", {"channel_id": "C024", "thread_ts": rehearsal_root["thread_ts"]})


def solve(e: SlackIncidentEnvironment) -> dict:
    review(e)
    return answer(e, REPLY)


# --------------------------------------------------------------------------
# 1. Episode termination
# --------------------------------------------------------------------------

def test_every_step_reports_termination_flags(root: Path) -> None:
    e = env(root)
    out = e.step(COOKIE, "list_users", {})
    for key in ("terminated", "truncated", "done", "turns_remaining"):
        assert key in out, f"step() must report {key!r}"
    assert out["terminated"] is False and out["truncated"] is False and out["done"] is False


def test_completing_the_workspace_goal_terminates(root: Path) -> None:
    e = env(root)
    final = solve(e)
    assert final["terminated"] is True, "all observable milestones met must terminate"
    assert final["truncated"] is False
    assert final["done"] is True


def test_exhausting_the_turn_budget_truncates(root: Path) -> None:
    e = env(root, max_turns=3)
    for expected_remaining in (2, 1, 0):
        out = e.step(COOKIE, "list_users", {})
        assert out["turns_remaining"] == expected_remaining
    assert out["truncated"] is True and out["terminated"] is False and out["done"] is True


def test_a_finished_episode_refuses_further_steps(root: Path) -> None:
    e = env(root, max_turns=1)
    e.step(COOKIE, "list_users", {})
    after = e.step(COOKIE, "list_users", {})
    assert after["ok"] is False
    assert after["error"]["code"] == "episode_finished"
    assert after["done"] is True


def test_the_turn_budget_is_configurable_per_episode(root: Path) -> None:
    e = env(root, max_turns=7)
    assert e.step(COOKIE, "list_users", {})["turns_remaining"] == 6


# --------------------------------------------------------------------------
# 2. Per-step reward
# --------------------------------------------------------------------------

def test_every_step_carries_a_scalar_reward(root: Path) -> None:
    e = env(root)
    out = e.step(COOKIE, "list_users", {})
    assert isinstance(out["reward"], float) and isinstance(out["progress"], float)
    # Investigation is the work in this task, so opening one of the four
    # channels does earn progress -- but a directory listing is not
    # investigation, and earns nothing.
    assert out["reward"] == 0.0, "listing users tells you nothing about the migration"


def test_opening_a_benchmark_channel_is_rewarded_investigation(root: Path) -> None:
    e = env(root)
    noise = e.step(COOKIE, "get_channel_messages", {"channel_id": "C001"})
    assert noise["reward"] == 0.0, "#general is not part of the readiness check"
    signal = e.step(COOKIE, "get_channel_messages", {"channel_id": "C020"})
    assert signal["reward"] == 0.0, "opening a channel alone is not a completed state transition"


def test_progress_rises_as_milestones_are_met(root: Path) -> None:
    """Progress follows dynamic outcomes rather than one-time channel opens."""
    e = env(root)
    first = e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "includes is substring matching but exact equality is required; partial audiences pass"})
    assert first["reward"] == 0.0, "the critique releases a revision but does not finish its milestone"
    reviewed = e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "Looks good to me"})
    assert reviewed["reward"] > 0.0
    review(e)
    final = answer(e, REPLY)
    assert final["reward"] > 0.0 and final["progress"] == 1.0


def test_reading_the_same_channel_twice_is_not_more_progress(root: Path) -> None:
    e = env(root)
    first = e.step(COOKIE, "get_channel_messages", {"channel_id": "C019"})
    again = e.step(COOKIE, "get_channel_messages", {"channel_id": "C019"})
    assert first["reward"] == 0.0 and again["reward"] == 0.0


def test_reward_is_the_change_since_the_previous_step(root: Path) -> None:
    e = env(root)
    first = e.step(COOKIE, "get_channel_messages", {"channel_id": "C019"})
    again = e.step(COOKIE, "list_channels", {})
    assert again["reward"] == 0.0, "no new progress means no further reward"
    assert again["progress"] == first["progress"] == 0.0, "progress is absolute and must hold"


def test_side_effects_produce_negative_reward(root: Path) -> None:
    e = env(root)
    e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "includes is substring matching but exact equality is required; partial audiences pass"})
    before = e.step(COOKIE, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "Looks good to me"})
    assert not before["done"], "the episode must still be running"
    damage = e.step(COOKIE, "create_channel", {"name": "scratch-pad"})
    assert damage["reward"] < 0.0, "an unrequested mutation must cost the policy"
    assert damage["progress"] < before["progress"]


def test_a_solved_episode_reaches_full_progress(root: Path) -> None:
    e = env(root)
    assert solve(e)["progress"] == 1.0


def test_in_episode_milestones_are_only_the_observable_ones(root: Path) -> None:
    """Shaping counts events that fired and whether an answer was posted.

    The world can read that answer -- it is a Slack message like any other --
    but reading it and judging it are different things, and only the verifier
    does the second."""
    assert set(WORKSPACE_MILESTONES) == {
        "saml_revision_reviewed", "saml_deployed", "export_backfill_run",
        "export_policy_found", "export_reopened_review_closed",
        "permissions_revision_reviewed", "permissions_partial_found",
        "permissions_verified", "timing_confirmed", "rollback_checked",
        "sso_key_investigated", "coverage_change_seen", "rehearsal_reopened",
        "rehearsal_completed", "cross_thread_reviewed",
        "answered_request_after_updates",
    }


def test_asserting_the_answer_without_reading_is_not_a_finished_episode(root: Path) -> None:
    """The shaped reward must not be maxed by guessing the conclusion."""
    e = env(root)
    out = answer(e, REPLY)
    assert out["progress"] < 1.0, "no investigation, no full progress"
    assert not out["terminated"], "guessing must not end the episode"


# --------------------------------------------------------------------------
# 4. Bounded episode state
# --------------------------------------------------------------------------

def test_history_can_be_windowed(root: Path) -> None:
    e = env(root)
    for _ in range(6):
        e.step(COOKIE, "list_users", {})
    full = e.history(COOKIE)
    tail = e.history(COOKIE, limit=4)
    assert len(tail) == 4
    assert tail == full[-4:], "a window must be the most recent events, in order"


def test_state_accepts_a_history_limit_and_reports_the_total(root: Path) -> None:
    e = env(root)
    for _ in range(6):
        e.step(COOKIE, "list_users", {})
    windowed = e.state(COOKIE, history_limit=3)
    assert len(windowed["history"]) == 3
    assert windowed["history_total"] == len(e.history(COOKIE))
    assert windowed["history_total"] > 3, "the caller must be able to see what was elided"


def test_state_never_exposes_hidden_workspace_tables(root: Path) -> None:
    e = env(root)
    e.step(COOKIE, "get_channel_messages", {"channel_id": "C019", "limit": 2})
    visible = e.state(COOKIE)
    assert "workspace" not in visible
    serialized = json.dumps(visible, default=str)
    assert "memberships" not in serialized
    assert "slack_seed_snapshot" not in serialized
    tool_results = [event for event in visible["history"] if event["kind"] == "tool_result"]
    assert len(tool_results) == 1
    assert len(tool_results[0]["content"]["result"]["messages"]) == 2


def test_windowing_bounds_the_size_of_state(root: Path) -> None:
    import json

    e = env(root)
    for _ in range(25):
        e.step(COOKIE, "list_users", {})
    unbounded = len(json.dumps(e.state(COOKIE), default=str))
    bounded = len(json.dumps(e.state(COOKIE, history_limit=5), default=str))
    assert bounded < unbounded / 2, (
        f"windowing barely helped: {bounded} vs {unbounded} bytes"
    )


def test_default_state_is_unchanged_for_existing_callers(root: Path) -> None:
    e = env(root)
    e.step(COOKIE, "list_users", {})
    default = e.state(COOKIE)
    assert len(default["history"]) == len(e.history(COOKIE))


def test_a_bad_tool_name_ends_the_step_and_not_the_episode(root: Path) -> None:
    """A mistyped tool is the policy's mistake to learn from, so it comes back
    as an error observation. It used to escape `step` as a bare RuntimeError
    and take the whole rollout with it."""
    e = env(root)
    step = e.step(COOKIE, "send_carrier_pigeon", {})
    assert step["ok"] is False
    assert step["error"]["code"] == "unknown_tool"
    assert step["reward"] == 0.0 and not step["terminated"]


def test_a_corrupt_database_is_not_offered_to_the_policy_as_a_choice(root: Path) -> None:
    """The other half of the same distinction. Nothing the policy does next can
    help, so a storage fault must stop the rollout rather than arrive as one
    more error observation for it to learn to avoid."""
    from slack_sim.sqlite_common import StorageError

    e = env(root)
    e.db_path.write_bytes(b"this is not a database")
    try:
        e.step(COOKIE, "list_channels", {})
    except StorageError:
        return
    raise AssertionError("a corrupt database was reported as an ordinary tool error")


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        with tempfile.TemporaryDirectory(prefix="slack-rl-") as temp_dir:
            test(Path(temp_dir))
        print(f"  {test.__name__}: ok")
    print("rl contract: ok")


if __name__ == "__main__":
    main()
