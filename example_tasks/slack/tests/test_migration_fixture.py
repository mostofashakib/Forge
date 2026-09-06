#!/usr/bin/env python3
"""Fixture invariants for the deep, dynamic Acme episode."""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

from dynamic_scenario import seeded
from slack_sim.clock import VIRTUAL_CLOCK, moment
from slack_sim.identity import LOGGED_IN_USER
from slack_sim.migration_truth import CLOSED_REVIEWS, CODE_REVIEWS, DEEP_EVIDENCE, OPEN_REVIEWS, REQUIRED_EVENTS, validate_against_seed
from slack_sim.seed import BENCHMARK_NOW, SLACK_CHANNELS, SLACK_LATENT_MESSAGES, SLACK_MESSAGES, SLACK_USERS
from slack_sim.service import export_state, get_channel_messages, get_thread_replies, list_channels, search_messages


def test_workspace_scale_and_clock(root: Path) -> None:
    db = seeded(root)
    state = export_state(db)
    assert len(SLACK_USERS) == 51
    assert len(SLACK_CHANNELS) == 24
    assert len(SLACK_MESSAGES) >= 1000
    assert len(state["messages"]) == len(SLACK_MESSAGES)
    assert BENCHMARK_NOW == moment(16, 15, 0)
    assert state["virtual_time"] == VIRTUAL_CLOCK.at(BENCHMARK_NOW)


def test_every_relevant_channel_has_long_running_top_level_history(_root: Path) -> None:
    channel_names = {channel.channel_id: channel.name for channel in SLACK_CHANNELS}
    top_level = Counter(
        message.conversation_id
        for message in SLACK_MESSAGES
        if message.thread_parent_id is None
    )
    required = {
        "acme-migration", "identity-eng", "data-ops", "enterprise-support", "debugging"
    }
    counts = {
        channel_names[channel_id]: count
        for channel_id, count in top_level.items()
        if channel_names.get(channel_id) in required
    }
    assert set(counts) == required
    assert all(count >= 200 for count in counts.values()), counts


def test_live_cutover_t0_contains_pressure_but_not_future_resolution(root: Path) -> None:
    db = seeded(root)
    state = export_state(db)
    by_id = {message["message_id"]: message for message in state["messages"]}
    assert {"CUT001", "CUT002", "CUT003", "CUT004", "CUT005", "CUT006", "CUT007"} <= set(by_id)
    pressure = "\n".join(by_id[message_id]["body"] for message_id in ("CUT004", "CUT005", "CUT006"))
    for deadline in ("4:30", "5:30", "6:30", "7:30", "8:30"):
        assert deadline in pressure
    assert "9:30" in by_id["CUT001"]["body"] and "reconfirmation" in by_id["CUT001"]["body"]
    assert "primary identity bridge coverage" in by_id["CUT003"]["body"]

    latent_ids = {message.message_id for message in SLACK_LATENT_MESSAGES}
    assert {"LAT021", "LAT022", "LAT027", "LAT031", "LAT033", "LAT035"} <= latent_ids
    assert latent_ids.isdisjoint(by_id)
    for phrase in (
        "9:30 PM was only", "rollback worker owner check", "Final migration rehearsal is running",
        "Corrected the recovery-config pin",
    ):
        assert search_messages(db, phrase, LOGGED_IN_USER.user_id)["total"] == 0


def test_deep_sso_and_access_context_contains_full_invariants(root: Path) -> None:
    db = seeded(root)
    actor = LOGGED_IN_USER.user_id
    sso = search_messages(db, "IDP-ACME-014", actor)
    sso_root = next(message for message in sso["matches"] if message["id"] == "MSG306")
    sso_thread = get_thread_replies(db, "C020", sso_root["thread_ts"], actor)
    sso_text = "\n".join(message["text"] for message in sso_thread["messages"])
    assert "acme-eu-2024" in sso_text and "acme-eu-2025" in sso_text
    assert "cache" in sso_text and "regional control plane" in sso_text

    access = search_messages(db, "ACME-ACCESS-04", actor)
    access_root = next(message for message in access["matches"] if message["id"] == "MSG795")
    access_thread = get_thread_replies(db, "C012", access_root["thread_ts"], actor)
    access_text = "\n".join(message["text"] for message in access_thread["messages"])
    for requirement in ("authenticate", "EU-Finance", "EU-Legal", "restricted area", "export action"):
        assert requirement in access_text


def test_deep_evidence_is_beyond_four_history_pages_but_searchable(root: Path) -> None:
    db = seeded(root)
    actor = LOGGED_IN_USER.user_id
    terms = {"sso": "IDP-ACME-014", "export": "ACME-DRYRUN-07", "permissions": "ACME-ACCESS-04", "pagination": "DIR-PAGE-311"}
    for name, item in DEEP_EVIDENCE.items():
        cursor = ""
        first_four: set[str] = set()
        for _ in range(4):
            page = get_channel_messages(db, item["channel"], actor, cursor=cursor)
            first_four.update(message["id"] for message in page["messages"])
            cursor = page["next_cursor"]
            assert cursor
        assert item["root"] not in first_four
        found = {message["id"] for message in search_messages(db, terms[name], actor)["matches"]}
        assert item["root"] in found


def test_latent_messages_are_completely_unobservable_at_t0(root: Path) -> None:
    db = seeded(root)
    actor = LOGGED_IN_USER.user_id
    state = export_state(db)
    latent_ids = {message.message_id for message in SLACK_LATENT_MESSAGES}
    assert latent_ids.isdisjoint({message["message_id"] for message in state["messages"]})
    assert {event["event_id"] for event in state["scenario_events"]} == REQUIRED_EVENTS
    assert all(event["status"] == "pending" and event["activated_ts"] is None for event in state["scenario_events"])
    for phrase in ("12,481,991", "EU-2 is no longer", "missing entitlement acme-eu-legal-export", "reconciliation assertion"):
        result = search_messages(db, phrase, actor)
        assert result["total"] == 0 and result["matches"] == []
    channels = list_channels(db, actor)["channels"]
    assert all(channel["unread_count"] >= 0 for channel in channels)
    debug = get_channel_messages(db, "C023", actor)
    saml_root = next(message for message in debug["messages"] if message["id"] == "MSG194")
    thread = get_thread_replies(db, "C023", saml_root["thread_ts"], actor)
    assert saml_root["reply_count"] == len(thread["messages"]) - 1
    assert latent_ids.isdisjoint({message["id"] for message in thread["messages"]})


def test_review_mix_and_seed_validation(_root: Path) -> None:
    validate_against_seed()
    assert len(OPEN_REVIEWS) == 5
    assert len(CLOSED_REVIEWS) == 2
    assert CODE_REVIEWS["Retry"]["expected"] == "LGTM"
    assert CODE_REVIEWS["provisionAllUsers"]["expected"] == "CRITIQUE"


def test_the_cutover_bridge_is_discoverable_but_not_joined(root: Path) -> None:
    """Ben has to find the bridge and join it; nothing tells him to."""
    from slack_sim.seed import BRIDGE_CHANNEL, SLACK_MESSAGES, build_slack_memberships

    roster = {m.user_id for m in build_slack_memberships() if m.channel_id == BRIDGE_CHANNEL}
    assert LOGGED_IN_USER.user_id not in roster
    assert "U051" not in roster, "the identity backup is the gap, not a member"
    assert {"U041", "U042", "U044", "U045", "U046", "U047"} <= roster

    visible_to_ben = [
        m for m in SLACK_MESSAGES
        if m.conversation_id != BRIDGE_CHANNEL and "acme-cutover-bridge" in m.text
    ]
    assert visible_to_ben, "a channel he cannot see must be referenced somewhere he can"
    for message in SLACK_MESSAGES:
        assert "join #acme-cutover-bridge" not in message.text.lower(), (
            "the seed points at the channel; it does not issue instructions"
        )


def test_the_bridge_carries_the_work_that_moved_there(root: Path) -> None:
    from slack_sim.seed import BRIDGE_CHANNEL, SLACK_LATENT_MESSAGES, SLACK_MESSAGES

    visible = [m for m in SLACK_MESSAGES if m.conversation_id == BRIDGE_CHANNEL]
    latent = [m for m in SLACK_LATENT_MESSAGES if m.conversation_id == BRIDGE_CHANNEL]
    assert len(visible) >= 8 and len(latent) >= 10
    body = " ".join(m.text.lower() for m in visible)
    for topic in ("rehearsal", "rollback", "coverage", "go/no-go", "5:30"):
        assert topic in body, topic


def test_seeded_side_state_is_lived_in_rather_than_authoritative(root: Path) -> None:
    from slack_sim.seed import (
        SLACK_NOTIFICATIONS, SLACK_PINS, SLACK_SAVED_ITEMS, SLACK_THREAD_FOLLOWS,
    )

    pinned = {pin.message_id for pin in SLACK_PINS}
    assert "MSG125" in pinned, "a stale pin, so pinned does not mean current"
    assert "MSG135" in pinned, "and a current one, so it does not mean stale either"

    ben = [n for n in SLACK_NOTIFICATIONS if n.user_id == LOGGED_IN_USER.user_id]
    assert 8 <= len(ben) <= 12
    assert len({n.kind for n in ben}) >= 3, "a real inbox is not all one kind"

    followed = {f.thread_id for f in SLACK_THREAD_FOLLOWS}
    saved = {item.message_id for item in SLACK_SAVED_ITEMS}
    assert not (followed & {"MSG194", "MSG200", "MSG214", "MSG219", "MSG224"})
    assert not any(mid.startswith("LAT") for mid in followed | saved | pinned)


def test_latent_side_state_names_only_real_targets(root: Path) -> None:
    from slack_sim.seed import ACME_SCENARIO, SLACK_LATENT_MESSAGES, SLACK_USERS

    latent_ids = {m.message_id for m in SLACK_LATENT_MESSAGES}
    user_ids = {u.user_id for u in SLACK_USERS}
    for notification in ACME_SCENARIO.latent_notifications:
        assert notification.message_id in latent_ids, notification.notification_id
        assert notification.user_id in user_ids
    for membership in ACME_SCENARIO.latent_memberships:
        assert membership.user_id in user_ids


def test_the_identity_backup_is_distinguishable_from_the_other_felix(root: Path) -> None:
    from slack_sim.seed import SLACK_USER_PROFILES, SLACK_USERS

    felixes = [u for u in SLACK_USERS if "Felix" in u.display_name]
    assert len(felixes) == 2, "the QA Felix stays; the identity one is new"
    titles = {p.user_id: p.title for p in SLACK_USER_PROFILES}
    assert len({titles[u.user_id] for u in felixes}) == 2
    assert len({u.team for u in felixes}) == 2


def main() -> None:
    tests = sorted((value for name, value in globals().items() if name.startswith("test_") and callable(value)), key=lambda fn: fn.__name__)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, test in enumerate(tests):
            test(root / str(index))
            print(f"  {test.__name__}: ok")
    print("migration fixture: ok")


if __name__ == "__main__":
    main()
