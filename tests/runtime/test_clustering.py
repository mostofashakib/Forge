# tests/runtime/test_clustering.py
from __future__ import annotations
import json
from unittest.mock import MagicMock
from forge.runtime.clustering import FailureClusterer
from forge.runtime.replay import EpisodeRecord


def make_record(episode_id: str, passed: bool, first_failed_check: str | None) -> EpisodeRecord:
    ep = MagicMock()
    ep.id = episode_id
    ep.passed = passed

    if first_failed_check is not None:
        vr_json = json.dumps([{
            "verifier_id": "v1",
            "passed": False,
            "score": 0.0,
            "checks": [{"name": first_failed_check, "passed": False, "score": 0.0}],
        }])
        step = MagicMock()
        step.verifier_results = vr_json
        steps = [step]
    else:
        steps = []

    return EpisodeRecord(episode=ep, steps=steps)


def test_cluster_groups_by_first_failed_check():
    records = [
        make_record("ep_1", passed=False, first_failed_check="ticket_solved"),
        make_record("ep_2", passed=False, first_failed_check="ticket_solved"),
        make_record("ep_3", passed=False, first_failed_check="ticket_solved"),
        make_record("ep_4", passed=False, first_failed_check="comment_added"),
        make_record("ep_5", passed=False, first_failed_check="comment_added"),
    ]
    clusters = FailureClusterer().cluster(records)
    assert clusters[0].check_name == "ticket_solved"
    assert clusters[0].count == 3
    assert clusters[1].check_name == "comment_added"
    assert clusters[1].count == 2


def test_cluster_skips_passed_episodes():
    records = [
        make_record("ep_1", passed=True, first_failed_check="ticket_solved"),
        make_record("ep_2", passed=False, first_failed_check="ticket_solved"),
    ]
    clusters = FailureClusterer().cluster(records)
    assert len(clusters) == 1
    assert clusters[0].count == 1


def test_cluster_returns_at_most_5_clusters():
    records = [
        make_record(f"ep_{i}", passed=False, first_failed_check=f"check_{i}")
        for i in range(10)
    ]
    clusters = FailureClusterer().cluster(records)
    assert len(clusters) <= 5


def test_cluster_episode_ids_capped_at_5():
    records = [
        make_record(f"ep_{i}", passed=False, first_failed_check="same_check")
        for i in range(10)
    ]
    clusters = FailureClusterer().cluster(records)
    assert len(clusters[0].episode_ids) <= 5


def test_cluster_sorted_by_count_descending():
    records = [
        make_record("ep_1", passed=False, first_failed_check="rare"),
        make_record("ep_2", passed=False, first_failed_check="common"),
        make_record("ep_3", passed=False, first_failed_check="common"),
        make_record("ep_4", passed=False, first_failed_check="common"),
    ]
    clusters = FailureClusterer().cluster(records)
    assert clusters[0].check_name == "common"


def test_cluster_skips_episodes_with_no_failed_checks():
    records = [
        make_record("ep_1", passed=False, first_failed_check=None),
    ]
    clusters = FailureClusterer().cluster(records)
    assert clusters == []


def test_cluster_skips_steps_with_malformed_json():
    ep = MagicMock()
    ep.id = "ep_bad"
    ep.passed = False
    step = MagicMock()
    step.verifier_results = "not valid json"
    record = EpisodeRecord(episode=ep, steps=[step])
    clusters = FailureClusterer().cluster([record])
    assert clusters == []
