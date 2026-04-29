import json
from forge.runtime.snapshot import StepSnapshot
from forge.runtime.trajectory import Trajectory, TrajectoryStore


def make_snapshot(episode_id: str, step_index: int, action_type: str) -> StepSnapshot:
    return StepSnapshot(
        episode_id=episode_id,
        step_index=step_index,
        state_hash_before="sha256:aaa",
        state_hash_after="sha256:bbb",
        action={"type": action_type},
        events=[{"type": "test_event", "entity_id": "x"}],
        reward=0.5,
        verifier_results=[],
        diff={"added": {}, "changed": {}, "removed": {}},
        terminated=False,
        truncated=False,
    )


def test_record_and_retrieve_steps():
    store = TrajectoryStore("ep_0000")
    store.record(make_snapshot("ep_0000", 0, "action_a"))
    store.record(make_snapshot("ep_0000", 1, "action_b"))
    traj = store.to_trajectory()
    assert len(traj.steps) == 2
    assert traj.steps[0].action["type"] == "action_a"


def test_events_flattens_all_step_events():
    store = TrajectoryStore("ep_0000")
    store.record(make_snapshot("ep_0000", 0, "a"))
    store.record(make_snapshot("ep_0000", 1, "b"))
    traj = store.to_trajectory()
    assert len(traj.events) == 2
    assert all(e["type"] == "test_event" for e in traj.events)


def test_to_jsonl_produces_one_json_object_per_line():
    store = TrajectoryStore("ep_0000")
    store.record(make_snapshot("ep_0000", 0, "a"))
    store.record(make_snapshot("ep_0000", 1, "b"))
    jsonl = store.to_jsonl()
    lines = jsonl.strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert obj["episode_id"] == "ep_0000"


def test_empty_trajectory_has_no_policy_violations():
    store = TrajectoryStore("ep_0000")
    assert store.to_trajectory().has_policy_violation is False
