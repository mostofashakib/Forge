"""The persona configuration endpoints."""
from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    from backend.app import database

    database._engine = None
    database._SessionLocal = None
    database.init_db()
    return TestClient(app)


@pytest.fixture
def ward(tmp_path, monkeypatch):
    """A generated environment with two actions and a reward block."""
    envs_dir = tmp_path / "generated_envs"
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(envs_dir))
    env_dir = envs_dir / "ward"
    (env_dir / "custom").mkdir(parents=True)
    (env_dir / "transitions").mkdir(parents=True)
    (env_dir / "transitions" / "__init__.py").write_text("")
    (env_dir / "transitions" / "post_message.py").write_text("")
    (env_dir / "transitions" / "review_chart.py").write_text("")
    (env_dir / "custom" / "config.yaml").write_text("reward:\n  base_success: 2.0\n")
    return env_dir


def nurse(**behavior):
    return {
        "id": "nurse",
        "name": "Priya",
        "role": "charge nurse",
        "behavior": {"allowed_actions": ["post_message"], **behavior},
    }


# --- reading --------------------------------------------------------------


def test_an_environment_with_no_cast_reports_a_disabled_population(client, ward):
    body = client.get("/api/envs/ward/personas").json()
    assert body["personas"]["enabled"] is False
    assert body["personas"]["roster"] == []


def test_the_environment_action_surface_is_reported_for_the_editor(client, ward):
    body = client.get("/api/envs/ward/personas").json()
    assert body["environment_actions"] == ["post_message", "review_chart"]


def test_no_premade_cast_is_pushed_at_the_author(client, ward):
    """People are written, not chosen from a list — see PersonaCastPicker."""
    assert "archetypes" not in client.get("/api/envs/ward/personas").json()


def test_an_unknown_environment_is_a_404(client, ward):
    assert client.get("/api/envs/nope/personas").status_code == 404


def test_a_traversal_attempt_is_rejected(client, ward):
    assert client.get("/api/envs/..%2F..%2Fetc/personas").status_code in (400, 404)


def test_unreadable_yaml_is_reported_rather_than_swallowed(client, ward):
    (ward / "custom" / "config.yaml").write_text("reward: [unclosed\n")
    assert client.get("/api/envs/ward/personas").status_code == 422


# --- writing --------------------------------------------------------------


def test_saving_a_cast_persists_it(client, ward):
    response = client.put(
        "/api/envs/ward/personas",
        json={"personas": {"enabled": True, "roster": [nurse()]}},
    )
    assert response.status_code == 200
    saved = yaml.safe_load((ward / "custom" / "config.yaml").read_text())
    assert saved["personas"]["roster"][0]["id"] == "nurse"


def test_saving_a_cast_leaves_the_rest_of_the_config_intact(client, ward):
    client.put(
        "/api/envs/ward/personas",
        json={"personas": {"enabled": True, "roster": [nurse()]}},
    )
    saved = yaml.safe_load((ward / "custom" / "config.yaml").read_text())
    assert saved["reward"]["base_success"] == 2.0


def test_a_saved_cast_reads_back_identically(client, ward):
    payload = {
        "enabled": True,
        "count": 3,
        "driver": "anthropic:claude-sonnet-5",
        "roster": [nurse(wake_on=["post_message"], latency_steps=2)],
        "archetypes": [
            {
                "archetype": "anxious_requester",
                "behavior": {"allowed_actions": ["post_message"]},
            }
        ],
    }
    written = client.put("/api/envs/ward/personas", json={"personas": payload}).json()
    read_back = client.get("/api/envs/ward/personas").json()
    assert read_back["personas"] == written["personas"]


def test_a_cast_bound_to_an_action_the_environment_lacks_is_refused(client, ward):
    response = client.put(
        "/api/envs/ward/personas",
        json={
            "personas": {
                "enabled": True,
                "roster": [{"id": "n", "name": "N", "behavior": {"allowed_actions": ["send_page"]}}],
            }
        },
    )
    assert response.status_code == 422
    assert "send_page" in response.json()["detail"]
    assert "post_message" in response.json()["detail"]


def test_a_refused_save_does_not_write_anything(client, ward):
    before = (ward / "custom" / "config.yaml").read_text()
    client.put(
        "/api/envs/ward/personas",
        json={
            "personas": {
                "roster": [{"id": "n", "name": "N", "behavior": {"allowed_actions": ["nope"]}}]
            }
        },
    )
    assert (ward / "custom" / "config.yaml").read_text() == before


def test_a_misspelled_trait_is_refused_with_the_field_named(client, ward):
    response = client.put(
        "/api/envs/ward/personas",
        json={"personas": {"roster": [{"id": "n", "name": "N", "traits": {"dilgence": 90}}]}},
    )
    assert response.status_code == 422
    assert "dilgence" in response.json()["detail"]


def test_an_out_of_range_trait_is_refused(client, ward):
    response = client.put(
        "/api/envs/ward/personas",
        json={"personas": {"roster": [{"id": "n", "name": "N", "traits": {"diligence": 900}}]}},
    )
    assert response.status_code == 422


# --- preview --------------------------------------------------------------


def test_preview_shows_the_cast_a_seed_actually_produces(client, ward):
    response = client.post(
        "/api/envs/ward/personas/preview",
        json={
            "personas": {
                "enabled": True,
                "count": 3,
                "roster": [nurse()],
                "archetypes": [
                    {
                        "archetype": "anxious_requester",
                        "behavior": {"allowed_actions": ["post_message"]},
                    }
                ],
            },
            "seed": 42,
        },
    )
    assert response.status_code == 200
    roster = response.json()["roster"]
    assert len(roster) == 3
    assert roster[0]["id"] == "nurse"


def test_preview_is_stable_for_the_same_seed(client, ward):
    payload = {
        "personas": {
            "enabled": True,
            "count": 4,
            "archetypes": [
                {"archetype": "anxious_requester", "behavior": {"allowed_actions": ["post_message"]}}
            ],
        },
        "seed": 7,
    }
    first = client.post("/api/envs/ward/personas/preview", json=payload).json()
    second = client.post("/api/envs/ward/personas/preview", json=payload).json()
    assert first == second


def test_preview_warns_about_a_persona_that_can_never_act(client, ward):
    response = client.post(
        "/api/envs/ward/personas/preview",
        json={
            "personas": {
                "enabled": True,
                "roster": [{"id": "n", "name": "Quiet", "behavior": {"allowed_actions": []}}],
            },
            "seed": 1,
        },
    )
    warnings = response.json()["warnings"]
    assert any("never act" in w for w in warnings)


def test_preview_does_not_warn_about_a_workable_cast(client, ward):
    """False-positive guard: warnings must mean something."""
    response = client.post(
        "/api/envs/ward/personas/preview",
        json={"personas": {"enabled": True, "roster": [nurse()]}, "seed": 1},
    )
    assert response.json()["warnings"] == []


def test_preview_refuses_a_count_it_cannot_fill(client, ward):
    response = client.post(
        "/api/envs/ward/personas/preview",
        json={"personas": {"enabled": True, "count": 5, "roster": [nurse()]}, "seed": 1},
    )
    assert response.status_code == 422
    assert "archetypes" in response.json()["detail"]


def test_preview_does_not_write_to_disk(client, ward):
    before = (ward / "custom" / "config.yaml").read_text()
    client.post(
        "/api/envs/ward/personas/preview",
        json={"personas": {"enabled": True, "roster": [nurse()]}, "seed": 1},
    )
    assert (ward / "custom" / "config.yaml").read_text() == before


# --- the launcher's on/off switch ----------------------------------------


def test_the_switch_turns_a_configured_cast_on(client, ward):
    client.put(
        "/api/envs/ward/personas",
        json={"personas": {"enabled": False, "roster": [nurse()]}},
    )
    response = client.put("/api/envs/ward/personas/enabled", json={"enabled": True})
    assert response.status_code == 200
    assert response.json()["personas"]["enabled"] is True
    saved = yaml.safe_load((ward / "custom" / "config.yaml").read_text())
    assert saved["personas"]["enabled"] is True


def test_the_switch_turns_a_cast_off_again(client, ward):
    client.put(
        "/api/envs/ward/personas",
        json={"personas": {"enabled": True, "roster": [nurse()]}},
    )
    response = client.put("/api/envs/ward/personas/enabled", json={"enabled": False})
    assert response.json()["personas"]["enabled"] is False


def test_the_switch_does_not_disturb_the_roster(client, ward):
    """A launcher that sent the whole population back would clobber edits."""
    client.put(
        "/api/envs/ward/personas",
        json={
            "personas": {
                "enabled": False,
                "count": 3,
                "driver": "anthropic:claude-sonnet-5",
                "roster": [nurse(wake_on=["post_message"], latency_steps=4)],
            }
        },
    )
    before = client.get("/api/envs/ward/personas").json()["personas"]
    client.put("/api/envs/ward/personas/enabled", json={"enabled": True})
    after = client.get("/api/envs/ward/personas").json()["personas"]
    assert after["roster"] == before["roster"]
    assert after["count"] == 3
    assert after["driver"] == "anthropic:claude-sonnet-5"


def test_the_switch_leaves_the_rest_of_the_config_intact(client, ward):
    client.put(
        "/api/envs/ward/personas",
        json={"personas": {"enabled": False, "roster": [nurse()]}},
    )
    client.put("/api/envs/ward/personas/enabled", json={"enabled": True})
    saved = yaml.safe_load((ward / "custom" / "config.yaml").read_text())
    assert saved["reward"]["base_success"] == 2.0


def test_turning_on_an_empty_cast_is_refused_with_a_next_step(client, ward):
    """Switching on nobody would silently do nothing — say so instead."""
    response = client.put("/api/envs/ward/personas/enabled", json={"enabled": True})
    assert response.status_code == 422
    assert "Simulated People page" in response.json()["detail"]


def test_turning_off_an_empty_cast_is_harmless(client, ward):
    """False-positive guard: the refusal above must not block the off direction."""
    assert (
        client.put("/api/envs/ward/personas/enabled", json={"enabled": False}).status_code
        == 200
    )


def test_the_switch_on_an_unknown_environment_is_a_404(client, ward):
    assert (
        client.put("/api/envs/nope/personas/enabled", json={"enabled": True}).status_code
        == 404
    )
