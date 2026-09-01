"""Choosing a cast while the environment is being built.

At creation time the environment does not exist, so nobody can be granted an
action — the app's endpoints only appear once it is generated. These tests pin
that constraint at the request boundary, the write that happens when the build
finishes, and the action surface the personas page reads back afterwards.
"""
from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from backend.app.api.personas import _container_endpoints, _environment_actions
from backend.app.api.sandbox import CreateSandboxRequest
from backend.app.main import app
from backend.app.worker.tasks import _write_personas


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    from backend.app import database

    database._engine = None
    database._SessionLocal = None
    database.init_db()
    return TestClient(app)


def cast(**overrides):
    payload = {
        "enabled": True,
        "driver": "scripted",
        "roster": [{"archetype": "meticulous_nurse", "id": "nurse"}],
    }
    payload.update(overrides)
    return payload


# --- the archetype library, available before any environment exists --------


def test_the_library_is_offered_without_an_environment(client):
    body = client.get("/api/persona-archetypes").json()
    assert "meticulous_nurse" in [a["id"] for a in body["archetypes"]]


def test_library_archetypes_arrive_unable_to_act(client):
    body = client.get("/api/persona-archetypes").json()
    assert all(a["behavior"]["allowed_actions"] == [] for a in body["archetypes"])


# --- the create request ----------------------------------------------------


def request_with(personas):
    return CreateSandboxRequest(env_name="ward", description="a ward", personas=personas)


def test_a_cast_is_accepted_on_the_create_request():
    req = request_with(cast())
    assert req.personas["roster"][0]["id"] == "nurse"
    assert req.personas["enabled"] is True


def test_a_cast_is_optional():
    assert CreateSandboxRequest(env_name="ward", description="x").personas is None


def test_granting_an_action_at_creation_is_refused_with_the_reason():
    """The environment's actions do not exist yet — there is nothing to bind."""
    with pytest.raises(ValueError, match="do not exist"):
        request_with(
            cast(roster=[{"id": "n", "name": "N", "behavior": {"allowed_actions": ["/x"]}}])
        )


def test_a_malformed_cast_is_refused_at_the_request_not_mid_build():
    with pytest.raises(ValueError, match="dilgence"):
        request_with(cast(roster=[{"id": "n", "name": "N", "traits": {"dilgence": 9}}]))


def test_an_out_of_range_trait_is_refused():
    with pytest.raises(ValueError):
        request_with(cast(roster=[{"id": "n", "name": "N", "traits": {"diligence": 900}}]))


def test_an_archetype_is_expanded_so_the_build_does_not_depend_on_the_library():
    req = request_with(cast())
    assert req.personas["roster"][0]["name"] == "Priya Raman"
    assert "archetype" not in req.personas["roster"][0]


# --- the write that happens when the build finishes ------------------------


def test_the_build_writes_the_cast_into_a_new_custom_directory(tmp_path):
    """Generated container environments have no custom/ dir until this runs."""
    written = _write_personas(tmp_path, cast())
    assert written == 1
    saved = yaml.safe_load((tmp_path / "custom" / "config.yaml").read_text())
    assert saved["personas"]["roster"][0]["id"] == "nurse"
    assert saved["personas"]["enabled"] is True


def test_the_written_cast_arrives_unable_to_act(tmp_path):
    _write_personas(tmp_path, cast())
    saved = yaml.safe_load((tmp_path / "custom" / "config.yaml").read_text())
    assert saved["personas"]["roster"][0]["behavior"]["allowed_actions"] == []


def test_the_write_preserves_an_existing_config(tmp_path):
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "config.yaml").write_text("reward:\n  base_success: 3.0\n")
    _write_personas(tmp_path, cast())
    saved = yaml.safe_load((custom / "config.yaml").read_text())
    assert saved["reward"]["base_success"] == 3.0
    assert saved["personas"]["enabled"] is True


def test_a_bad_cast_does_not_fail_the_build(tmp_path):
    """Losing the cast is worth far less than losing a build that succeeded."""
    assert _write_personas(tmp_path, {"enabeld": True}) == 0


def test_the_written_cast_reads_back_through_the_runtime_loader(tmp_path):
    from forge.personas.config import load_population

    _write_personas(tmp_path, cast())
    saved = yaml.safe_load((tmp_path / "custom" / "config.yaml").read_text())
    population = load_population(saved["personas"])
    assert population.enabled
    assert population.roster[0].profile.traits.diligence == 95


# --- the action surface the personas page reads back -----------------------


def app_dir(tmp_path, source: str):
    directory = tmp_path / "app"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "main.py").write_text(source)
    return directory


def test_generated_container_endpoints_are_discovered(tmp_path):
    directory = app_dir(
        tmp_path,
        '@app.post("/send_message")\ndef send(): ...\n'
        '@app.post("/approve")\ndef approve(): ...\n',
    )
    assert _container_endpoints(directory) == ["/approve", "/send_message"]


def test_platform_routes_are_excluded():
    """These are Forge's own control plane, not the environment's actions."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        directory = app_dir(Path(tmp), (
            '@app.post("/forge/reset")\ndef reset(): ...\n'
            '@app.post("/ui")\ndef ui(): ...\n'
            '@app.post("/real_action")\ndef real(): ...\n'
        ))
        assert _container_endpoints(directory) == ["/real_action"]


def test_read_only_routes_are_not_actions(tmp_path):
    directory = app_dir(
        tmp_path,
        '@app.get("/messages")\ndef read(): ...\n@app.post("/send")\ndef send(): ...\n',
    )
    assert _container_endpoints(directory) == ["/send"]


def test_routes_on_a_router_are_found_too(tmp_path):
    directory = app_dir(tmp_path, "")
    (directory / "routes.py").write_text('@router.post("/escalate")\ndef go(): ...\n')
    assert "/escalate" in _container_endpoints(directory)


def test_unparseable_source_yields_no_actions_rather_than_raising(tmp_path):
    directory = app_dir(tmp_path, "def broken(:\n")
    assert _container_endpoints(directory) == []


def test_a_missing_app_directory_yields_no_actions(tmp_path):
    assert _container_endpoints(tmp_path / "nope") == []


def test_the_personas_page_reads_a_generated_container_surface(client, tmp_path, monkeypatch):
    envs_dir = tmp_path / "generated_envs"
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(envs_dir))
    env_dir = envs_dir / "ward"
    app_dir(env_dir, '@app.post("/send_message")\ndef send(): ...\n')
    body = client.get("/api/envs/ward/personas").json()
    assert body["environment_actions"] == ["/send_message"]


def test_the_personas_page_works_before_any_custom_directory_exists(
    client, tmp_path, monkeypatch
):
    """The builder's environments have no custom/ dir — the page must not 404."""
    envs_dir = tmp_path / "generated_envs"
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(envs_dir))
    app_dir(envs_dir / "ward", '@app.post("/send_message")\ndef send(): ...\n')
    assert client.get("/api/envs/ward/personas").status_code == 200


def test_saving_creates_the_custom_directory(client, tmp_path, monkeypatch):
    envs_dir = tmp_path / "generated_envs"
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(envs_dir))
    env_dir = envs_dir / "ward"
    app_dir(env_dir, '@app.post("/send_message")\ndef send(): ...\n')
    response = client.put(
        "/api/envs/ward/personas",
        json={
            "personas": {
                "enabled": True,
                "roster": [
                    {
                        "id": "nurse",
                        "name": "N",
                        "behavior": {"allowed_actions": ["/send_message"]},
                    }
                ],
            }
        },
    )
    assert response.status_code == 200
    assert (env_dir / "custom" / "config.yaml").exists()


def test_an_environment_that_does_not_exist_is_still_a_404(client, tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    assert client.get("/api/envs/ghost/personas").status_code == 404
