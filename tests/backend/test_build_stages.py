"""The post-build validation stages of build_sandbox_task."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from backend.app.models import SandboxEnvironment
from backend.app.worker import tasks
from forge.envgen.correctness_validator import (
    CorrectnessFinding, CorrectnessValidationError, CorrectnessValidationResult,
)
from forge.envgen.post_generation_validator import ValidationResult
from forge.schema.state_schema import StateSchemaManifest

MANIFEST = StateSchemaManifest(env_name="env", fields={"items": {"type": "array"}})
FIXED = StateSchemaManifest(env_name="env", fields={"items": {"type": "array"}, "count": {"type": "integer"}})


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/build.db")
    from backend.app import database
    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_SessionLocal", None)
    database.init_db()
    with database.get_session_factory()() as db:
        db.add(SandboxEnvironment(
            id="env", status="running", ttl_days=1,
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        ))
        db.commit()
    env_dir = tmp_path / "env"
    (env_dir / "app").mkdir(parents=True)
    (env_dir / "app" / "main.py").write_text("app = 1\n")
    return env_dir


def _sandbox():
    from backend.app import database
    with database.get_session_factory()() as db:
        return db.get(SandboxEnvironment, "env")


# ── correctness gate ─────────────────────────────────────────────────────────

async def test_correctness_gate_passes_quietly():
    logs: list[dict] = []
    with patch("forge.envgen.correctness_validator.CorrectnessValidator") as validator:
        validator.return_value.validate.return_value = CorrectnessValidationResult(passed=True)
        await tasks._check_correctness("http://x", ["a"], logs.append)
    validator.return_value.validate.assert_called_once_with(["a"])
    assert logs[-1] == {"log": "[forge] correctness validation passed ✓"}


async def test_correctness_gate_rejects_a_broken_environment():
    logs: list[dict] = []
    failed = CorrectnessValidationResult(
        passed=False, findings=[CorrectnessFinding("reset", "reset keeps state")]
    )
    with patch("forge.envgen.correctness_validator.CorrectnessValidator") as validator:
        validator.return_value.validate.return_value = failed
        with pytest.raises(CorrectnessValidationError):
            await tasks._check_correctness("http://x", ["a"], logs.append)
    assert {"log": "[forge] correctness FAIL [reset]: reset keeps state"} in logs


async def test_correctness_gate_that_cannot_run_does_not_fail_the_build():
    logs: list[dict] = []
    with patch("forge.envgen.correctness_validator.CorrectnessValidator") as validator:
        validator.return_value.validate.side_effect = ConnectionError("not up")
        await tasks._check_correctness("http://x", ["a"], logs.append)
    assert logs[-1] == {"log": "[forge] correctness validation could not run: not up"}


# ── manifest validation ─────────────────────────────────────────────────────

async def _validate(env_dir, logs):
    await tasks._validate_manifest(
        env_name="env", description="d", compiler_input=MagicMock(),
        base_url="http://x", env_dir=env_dir, publish=logs.append,
    )


async def test_no_manifest_means_nothing_to_validate(db_env):
    with patch("forge.envgen.post_generation_validator.PostGenerationValidator") as validator:
        await _validate(db_env, [])
    validator.assert_not_called()


async def test_a_passing_manifest_is_stored_on_the_sandbox(db_env):
    (db_env / "state_schema.json").write_text(MANIFEST.model_dump_json())
    with patch("forge.envgen.post_generation_validator.PostGenerationValidator") as validator:
        validator.return_value.validate.return_value = ValidationResult(passed=True, coverage_score=1.0)
        await _validate(db_env, [])
    assert StateSchemaManifest.model_validate_json(_sandbox().state_schema) == MANIFEST
    assert _sandbox().validation_missing_fields is None


async def test_failing_manifest_is_retried_through_the_state_bridge_then_recorded(db_env):
    (db_env / "state_schema.json").write_text(MANIFEST.model_dump_json())
    bridge_runs: list[list[str]] = []

    class _Bridge:
        def __init__(self, missing_fields_feedback):
            self._feedback = missing_fields_feedback

        async def run(self, _ctx, bus):
            bridge_runs.append(self._feedback)
            assert bus.get("instrumented_code") == {"main.py": "app = 1\n"}
            await bus.publish("state_schema_manifest", FIXED)
            await bus.publish("state_bridge_code", "# bridge\n")

    with patch("forge.envgen.post_generation_validator.PostGenerationValidator") as validator, \
         patch("forge.envgen.agents.state_bridge.StateBridgeAgent", _Bridge):
        validator.return_value.validate.return_value = ValidationResult(
            passed=False, missing_fields=["count"]
        )
        await _validate(db_env, [])

    assert bridge_runs == [["count"], ["count"]]
    assert StateSchemaManifest.model_validate_json((db_env / "state_schema.json").read_text()) == FIXED
    assert (db_env / "container_env.py").read_text() == "# bridge\n"
    assert json.loads(_sandbox().validation_missing_fields) == ["count"]
    assert _sandbox().state_schema is None


async def test_unreachable_container_stops_manifest_validation(db_env):
    (db_env / "state_schema.json").write_text(MANIFEST.model_dump_json())
    logs: list[dict] = []
    with patch("forge.envgen.post_generation_validator.PostGenerationValidator") as validator:
        validator.return_value.validate.side_effect = ConnectionError("refused")
        await _validate(db_env, logs)
    assert validator.return_value.validate.call_count == 1
    assert logs[-1] == {"log": "[forge] manifest validation error (container not ready?): refused"}
    assert _sandbox().state_schema is None
