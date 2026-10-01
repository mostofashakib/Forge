"""Finding an environment for the task factory and opening its runner and profile."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import SandboxEnvironment
from backend.app.services.task_factory_targets import TargetUnavailable, list_targets, open_target
from forge.envgen.container import ContainerRuntime
from forge.taskfactory.runners.cli import CliRunner
from forge.taskfactory.runners.container_app import ContainerAppRunner
from forge.taskfactory.runners.in_process import InProcessRunner
from tests.backend.test_env_loader_determinism import _DETERMINISTIC_WRAPPER, _write_env
from tests.envgen.fake_docker import FakeDocker


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def envs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    monkeypatch.delenv("FORGE_DEV_NETWORK", raising=False)
    for mod in [m for m in sys.modules if m.startswith("generated_envs")]:
        del sys.modules[mod]
    (tmp_path / "generated_envs").mkdir()
    return tmp_path


def _sandbox(db, name, env_type, status="running", container_id=None, expired=False):
    db.add(SandboxEnvironment(
        id=name, status=status, env_type=env_type, container_id=container_id,
        expires_at=datetime.now(timezone.utc) + timedelta(days=-1 if expired else 30),
    ))
    db.commit()


def test_targets_cover_sandboxes_and_compiled_packages(db, envs_dir):
    _write_env(envs_dir, "ledger", _DETERMINISTIC_WRAPPER)
    _sandbox(db, "shell", "cli", container_id="c1")
    _sandbox(db, "web", "browser", status="stopped")
    _sandbox(db, "mail", "premade:gmail", container_id="c2")

    targets = {t.name: t for t in list_targets(db)}

    assert targets["ledger"].family == "state" and targets["ledger"].env_type == "in_process"
    assert targets["shell"].family == "cli" and targets["shell"].ready
    assert targets["mail"].family == "state"
    assert not targets["web"].ready
    assert "Start" in targets["web"].reason


def test_expired_or_deleted_sandboxes_are_not_targets(db, envs_dir):
    _sandbox(db, "old", "cli", expired=True)
    _sandbox(db, "gone", "cli", status="deleted")

    assert [t.name for t in list_targets(db)] == []


def test_an_unknown_environment_is_unavailable(db, envs_dir):
    with pytest.raises(TargetUnavailable, match="nope"):
        with open_target(db, "nope", "tb_1"):
            pass


def test_a_stopped_environment_is_unavailable(db, envs_dir):
    _sandbox(db, "shell", "cli", status="stopped", container_id="c1")

    with pytest.raises(TargetUnavailable, match="stopped"):
        with open_target(db, "shell", "tb_1"):
            pass


def test_an_in_process_target_samples_its_reset_state(db, envs_dir):
    _write_env(envs_dir, "ledger", _DETERMINISTIC_WRAPPER)

    with open_target(db, "ledger", "tb_1") as target:
        profile = target.profile()
        runner = target.runner

    assert isinstance(runner, InProcessRunner)
    assert profile.tool_names == ["increment"]
    assert profile.tools[0].params == ()
    assert "counter" in profile.state_sample


@pytest.fixture
def daemon():
    fake = FakeDocker()
    with patch("forge.envgen.container.docker.from_env", return_value=fake), \
         patch("backend.app.services.task_factory_targets.docker.from_env", return_value=fake), \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        yield fake


def test_a_cli_target_forks_from_its_container(db, envs_dir, daemon):
    with patch("forge.envgen.container.ensure_cli_image", return_value="forge-cli:test"):
        cid, _ = ContainerRuntime().run_cli("shell")
    _sandbox(db, "shell", "cli", container_id=cid)

    with open_target(db, "shell", "tb_1") as target:
        assert isinstance(target.runner, CliRunner)
        assert target.profile().family == "cli"


def test_an_app_target_talks_to_the_gateway_port(db, envs_dir, daemon):
    app_id, port = ContainerRuntime().run("mail", "forge-env-mail:latest")
    _sandbox(db, "mail", "general", container_id=app_id)

    with open_target(db, "mail", "tb_1") as target:
        assert isinstance(target.runner, ContainerAppRunner)
        assert target.runner._client.base_url.port == port


def test_an_app_without_its_gateway_is_unavailable(db, envs_dir, daemon):
    app_id, _ = ContainerRuntime().run("mail", "forge-env-mail:latest")
    daemon.named("forge-mail-gw").remove(force=True)
    _sandbox(db, "mail", "general", container_id=app_id)

    with pytest.raises(TargetUnavailable, match="Start"):
        with open_target(db, "mail", "tb_1"):
            pass


def test_in_process_tools_show_parameter_types_and_allowed_values(db, envs_dir, monkeypatch):
    from forge.extraction.schemas import ActionDef, ActionParam, CompilerInput, EntityDef, FieldDef

    _write_env(envs_dir, "ledger", _DETERMINISTIC_WRAPPER)
    compiled = CompilerInput(
        project_name="ledger", domain="test",
        entities=[EntityDef(name="counter", fields=[FieldDef(name="id", type="string")])],
        actions=[ActionDef(name="increment", params=[
            ActionParam(name="counter_id", type="string"),
            ActionParam(name="mode", type="enum", values=["fast", "slow"]),
        ])],
        policies=[], tasks=[],
    )
    monkeypatch.setattr(
        "forge.benchmark.compiled_tasks.db_compiler_input_loader", lambda factory: lambda name: compiled,
    )

    with open_target(db, "ledger", "tb_1") as target:
        tool = target.profile().tools[0]

    assert tool.params == ("counter_id: string", "mode: fast|slow")
