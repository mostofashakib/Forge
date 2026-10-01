"""Find an environment for the task factory and open its runner and profile.

An environment is either a sandbox (custom app, premade app, CLI or browser
container) or a compiled in-process package under generated_envs.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import docker
import httpx
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.app.models import SandboxEnvironment
from forge.contracts.transport import DEFAULT_TIMEOUT_S
from forge.envgen.container import ContainerRuntime
from forge.settings import generated_envs_root
from forge.taskfactory.profile import BROWSER_NOTES, CLI_NOTES, EnvironmentProfile, ToolInfo
from forge.taskfactory.runner import TaskRunner
from forge.taskfactory.runners.browser import BrowserRunner
from forge.taskfactory.runners.cli import CliRunner
from forge.taskfactory.runners.container_app import ContainerAppRunner
from forge.taskfactory.runners.in_process import InProcessRunner

IN_PROCESS = "in_process"
_FAMILY = {"cli": "cli", "browser": "browser"}
_DEVTOOLS_PORT = 9222


class TargetUnavailable(RuntimeError):
    """The environment cannot be used right now, with the reason in the message."""


@dataclass(frozen=True)
class TargetInfo:
    name: str
    env_type: str
    family: str
    ready: bool
    reason: str | None = None


@dataclass
class Target:
    runner: TaskRunner
    # Reading a profile can reset an app, so the caller holds the env lock.
    profile: Callable[[], EnvironmentProfile]


def _active_sandboxes(db: Session) -> list[SandboxEnvironment]:
    return (
        db.query(SandboxEnvironment)
        .filter(SandboxEnvironment.status.notin_(["deleted", "expired"]))
        .filter(or_(SandboxEnvironment.expires_at.is_(None), SandboxEnvironment.expires_at > datetime.now(timezone.utc)))
        .order_by(SandboxEnvironment.id)
        .all()
    )


def _in_process_names(db: Session) -> list[str]:
    root = generated_envs_root()
    if not root.exists():
        return []
    sandbox_names = {row.id for row in db.query(SandboxEnvironment.id).all()}
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "gym_wrapper.py").exists() and p.name not in sandbox_names
    )


def list_targets(db: Session) -> list[TargetInfo]:
    targets = [TargetInfo(name, IN_PROCESS, "state", True) for name in _in_process_names(db)]
    for sandbox in _active_sandboxes(db):
        ready = sandbox.status == "running"
        targets.append(TargetInfo(
            name=sandbox.id,
            env_type=sandbox.env_type,
            family=_FAMILY.get(sandbox.env_type, "state"),
            ready=ready,
            reason=None if ready else f"The environment is {sandbox.status}. Start it, then create tasks.",
        ))
    return sorted(targets, key=lambda t: t.name)


@contextmanager
def open_target(db: Session, env_name: str, batch_id: str) -> Iterator[Target]:
    sandbox = db.get(SandboxEnvironment, env_name)
    if sandbox is None:
        if env_name not in _in_process_names(db):
            raise TargetUnavailable(f"no environment named {env_name!r}")
        yield _in_process_target(env_name)
        return
    if sandbox.status != "running" or not sandbox.container_id:
        raise TargetUnavailable(f"{env_name} is {sandbox.status}. Start it, then create tasks.")
    client = docker.from_env()
    try:
        try:
            container = client.containers.get(sandbox.container_id)
        except docker.errors.NotFound:
            raise TargetUnavailable(f"{env_name}'s container is gone. Start it, then create tasks.") from None
        runtime = ContainerRuntime(client)
        if sandbox.env_type == "cli":
            yield Target(
                CliRunner(runtime, env_name, container.id, batch_id=batch_id),
                lambda: EnvironmentProfile(env_name=env_name, env_type="cli", family="cli", notes=CLI_NOTES),
            )
        elif sandbox.env_type == "browser":
            yield _browser_target(env_name, _gateway_port(runtime, container, env_name, _DEVTOOLS_PORT))
        else:
            port = _gateway_port(runtime, container, env_name, None)
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=DEFAULT_TIMEOUT_S) as http:
                runner = ContainerAppRunner(http)
                yield Target(runner, lambda: runner.profile(env_name, sandbox.env_type))
    finally:
        client.close()


def _gateway_port(runtime: ContainerRuntime, container, env_name: str, container_port: int | None) -> int:
    port = runtime.host_port(container, container_port) if container_port else runtime.host_port(container)
    if port is None:
        raise TargetUnavailable(f"{env_name} has no gateway. Press Start to recreate it, then create tasks.")
    return port


def _in_process_target(env_name: str) -> Target:
    from backend.app.database import get_session_factory
    from backend.app.utils.env_loader import forge_env_builder
    from forge.benchmark.compiled_tasks import db_compiler_input_loader

    build = forge_env_builder(env_name)

    def profile() -> EnvironmentProfile:
        env = build(max_steps=1)
        env.reset(seed=0)
        state = env.state.get()
        compiler_input = db_compiler_input_loader(get_session_factory())(env_name)
        if compiler_input is not None:
            tools = tuple(ToolInfo(name=a.name, params=tuple(p.name for p in a.params)) for a in compiler_input.actions)
        else:
            tools = tuple(ToolInfo(name=name) for name in sorted(env.backend.action_types))
        return EnvironmentProfile(
            env_name=env_name, env_type=IN_PROCESS, family="state",
            tools=tools, state_sample=state,
        )

    return Target(InProcessRunner(lambda max_steps: build(max_steps=max_steps)), profile)


def _browser_target(env_name: str, cdp_port: int) -> Target:
    @contextmanager
    def connect():
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{cdp_port}")
            try:
                yield browser
            finally:
                browser.close()

    return Target(
        BrowserRunner(connect),
        lambda: EnvironmentProfile(env_name=env_name, env_type="browser", family="browser", notes=BROWSER_NOTES),
    )
