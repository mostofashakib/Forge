"""Run tasks in a custom or premade app through its Forge control plane.

Seed rows merge into, and checks read, the app's full restorable state:
`/forge/dump` when the app has one (premade apps, whose `/forge/state` is a
view that hides archived or sent items), else `/forge/state` (generated apps,
where the two are the same). A batch keeps
the app's full state and restores it afterwards, so creating tasks leaves
the environment as it found it.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import httpx

from forge.runtime.tools import OpenAPIToolProvider
from forge.taskfactory.profile import EnvironmentProfile, ToolInfo, param_hint
from forge.taskfactory.runner import SeedError, StepResult, fingerprint_of
from forge.taskfactory.schemas import Check, GoldenStep, TaskDraft
from forge.taskfactory.state_checks import CheckOutcome, apply_seed, evaluate_state_check
from forge.taskfactory.static_check import is_control_plane

class AppControlError(RuntimeError):
    """The app's control plane did not answer as the Forge contract says."""


class ContainerAppRunner:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def profile(self, env_name: str, env_type: str) -> EnvironmentProfile:
        """The app's actions and its full reset state."""
        manifest = OpenAPIToolProvider(self._client).action_manifest()
        if not manifest:
            raise AppControlError(f"{env_name} exposes no actions in /openapi.json")
        tools = tuple(
            ToolInfo(
                name=action["endpoint"],
                description=action.get("description", ""),
                params=tuple(
                    param_hint(name, spec.get("type"), spec.get("enum"))
                    for name, spec in (action.get("request_schema") or {}).get("properties", {}).items()
                ),
            )
            for action in manifest
        )
        with self.batch():
            self._control("POST", "/forge/reset", json={})
            state_sample = self._restorable_state()
        return EnvironmentProfile(
            env_name=env_name, env_type=env_type, family="state",
            tools=tools, state_sample=state_sample,
        )

    @contextmanager
    def batch(self) -> Iterator[None]:
        # Held in memory, not in a snapshot slot: an app's reset may clear its slots.
        saved = self._restorable_state()
        try:
            yield
        finally:
            self._control("POST", "/forge/restore-state", json=saved)

    @contextmanager
    def session(self, draft: TaskDraft) -> Iterator["_AppSession"]:
        self._control("POST", "/forge/reset", json={})
        if draft.seed.records:
            try:
                seeded = apply_seed(self._restorable_state(), draft.seed.records)
            except ValueError as exc:
                raise SeedError(str(exc)) from exc
            response = self._client.post("/forge/restore-state", json=seeded)
            if not response.is_success:
                raise SeedError(f"the app refused the seeded state: HTTP {response.status_code} {response.text[:300]}")
        yield _AppSession(self)

    def _restorable_state(self) -> dict:
        response = self._client.get("/forge/dump")
        if response.status_code == 404:
            return self._state()
        if not response.is_success:
            raise AppControlError(f"GET /forge/dump answered HTTP {response.status_code}")
        return response.json()

    def _state(self) -> dict:
        return self._control("GET", "/forge/state").json()

    def _control(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        response = self._client.request(method, path, **kwargs)
        if not response.is_success:
            raise AppControlError(f"{method} {path} answered HTTP {response.status_code}: {response.text[:300]}")
        return response


class _AppSession:
    def __init__(self, runner: ContainerAppRunner) -> None:
        self._runner = runner
        self._called: list[str] = []

    def step(self, step: GoldenStep) -> StepResult:
        if is_control_plane(step.tool):
            return StepResult(ok=False, error=f"{step.tool!r} is the control plane, not an action")
        try:
            response = self._runner._client.post(step.tool, json=step.args)
        except httpx.HTTPError as exc:
            return StepResult(ok=False, error=f"request failed: {exc}")
        body = _json_or_none(response)
        if not response.is_success:
            return StepResult(ok=False, error=f"HTTP {response.status_code}: {response.text[:300]}")
        if isinstance(body, dict) and body.get("ok") is False:
            return StepResult(ok=False, error=str(body.get("error") or body))
        self._called.append(step.tool)
        return StepResult(ok=True, output=response.text[:500])

    def evaluate(self, checks: Sequence[Check]) -> list[CheckOutcome]:
        state = self._runner._restorable_state()
        return [evaluate_state_check(check, state, self._called) for check in checks]

    def fingerprint(self) -> str:
        return fingerprint_of(self._runner._restorable_state())


def _json_or_none(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None
