from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from forge.schema.state_schema import StateSchemaManifest

logger = logging.getLogger(__name__)


@dataclass
class CorrectnessFinding:
    category: str
    message: str


@dataclass
class CorrectnessValidationResult:
    passed: bool
    findings: list[CorrectnessFinding] = field(default_factory=list)


class CorrectnessValidationError(RuntimeError):
    def __init__(self, result: CorrectnessValidationResult) -> None:
        messages = "; ".join(f.message for f in result.findings)
        super().__init__("Environment failed correctness validation: " + messages)
        self.result = result


class CorrectnessValidator:
    """Proves reset fidelity and snapshot/restore round-trips on a live container."""

    def __init__(self, base_url: str, http_timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = http_timeout

    def validate(
        self,
        action_names: list[str],
        manifest: StateSchemaManifest | None = None,
    ) -> CorrectnessValidationResult:
        findings: list[CorrectnessFinding] = []
        with httpx.Client(base_url=self._base_url, timeout=self._timeout) as client:

            def state() -> dict:
                return client.get("/forge/state").raise_for_status().json()

            def reset() -> None:
                client.post("/forge/reset").raise_for_status()

            def mutate() -> None:
                for name in action_names:
                    try:
                        client.post(f"/{name}")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[correctness] could not exercise /%s: %s", name, exc)

            # ── Reset fidelity ────────────────────────────────────────────
            reset()
            baseline = state()
            reset()
            if state() != baseline:
                findings.append(CorrectnessFinding(
                    "reset_fidelity", "Two consecutive resets produced different state",
                ))
            mutate()
            reset()
            if state() != baseline:
                findings.append(CorrectnessFinding(
                    "reset_fidelity",
                    "Reset did not restore the exact initial universe (state drift after actions)",
                ))

            # ── Snapshot / restore round-trip ─────────────────────────────
            reset()
            snapshot = state()
            client.post("/forge/snapshot", json={"slot": "correctness"}).raise_for_status()
            mutate()
            client.post("/forge/restore/correctness").raise_for_status()
            restored = state()
            if restored != snapshot:
                findings.append(CorrectnessFinding(
                    "snapshot_restore", "Restore did not reproduce the snapshotted state",
                ))
            elif manifest is not None:
                for fname in manifest.fields:
                    if fname not in restored:
                        findings.append(CorrectnessFinding(
                            "snapshot_restore",
                            f"Restored state is missing declared field {fname!r}",
                        ))

        return CorrectnessValidationResult(passed=not findings, findings=findings)
