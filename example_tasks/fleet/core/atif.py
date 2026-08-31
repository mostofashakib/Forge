"""ATIF trajectory serialization helpers.

The Harbor task runtime provides ``harbor.models.trajectories``. Local unit tests
in this repository do not depend on the full Harbor package, so this module uses
the Harbor Pydantic models when they are installed and falls back to the same
JSON shape otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from fleet.core.serialization import canonical_json

SCHEMA_VERSION = "ATIF-v1.7"


def atif_trajectory_to_dict(
    *,
    session_id: str,
    agent_name: str,
    agent_version: str | None,
    model_name: str | None,
    steps: list[dict[str, Any]],
    final_metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    trajectory_id: str | None = None,
) -> dict[str, Any]:
    """Build an ATIF-v1.7 trajectory dict from Harbor core models when available."""

    try:
        return _harbor_trajectory_to_dict(
            session_id=session_id,
            agent_name=agent_name,
            agent_version=agent_version,
            model_name=model_name,
            steps=steps,
            final_metrics=final_metrics,
            extra=extra,
            trajectory_id=trajectory_id,
        )
    except ModuleNotFoundError:
        return _fallback_trajectory_to_dict(
            session_id=session_id,
            agent_name=agent_name,
            agent_version=agent_version,
            model_name=model_name,
            steps=steps,
            final_metrics=final_metrics,
            extra=extra,
            trajectory_id=trajectory_id,
        )


def _harbor_trajectory_to_dict(
    *,
    session_id: str,
    agent_name: str,
    agent_version: str | None,
    model_name: str | None,
    steps: list[dict[str, Any]],
    final_metrics: dict[str, Any] | None,
    extra: dict[str, Any] | None,
    trajectory_id: str | None,
) -> dict[str, Any]:
    from harbor.models.trajectories import (  # type: ignore[import-not-found]
        Agent,
        FinalMetrics,
        Metrics,
        Observation,
        ObservationResult,
        Step,
        ToolCall,
        Trajectory,
    )

    agent_extra = dict((extra or {}).get("agent", {}))
    if trajectory_id:
        agent_extra.setdefault("trajectory_id", trajectory_id)
    agent = _model(
        Agent,
        {
            "name": agent_name,
            "version": agent_version or "",
            "model_name": model_name,
            "extra": agent_extra or None,
        },
    )
    trajectory = _model(
        Trajectory,
        {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "agent": agent,
            "steps": [_harbor_step_to_model(Step, ToolCall, Observation, ObservationResult, Metrics, step) for step in steps],
            "final_metrics": _model(FinalMetrics, final_metrics) if final_metrics else None,
            "extra": extra,
        },
    )
    payload = trajectory.to_json_dict()
    if trajectory_id:
        payload["trajectory_id"] = trajectory_id
    if final_metrics and "final_metrics" not in payload:
        payload["final_metrics"] = _exclude_none(final_metrics)
    if extra and "extra" not in payload:
        payload["extra"] = _exclude_none(extra)
    return payload


def _harbor_step_to_model(
    step_cls: Any,
    tool_call_cls: Any,
    observation_cls: Any,
    observation_result_cls: Any,
    metrics_cls: Any,
    step: dict[str, Any],
) -> Any:
    data = dict(step)
    if "tool_calls" in data:
        data["tool_calls"] = [_model(tool_call_cls, call) for call in data["tool_calls"]]
    if data.get("observation"):
        observation = dict(data["observation"])
        observation["results"] = [
            _model(observation_result_cls, result) for result in observation.get("results", [])
        ]
        data["observation"] = _model(observation_cls, observation)
    if data.get("metrics"):
        data["metrics"] = _model(metrics_cls, data["metrics"])
    return _model(step_cls, data)


def _model(model_cls: Any, data: dict[str, Any]) -> Any:
    fields = getattr(model_cls, "model_fields", None)
    if fields is None:
        fields = getattr(model_cls, "__fields__", None)
    if fields:
        supported = {key: value for key, value in data.items() if key in fields}
        unsupported = {key: value for key, value in data.items() if key not in fields and value is not None}
        if unsupported and "extra" in fields:
            existing_extra = supported.get("extra") or {}
            supported["extra"] = {**existing_extra, **unsupported}
        return model_cls(**supported)
    return model_cls(**data)


def _fallback_trajectory_to_dict(
    *,
    session_id: str,
    agent_name: str,
    agent_version: str | None,
    model_name: str | None,
    steps: list[dict[str, Any]],
    final_metrics: dict[str, Any] | None,
    extra: dict[str, Any] | None,
    trajectory_id: str | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "agent": {
            "name": agent_name,
            "version": agent_version,
            "model_name": model_name,
        },
        "steps": steps,
    }
    if trajectory_id:
        payload["trajectory_id"] = trajectory_id
    if final_metrics:
        payload["final_metrics"] = final_metrics
    if extra:
        payload["extra"] = extra
    return _exclude_none(payload)


def _exclude_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _exclude_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_exclude_none(item) for item in value]
    return value


def write_trajectory(path: str | Path, trajectory: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(trajectory), encoding="utf-8")
