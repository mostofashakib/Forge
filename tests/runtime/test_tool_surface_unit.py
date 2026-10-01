"""ToolSurface lists the core tools plus only the modalities that are attached."""
from __future__ import annotations

from types import SimpleNamespace

from forge.runtime.control import SUBMIT_ACTION
from forge.runtime.snapshot import ToolSpec
from forge.runtime.tool_surface import ToolSurface


def _modality(name: str, tools: list[str]):
    return SimpleNamespace(name=name, schema=SimpleNamespace(tool_specs=lambda: [ToolSpec(name=t) for t in tools]))


def test_documented_tools_keep_their_spec_and_submit_is_described():
    documented = ToolSpec(name="send", description="Send a message")
    surface = ToolSurface({"send", "archive", SUBMIT_ACTION}, {"send": documented}, [])

    specs = {spec.name: spec for spec in surface.tool_specs()}
    assert specs["send"] is documented
    assert specs["archive"] == ToolSpec(name="archive")
    assert "grade" in specs[SUBMIT_ACTION].description


def test_absent_modalities_are_not_advertised():
    surface = ToolSurface({"send"}, {}, [None, _modality("rest_use", ["GET /x"]), None])

    assert surface.capabilities() == ["tool_use", "rest_use"]
    assert [s.name for s in surface.by_modality()["rest_use"]] == ["GET /x"]
    assert "mcp_use" not in surface.by_modality()
