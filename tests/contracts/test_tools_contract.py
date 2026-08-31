from __future__ import annotations

from collections.abc import Sequence

import pytest

from forge.contracts import ToolProvider, ToolSpec


class _Static(ToolProvider):
    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = tools

    def tools(self) -> Sequence[ToolSpec]:
        return list(self._tools)


def test_a_provider_lists_its_tools():
    provider = _Static([ToolSpec(name="close_ticket")])
    assert [t.name for t in provider.tools()] == ["close_ticket"]


def test_a_provider_with_no_tools_stays_empty():
    # False-positive guard: a shell environment genuinely has no tool schema.
    assert list(_Static([]).tools()) == []


def test_a_provider_missing_tools_cannot_be_instantiated():
    class Incomplete(ToolProvider):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
