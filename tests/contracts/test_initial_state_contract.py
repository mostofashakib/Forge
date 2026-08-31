from __future__ import annotations

from collections.abc import Mapping

import pytest

from forge.contracts import InitialStateProvider


class _Seeded(InitialStateProvider):
    def reset(self, ctx, *, seed: int | None, options: Mapping[str, object]) -> dict:
        return {"seed": seed, "options": dict(options)}


def test_the_seed_is_an_explicit_keyword_not_smuggled_in_options():
    state = _Seeded().reset(None, seed=7, options={})
    assert state["seed"] == 7
    assert state["options"] == {}


def test_an_unseeded_reset_is_representable():
    # False-positive guard: seed=None is a valid, distinct request for the
    # provider's fixed baseline — it must not be confused with seed=0.
    assert _Seeded().reset(None, seed=None, options={})["seed"] is None


def test_seed_must_be_passed_by_keyword():
    # Negative: positional seed is rejected, so call sites cannot drift.
    with pytest.raises(TypeError):
        _Seeded().reset(None, 7, {})


def test_a_provider_missing_reset_cannot_be_instantiated():
    class Incomplete(InitialStateProvider):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
