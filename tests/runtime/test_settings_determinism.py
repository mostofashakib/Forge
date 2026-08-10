from __future__ import annotations

import pytest

from forge.settings import determinism_enabled, experiment_seed


def test_determinism_defaults_on(monkeypatch):
    monkeypatch.delenv("FORGE_DETERMINISM", raising=False)
    assert determinism_enabled() is True
    assert experiment_seed(7) == 7


def test_determinism_off_drops_seed(monkeypatch):
    monkeypatch.setenv("FORGE_DETERMINISM", "off")
    assert determinism_enabled() is False
    assert experiment_seed(7) is None


def test_invalid_determinism_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("FORGE_DETERMINISM", "maybe")
    with pytest.raises(ValueError, match="must be 'on' or 'off'"):
        determinism_enabled()
