"""sandbox_url: every URL to a sandbox port, built from one host setting."""
from __future__ import annotations

import pytest

from forge.settings import sandbox_url


def test_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("FORGE_SANDBOX_HOST", raising=False)

    assert sandbox_url(8123) == "http://127.0.0.1:8123"


def test_uses_the_configured_host(monkeypatch):
    monkeypatch.setenv("FORGE_SANDBOX_HOST", "10.0.0.5")

    assert sandbox_url(8123) == "http://10.0.0.5:8123"


def test_keeps_the_requested_scheme(monkeypatch):
    monkeypatch.setenv("FORGE_SANDBOX_HOST", "10.0.0.5")

    assert sandbox_url(9222, scheme="ws") == "ws://10.0.0.5:9222"


def test_blank_host_falls_back_to_loopback(monkeypatch):
    monkeypatch.setenv("FORGE_SANDBOX_HOST", "  ")

    assert sandbox_url(8123) == "http://127.0.0.1:8123"


@pytest.mark.parametrize("port", [0, -1, 65536, None])
def test_rejects_an_invalid_port(port):
    with pytest.raises(ValueError, match="port"):
        sandbox_url(port)
