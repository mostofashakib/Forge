"""A compiled in-process environment cannot hand the agent the internet.

In-process packages run inside the worker, and the agent's tools are their
transitions. A package that imports a network module would let an action
reach the internet, so the loader refuses to import it, even when
determinism checks are off.
"""
import sys

import pytest

from backend.app.utils.env_loader import load_forge_env
from tests.backend.test_env_loader_determinism import _DETERMINISTIC_WRAPPER, _write_env


@pytest.fixture
def envs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    monkeypatch.delenv("FORGE_DETERMINISM", raising=False)
    monkeypatch.delenv("FORGE_DEV_NETWORK", raising=False)
    for mod in [m for m in sys.modules if m.startswith("generated_envs")]:
        del sys.modules[mod]
    return tmp_path


@pytest.mark.parametrize("network_line", ["import httpx", "import urllib.request", "import socket"])
def test_a_package_with_network_access_is_refused_before_import(envs_dir, network_line, monkeypatch):
    _write_env(envs_dir, "leaky_env", f"{network_line}\n" + _DETERMINISTIC_WRAPPER)
    monkeypatch.setenv("FORGE_DETERMINISM", "off")  # egress is never an ablation

    with pytest.raises(RuntimeError, match="network"):
        load_forge_env("leaky_env", telemetry=None)

    assert "generated_envs.leaky_env.gym_wrapper" not in sys.modules


def test_a_package_without_network_access_still_loads(envs_dir):
    # False-positive guard.
    _write_env(envs_dir, "clean_env", _DETERMINISTIC_WRAPPER)

    assert load_forge_env("clean_env", telemetry=None) is not None
