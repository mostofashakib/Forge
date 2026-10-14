"""Only forge/settings.py builds a URL to a sandbox port, so the sandbox host is one setting."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "forge" / "settings.py"
# A URL with a scheme, a loopback host, and a port taken from a variable.
HAND_BUILT = re.compile(r"""https?://(localhost|127\.0\.0\.1):\{""")


def _sources() -> list[Path]:
    return [
        path
        for package in ("forge", "backend")
        for path in (ROOT / package).rglob("*.py")
        if path != SETTINGS and "__pycache__" not in path.parts
    ]


def test_no_module_outside_settings_builds_a_loopback_url_from_a_port():
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in _sources()
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if HAND_BUILT.search(line)
    ]

    assert offenders == []


def test_the_pattern_catches_both_loopback_spellings_and_ignores_a_fixed_port():
    assert HAND_BUILT.search('f"http://localhost:{port}"')
    assert HAND_BUILT.search('f"http://127.0.0.1:{cdp_port}"')
    assert not HAND_BUILT.search('"http://localhost:11434"')
    assert not HAND_BUILT.search('f"TCP:127.0.0.1:{BROWSER_CDP_PORT}"')
