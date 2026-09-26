"""Worker boot pre-warms everything an environment launch needs."""
from __future__ import annotations

from unittest.mock import patch

from backend.app.worker import celery_app


class _InlineThread:
    """Runs the pre-warm target synchronously so the test can observe it."""

    def __init__(self, target, **_kwargs):
        self._target = target

    def start(self):
        self._target()


def test_boot_prewarm_also_builds_the_cli_image(monkeypatch):
    monkeypatch.delenv("FORGE_DISABLE_PREWARM", raising=False)
    with patch.object(celery_app.threading, "Thread", _InlineThread), \
         patch("forge.envgen.container.prewarm_standard_base_images", return_value={}), \
         patch("forge.envgen.container.ensure_cli_image") as ensure:
        celery_app._prewarm_base_images_on_boot()

    ensure.assert_called_once_with()


def test_disabled_prewarm_does_not_build_the_cli_image(monkeypatch):
    monkeypatch.setenv("FORGE_DISABLE_PREWARM", "1")
    with patch("forge.envgen.container.ensure_cli_image") as ensure:
        celery_app._prewarm_base_images_on_boot()

    ensure.assert_not_called()
