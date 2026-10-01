"""Premade images build from a staged copy of the template plus the shared protocol."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from backend.app.worker import sandbox_build_tasks
from backend.app.worker.sandbox_build_tasks import BuildRequest, build_premade

PREMADE_ROOT = Path(__file__).resolve().parents[2] / "docker" / "premade"


class _Reporter:
    def __init__(self) -> None:
        self.ran: tuple | None = None

    def log(self, _message: str) -> None:
        pass

    def building(self) -> None:
        pass

    def running(self, container_id, port, image_tag) -> None:
        self.ran = (container_id, port, image_tag)


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def _capture_build(monkeypatch, *, fail: bool = False) -> dict:
    seen: dict = {}

    async def fake_build_and_run(env_name, context_dir):
        seen["context"] = context_dir
        seen["files"] = sorted(path.name for path in context_dir.iterdir())
        # The real build rewrites requirements.txt inside its context.
        (context_dir / "requirements.txt").write_text("rewritten\n")
        if fail:
            raise RuntimeError("docker build failed")
        return "image:tag", "cid", 8123

    monkeypatch.setattr(sandbox_build_tasks, "_build_and_run", fake_build_and_run)
    return seen


@pytest.mark.parametrize("template", ["gmail", "slack"])
def test_the_build_context_holds_the_template_and_the_shared_protocol(monkeypatch, template):
    seen = _capture_build(monkeypatch)
    reporter = _Reporter()

    asyncio.run(build_premade(BuildRequest(env_name="env", env_type=f"premade:{template}"), reporter))

    assert "forge_protocol.py" in seen["files"]
    assert {"app.py", "Dockerfile", f"{template}_seed.py", "ui.html"} <= set(seen["files"])
    assert reporter.ran == ("cid", 8123, "image:tag")


def test_building_never_writes_into_the_tracked_template(monkeypatch):
    before = _tree_digest(PREMADE_ROOT / "gmail")
    _capture_build(monkeypatch)

    asyncio.run(build_premade(BuildRequest(env_name="env", env_type="premade:gmail"), _Reporter()))

    assert _tree_digest(PREMADE_ROOT / "gmail") == before


def test_the_staged_context_is_removed_even_when_the_build_fails(monkeypatch):
    seen = _capture_build(monkeypatch, fail=True)

    with pytest.raises(RuntimeError, match="docker build failed"):
        asyncio.run(build_premade(BuildRequest(env_name="env", env_type="premade:gmail"), _Reporter()))

    assert not seen["context"].exists()


def test_an_unknown_template_fails_before_staging(monkeypatch):
    seen = _capture_build(monkeypatch)

    with pytest.raises(FileNotFoundError, match="Premade template 'nope' not found"):
        asyncio.run(build_premade(BuildRequest(env_name="env", env_type="premade:nope"), _Reporter()))

    assert seen == {}


@pytest.mark.parametrize("template", ["_shared", "../gmail", "gmail/.."])
def test_only_a_bundled_app_folder_can_be_built(monkeypatch, template):
    seen = _capture_build(monkeypatch)

    with pytest.raises(FileNotFoundError):
        asyncio.run(build_premade(BuildRequest(env_name="env", env_type=f"premade:{template}"), _Reporter()))

    assert seen == {}
