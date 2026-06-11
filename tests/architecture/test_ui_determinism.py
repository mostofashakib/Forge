# tests/architecture/test_ui_determinism.py
"""UI environments must be motion-free and verify against the DB, not the UI."""
from __future__ import annotations
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREMADE = ["docker/premade/gmail", "docker/premade/slack"]


# ---------------------------------------------------------------------------
# Animations disabled
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env_dir", PREMADE)
def test_premade_ui_ships_animation_kill_switch(env_dir):
    html = (ROOT / env_dir / "ui.html").read_text()
    assert 'id="forge-no-motion"' in html, f"{env_dir}/ui.html missing the no-motion override"
    kill_switch = html[html.index('id="forge-no-motion"'):]
    assert "animation: none !important" in kill_switch
    assert "transition: none !important" in kill_switch


def test_browser_runner_disables_motion_on_pages():
    from forge.envgen.browser_runner import BrowserEpisodeRunner

    class FakeContext:
        def __init__(self):
            self.init_scripts = []

        def add_init_script(self, script):
            self.init_scripts.append(script)

    class FakePage:
        def __init__(self):
            self.media = None
            self.style_tags = []

        def emulate_media(self, reduced_motion=None):
            self.media = reduced_motion

        def add_style_tag(self, content=""):
            self.style_tags.append(content)

    ctx, page = FakeContext(), FakePage()
    BrowserEpisodeRunner._disable_motion(ctx, page)

    assert page.media == "reduce"
    assert any("animation: none !important" in tag for tag in page.style_tags)
    assert any("animation: none !important" in s for s in ctx.init_scripts)


def test_browser_runner_episode_wires_in_motion_suppression():
    from forge.envgen.browser_runner import BrowserEpisodeRunner

    assert "_disable_motion" in inspect.getsource(BrowserEpisodeRunner.run_episode)


# ---------------------------------------------------------------------------
# SQLite as the single source of truth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env_dir", PREMADE)
def test_premade_app_state_is_sqlite_backed(env_dir):
    app = (ROOT / env_dir / "app.py").read_text()
    assert 'sqlite:///' in app, f"{env_dir}/app.py must persist state in SQLite"


@pytest.mark.parametrize("env_dir", PREMADE)
def test_premade_forge_state_reads_from_db(env_dir):
    app = (ROOT / env_dir / "app.py").read_text()
    state_fn = app[app.index("def forge_state"):]
    state_fn = state_fn[: state_fn.index("\n@")]  # body up to the next route
    assert "_get_state_dict(db)" in state_fn, (
        f"{env_dir}: /forge/state must read from the DB session — "
        "the DB is the single source of truth"
    )


def test_generated_apps_are_required_to_use_sqlite():
    src = (ROOT / "forge/envgen/agents/app_generator.py").read_text()
    assert "SQLite persistence" in src


def test_container_verification_observes_db_state_not_ui():
    src = (ROOT / "forge/envgen/episode_runner.py").read_text()
    assert '"/forge/state"' in src, (
        "episode verification must observe /forge/state (DB-backed), never scrape the UI"
    )
