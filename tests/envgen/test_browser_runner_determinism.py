"""Browser episodes must not depend on the host's clock or its current load.

Two sources of drift are pinned here. A page reads `Date.now()`, so an app
that renders "today" or relative times shows a different page every day.
And a fixed sleep after each action is too short when the host is busy, so
the "after" screenshot sometimes catches a half-loaded page.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from forge.envgen.browser_runner import (
    BROWSER_FIXED_TIME,
    BrowserEpisodeConfig,
    BrowserEpisodeRunner,
)


class _FakePage:
    def __init__(self, events: list) -> None:
        self._events = events
        self.url = "http://app.local/"
        self.mouse = MagicMock()
        self.keyboard = MagicMock()

    def screenshot(self, type: str) -> bytes:
        self._events.append(("screenshot",))
        return b"png"

    def wait_for_load_state(self, state: str, timeout: float) -> None:
        self._events.append(("wait_for_load_state", state, timeout))

    def goto(self, url: str, wait_until: str, timeout: float) -> None:
        self._events.append(("goto", url, timeout))

    def emulate_media(self, **_kwargs) -> None:
        pass

    def add_style_tag(self, **_kwargs) -> None:
        pass


class _FakeContext:
    def __init__(self, events: list) -> None:
        self._events = events
        self.pages: list[_FakePage] = []  # a new context opens with no tabs
        self.closed = False
        self.clock = MagicMock()
        self.clock.set_fixed_time.side_effect = (
            lambda when: events.append(("set_fixed_time", when))
        )

    def add_init_script(self, _script: str) -> None:
        pass

    def new_page(self) -> _FakePage:
        page = _FakePage(self._events)
        self.pages.append(page)
        return page

    def close(self) -> None:
        self.closed = True


def _run(actions: list[dict], *, shared: "_FakeContext | None" = None, fresh: "_FakeContext | None" = None) -> list:
    """Run one episode against a fake CDP browser and return the page events."""
    events: list = []
    ctx = fresh or _FakeContext(events)
    browser = MagicMock(contexts=[shared or _FakeContext([])])
    browser.new_context.return_value = ctx
    pw = MagicMock()
    pw.chromium.connect_over_cdp.return_value = browser

    @contextmanager
    def fake_sync_playwright():
        yield pw

    agent = MagicMock()
    agent.act.side_effect = [*actions, {"action_type": "submit"}]
    scorer = MagicMock()
    scorer.score_with_image.return_value = 1.0
    runner = BrowserEpisodeRunner(
        BrowserEpisodeConfig(cdp_url="http://cdp", objective="o", max_steps=5),
        scorer=scorer,
    )
    with patch("playwright.sync_api.sync_playwright", fake_sync_playwright), \
         patch.object(runner, "_wait_for_cdp", return_value=True), \
         patch("forge.envgen.browser_runner.time.sleep") as sleep:
        runner.run_episode(agent)
    events.append(("sleep_calls", sleep.call_count))
    return events


def test_the_page_clock_is_frozen_before_the_agent_sees_anything():
    events = _run([])

    assert events[0] == ("set_fixed_time", BROWSER_FIXED_TIME)


def test_each_action_waits_for_the_page_to_settle_instead_of_sleeping():
    events = _run([{"action_type": "click", "x": 1, "y": 2}])

    waits = [e for e in events if e[0] == "wait_for_load_state"]
    assert waits == [
        ("wait_for_load_state", "networkidle", BrowserEpisodeConfig.settle_timeout_ms)
    ]
    assert ("sleep_calls", 0) in events


def test_the_settle_wait_happens_between_the_action_and_the_after_screenshot():
    # False-positive guard: waiting after the screenshot would not help.
    events = _run([{"action_type": "click", "x": 1, "y": 2}])

    names = [e[0] for e in events]
    wait_at = names.index("wait_for_load_state")
    assert names[wait_at + 1] == "screenshot"


def test_navigation_gets_the_same_generous_fixed_timeout():
    events = _run([{"action_type": "navigate", "url": "http://app.local/inbox"}])

    gotos = [e for e in events if e[0] == "goto"]
    assert gotos == [("goto", "http://app.local/inbox", BrowserEpisodeConfig.settle_timeout_ms)]


def test_each_episode_gets_a_fresh_context_and_closes_it():
    # The browser process stays warm, but cookies, storage, and tabs from the
    # last episode must not carry into this one.
    shared_events: list = []
    shared = _FakeContext(shared_events)
    shared.new_page()
    fresh = _FakeContext([])

    _run([{"action_type": "click", "x": 1, "y": 2}], shared=shared, fresh=fresh)

    assert shared_events == []  # the long-lived default context was never touched
    assert fresh.closed is True
