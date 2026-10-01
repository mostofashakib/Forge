"""The browser runner: task pages served by request interception, DOM and URL checks.

The Playwright surface is faked: a page with a tiny DOM, a context that
records its route handler, and a browser that hands out contexts.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from forge.taskfactory.pass_k import run_pass_k
from forge.taskfactory.runners.browser import TASK_ORIGIN, BrowserRunner, evaluate_dom_check, perform_step
from forge.taskfactory.schemas import Check, GoldenStep, TaskDraft, TaskPage, TaskSeed


class FakeError(Exception):
    pass


class FakeLocator:
    def __init__(self, page: "FakePage", selector: str) -> None:
        self._page, self._selector = page, selector

    def count(self) -> int:
        return 1 if self._selector in self._page.dom else 0

    @property
    def first(self) -> "FakeLocator":
        return self

    def inner_text(self, timeout=None) -> str:
        return self._page.dom[self._selector]["text"]

    def input_value(self, timeout=None) -> str:
        return self._page.dom[self._selector].get("value", "")

    def get_attribute(self, name, timeout=None):
        return self._page.dom[self._selector].get(name)


class FakePage:
    def __init__(self) -> None:
        self.dom: dict[str, dict] = {}
        self.url = "about:blank"
        self.calls: list[tuple] = []

    def goto(self, url, **kwargs):
        self.calls.append(("goto", url))
        self.url = url
        path = url.removeprefix(TASK_ORIGIN)
        self.dom = {"#name": {"text": "", "value": ""}, "#go": {"text": "Go"}, "#status": {"text": ""}} if path == "/form" else {}

    def fill(self, selector, value, **kwargs):
        self._need(selector)
        self.dom[selector]["value"] = value

    def click(self, selector, **kwargs):
        self._need(selector)
        if selector == "#go":
            self.dom["#status"]["text"] = f"Submitted for {self.dom['#name']['value']}"

    def select_option(self, selector, value, **kwargs):
        self._need(selector)

    def check(self, selector, **kwargs):
        self._need(selector)

    def press(self, selector, key, **kwargs):
        self.calls.append(("press", selector, key))

    def locator(self, selector):
        return FakeLocator(self, selector)

    def content(self):
        return repr(sorted(self.dom.items()))

    def wait_for_load_state(self, *args, **kwargs):
        pass

    def _need(self, selector):
        if selector not in self.dom:
            raise FakeError(f"no element matches {selector}")


class FakeClock:
    def __init__(self) -> None:
        self.fixed = None

    def set_fixed_time(self, value):
        self.fixed = value


class FakeContext:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.route_handler = None
        self.closed = False
        self.page = FakePage()

    def route(self, pattern, handler):
        self.route_handler = handler

    def new_page(self):
        return self.page

    def close(self):
        self.closed = True


class FakeBrowser:
    def __init__(self) -> None:
        self.contexts: list[FakeContext] = []

    def new_context(self):
        context = FakeContext()
        self.contexts.append(context)
        return context


def _runner(browser: FakeBrowser) -> BrowserRunner:
    @contextmanager
    def connect():
        yield browser

    return BrowserRunner(connect)


DRAFT = TaskDraft(
    slot=0, title="Submit the form", objective="Submit the form as Sam.",
    seed=TaskSeed(pages=[TaskPage(path="/form", html="<form>…</form>")], start_path="/form"),
    golden=[
        GoldenStep(tool="fill", args={"selector": "#name", "value": "Sam"}),
        GoldenStep(tool="click", args={"selector": "#go"}),
    ],
    checks=[Check(kind="dom", selector="#status", prop="text", op="equals", value="Submitted for Sam")],
)


def test_a_sound_task_passes_k_runs_each_in_a_fresh_closed_context():
    browser = FakeBrowser()
    runner = _runner(browser)

    with runner.batch():
        result = run_pass_k(runner, DRAFT, k=3)

    assert result.passed, result.reason
    assert len(browser.contexts) == 3
    assert all(c.closed for c in browser.contexts)
    assert all(c.clock.fixed is not None for c in browser.contexts)


def test_the_session_opens_the_start_page_on_the_task_origin():
    browser = FakeBrowser()
    runner = _runner(browser)

    with runner.batch(), runner.session(DRAFT):
        pass

    assert browser.contexts[0].page.calls[0] == ("goto", f"{TASK_ORIGIN}/form")


class FakeRoute:
    def __init__(self, url: str) -> None:
        self.request = type("Req", (), {"url": url})()
        self.fulfilled = None
        self.aborted = False

    def fulfill(self, **kwargs):
        self.fulfilled = kwargs

    def abort(self, *args):
        self.aborted = True


def test_task_pages_are_served_and_every_other_request_is_aborted():
    browser = FakeBrowser()
    runner = _runner(browser)
    with runner.batch(), runner.session(DRAFT):
        handler = browser.contexts[0].route_handler

    page_route, other_route, missing_route = (
        FakeRoute(f"{TASK_ORIGIN}/form"), FakeRoute("https://example.com/x"), FakeRoute(f"{TASK_ORIGIN}/nope"),
    )
    for route in (page_route, other_route, missing_route):
        handler(route)

    assert page_route.fulfilled["body"] == "<form>…</form>"
    assert other_route.aborted
    assert missing_route.fulfilled["status"] == 404


def test_a_step_on_a_missing_element_fails_instead_of_raising():
    page = FakePage()
    page.goto(f"{TASK_ORIGIN}/form")

    result = perform_step(page, GoldenStep(tool="click", args={"selector": "#nope"}))

    assert not result.ok
    assert "#nope" in result.error


def test_goto_resolves_paths_against_the_task_origin():
    page = FakePage()

    perform_step(page, GoldenStep(tool="goto", args={"path": "/form"}))

    assert page.url == f"{TASK_ORIGIN}/form"


@pytest.mark.parametrize("check, passed", [
    (Check(kind="dom", selector="#go", op="exists"), True),
    (Check(kind="dom", selector="#missing", op="exists"), False),
    (Check(kind="dom", selector="#missing", op="absent"), True),
    (Check(kind="dom", selector="#go", prop="text", op="contains", value="G"), True),
    (Check(kind="dom", selector="#go", prop="text", op="equals", value="Stop"), False),
    (Check(kind="dom", selector="#missing", prop="text", op="equals", value="x"), False),
    (Check(kind="url", op="equals", value="/form"), True),
    (Check(kind="url", op="contains", value="thanks"), False),
])
def test_dom_and_url_checks(check, passed):
    page = FakePage()
    page.goto(f"{TASK_ORIGIN}/form")

    assert evaluate_dom_check(page, check).passed is passed


def test_a_session_outside_a_batch_is_refused():
    with pytest.raises(RuntimeError, match="batch"):
        with _runner(FakeBrowser()).session(DRAFT):
            pass
