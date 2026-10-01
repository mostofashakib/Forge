"""Run tasks in the sandboxed browser, serving each task's own pages.

The browser has no internet and no server to load. Each session opens a
fresh context whose requests to http://task.local are answered from the
task's stored pages by Playwright's request interception. Every other
request is aborted, so a session is fully deterministic.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Any
from urllib.parse import urlsplit

from forge.envgen.browser_runner import BROWSER_FIXED_TIME
from forge.taskfactory.runner import StepResult, fingerprint_of
from forge.taskfactory.schemas import Check, GoldenStep, TaskDraft
from forge.taskfactory.state_checks import CheckOutcome

TASK_ORIGIN = "http://task.local"
ACTION_TIMEOUT_MS = 10_000


class BrowserRunner:
    def __init__(self, connect: Callable[[], AbstractContextManager[Any]]) -> None:
        """`connect()` yields a connected Playwright browser for one round."""
        self._connect = connect
        self._browser: Any = None

    @contextmanager
    def batch(self) -> Iterator[None]:
        with self._connect() as browser:
            self._browser = browser
            try:
                yield
            finally:
                self._browser = None

    @contextmanager
    def session(self, draft: TaskDraft) -> Iterator["_BrowserSession"]:
        if self._browser is None:
            raise RuntimeError("a browser session needs an open batch, which holds the connection")
        pages = {page.path: page.html for page in draft.seed.pages}
        context = self._browser.new_context()
        try:
            context.clock.set_fixed_time(BROWSER_FIXED_TIME)
            context.route("**/*", _page_server(pages))
            page = context.new_page()
            page.goto(f"{TASK_ORIGIN}{draft.seed.start_path}", wait_until="domcontentloaded", timeout=ACTION_TIMEOUT_MS)
            yield _BrowserSession(page)
        finally:
            context.close()


def _page_server(pages: dict[str, str]) -> Callable[[Any], None]:
    def handle(route: Any) -> None:
        url = urlsplit(route.request.url)
        if f"{url.scheme}://{url.netloc}" != TASK_ORIGIN:
            route.abort()
        elif url.path in pages:
            route.fulfill(status=200, content_type="text/html; charset=utf-8", body=pages[url.path])
        else:
            route.fulfill(status=404, content_type="text/plain", body="not found")

    return handle


def perform_step(page: Any, step: GoldenStep) -> StepResult:
    """Do one golden step. Any Playwright error is a failed step."""
    args = step.args
    timeout = ACTION_TIMEOUT_MS
    try:
        if step.tool == "goto":
            page.goto(f"{TASK_ORIGIN}{args['path']}", wait_until="domcontentloaded", timeout=timeout)
        elif step.tool == "click":
            page.click(args["selector"], timeout=timeout)
        elif step.tool == "fill":
            page.fill(args["selector"], str(args["value"]), timeout=timeout)
        elif step.tool == "select":
            page.select_option(args["selector"], str(args["value"]), timeout=timeout)
        elif step.tool == "check":
            page.check(args["selector"], timeout=timeout)
        elif step.tool == "press":
            page.press(args.get("selector") or "body", str(args["key"]), timeout=timeout)
        else:
            return StepResult(ok=False, error=f"{step.tool!r} is not a browser step")
        page.wait_for_load_state("domcontentloaded", timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — Playwright raises its own error types
        return StepResult(ok=False, error=f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}")
    return StepResult(ok=True)


def _path(page: Any) -> str:
    url = urlsplit(page.url)
    return url.path + (f"?{url.query}" if url.query else "")


def evaluate_dom_check(page: Any, check: Check) -> CheckOutcome:
    """Evaluate a dom or url check against the live page. Never raises."""
    if check.kind == "url":
        actual = _path(page)
        passed = actual == check.value if check.op == "equals" else str(check.value) in actual
        return CheckOutcome(passed, "" if passed else f"page is at {actual!r}, wanted {check.op} {check.value!r}")
    try:
        locator = page.locator(check.selector)
        count = locator.count()
        if check.op == "exists":
            return CheckOutcome(count > 0, "" if count else f"nothing matches {check.selector}")
        if check.op == "absent":
            return CheckOutcome(count == 0, f"{count} elements match {check.selector}" if count else "")
        if count == 0:
            return CheckOutcome(False, f"nothing matches {check.selector}")
        element = locator.first
        if check.prop == "value":
            actual = element.input_value(timeout=ACTION_TIMEOUT_MS)
        elif check.prop == "attr":
            actual = element.get_attribute(check.attr, timeout=ACTION_TIMEOUT_MS) or ""
        else:
            actual = element.inner_text(timeout=ACTION_TIMEOUT_MS)
    except Exception as exc:  # noqa: BLE001
        return CheckOutcome(False, f"could not read {check.selector}: {str(exc).splitlines()[0][:200]}")
    actual, wanted = actual.strip(), str(check.value).strip()
    passed = actual == wanted if check.op == "equals" else wanted in actual
    return CheckOutcome(passed, "" if passed else f"{check.selector} {check.prop or 'text'} is {actual!r}, wanted {check.op} {wanted!r}")


class _BrowserSession:
    def __init__(self, page: Any) -> None:
        self._page = page

    def step(self, step: GoldenStep) -> StepResult:
        return perform_step(self._page, step)

    def evaluate(self, checks: Sequence[Check]) -> list[CheckOutcome]:
        return [evaluate_dom_check(self._page, check) for check in checks]

    def fingerprint(self) -> str:
        return fingerprint_of({"path": _path(self._page), "dom": self._page.content()})
