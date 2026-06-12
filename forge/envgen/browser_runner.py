from __future__ import annotations
import base64
import logging
import time
from dataclasses import dataclass
from functools import partial
from datetime import datetime, timezone
from pathlib import Path

from forge.envgen.episode_base import (
    BaseEpisodeConfig,
    BaseEpisodeResult,
    TerminationMonitor,
)
from forge.envgen.objective import ObjectiveScorer
from forge.runtime.interaction import BrowserUse, BrowserUseSchema

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class BrowserEpisodeConfig(BaseEpisodeConfig):
    cdp_url: str       # e.g. "http://localhost:9222"
    max_steps: int = 20
    action_settle_s: float = 1.0   # seconds to wait after each action


@dataclass(kw_only=True)
class BrowserEpisodeResult(BaseEpisodeResult):
    pass


class BrowserEpisodeRunner:
    def __init__(self, config: BrowserEpisodeConfig, scorer: ObjectiveScorer | None = None) -> None:
        self._cfg = config
        self._scorer = scorer or ObjectiveScorer()

    def _wait_for_cdp(self, max_retries: int = 20, delay: float = 3.0) -> bool:
        import requests
        for attempt in range(max_retries):
            try:
                resp = requests.get(f"{self._cfg.cdp_url}/json/version", timeout=5)
                if resp.status_code == 200:
                    logger.info("[browser-ep] CDP ready after %d attempt(s)", attempt + 1)
                    return True
            except Exception as exc:
                logger.debug("[browser-ep] CDP attempt %d/%d: %s", attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                time.sleep(delay)
        logger.error("[browser-ep] CDP at %s not reachable after %d attempts", self._cfg.cdp_url, max_retries)
        return False

    # Killing motion keeps screenshots stable between identical runs, so
    # vision-based scoring doesn't flake on half-played transitions.
    _NO_MOTION_CSS = (
        "*, *::before, *::after {"
        " animation: none !important;"
        " transition: none !important;"
        " scroll-behavior: auto !important; }"
    )

    @staticmethod
    def _disable_motion(ctx, page) -> None:
        """Disable CSS animations/transitions on the current and all future pages."""
        css = BrowserEpisodeRunner._NO_MOTION_CSS
        init_script = (
            "const apply = () => {"
            " const s = document.createElement('style');"
            f" s.textContent = `{css}`;"
            " (document.head || document.documentElement).appendChild(s); };"
            "document.readyState === 'loading'"
            " ? document.addEventListener('DOMContentLoaded', apply) : apply();"
        )
        try:
            ctx.add_init_script(init_script)
            page.emulate_media(reduced_motion="reduce")
            page.add_style_tag(content=css)
        except Exception as exc:
            logger.warning("[browser-ep] could not disable animations: %s", exc)

    @staticmethod
    def _screenshot(page) -> str:
        return base64.b64encode(page.screenshot(type="png")).decode()

    @staticmethod
    def browser_use_for(page, schema: BrowserUseSchema | None = None) -> BrowserUse:
        """The BrowserUse contract a browser environment grants the agent."""
        return BrowserUse(
            schema=schema or BrowserUseSchema(),
            executor=partial(BrowserEpisodeRunner._apply_action, page),
        )

    @staticmethod
    def _apply_action(page, action: dict) -> None:
        atype = action.get("action_type", "noop")
        if atype == "click":
            page.mouse.click(action.get("x", 0), action.get("y", 0))
        elif atype == "type":
            page.keyboard.type(action.get("text", ""))
        elif atype == "press":
            page.keyboard.press(action.get("key", "Return"))
        elif atype == "navigate":
            url = action.get("url", "")
            if url:
                page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        elif atype == "scroll":
            page.mouse.wheel(action.get("delta_x", 0), action.get("delta_y", 0))

    def run_episode(self, agent, episode_id: str | None = None, jsonl_path: Path | None = None) -> BrowserEpisodeResult:
        from playwright.sync_api import sync_playwright

        result = BrowserEpisodeResult()

        if not self._wait_for_cdp():
            result.termination_reason = "container_unreachable"
            result.completed_at = datetime.now(timezone.utc)
            if jsonl_path is not None:
                result.write_jsonl(jsonl_path)
            return result

        monitor = TerminationMonitor(self._cfg)

        try:
            with sync_playwright() as pw:
                browser = pw.chromium.connect_over_cdp(self._cfg.cdp_url)
                ctx = browser.contexts[0] if browser.contexts else browser.new_context()
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                self._disable_motion(ctx, page)
                browser_use = self.browser_use_for(page)

                for step_idx in range(self._cfg.max_steps):
                    ss_before = self._screenshot(page)
                    current_url = page.url

                    try:
                        action = agent.act(
                            screenshot_b64=ss_before,
                            objective=self._cfg.objective,
                            action_history=[s["action"] for s in result.steps[-5:]],
                        )
                    except Exception as exc:
                        logger.warning("[browser-ep] step %d: agent.act failed: %s", step_idx, exc)
                        action = {"action_type": "noop", "reasoning": f"agent error: {exc}"}

                    try:
                        browser_use.execute(action)
                        time.sleep(self._cfg.action_settle_s)
                    except Exception as exc:
                        logger.debug("[browser-ep] step %d: action failed: %s", step_idx, exc)

                    ss_after = self._screenshot(page)
                    score = self._scorer.score_with_image(ss_after, page.url, self._cfg.objective)
                    result.total_reward += score

                    step_record = {
                        "step_index": step_idx,
                        "action": action,
                        "screenshot_before": ss_before,
                        "screenshot_after": ss_after,
                        "url_before": current_url,
                        "url_after": page.url,
                        "objective_score": score,
                        "reward": score,
                    }
                    result.steps.append(step_record)
                    result.final_objective_score = score

                    logger.info(
                        "[browser-ep] step %02d/%d  action=%s  score=%.2f  url=%s",
                        step_idx + 1, self._cfg.max_steps,
                        action.get("action_type"), score, page.url[:60],
                    )

                    reason = monitor.observe(score)
                    if reason is not None:
                        result.termination_reason = reason
                        break
                else:
                    result.termination_reason = "max_steps"

        except Exception as exc:
            logger.exception("[browser-ep] runner crashed: %s", exc)
            result.termination_reason = f"runner_error: {exc}"

        result.completed_at = datetime.now(timezone.utc)
        if result.steps:
            result.total_reward = result.total_reward / len(result.steps)
        if jsonl_path is not None:
            result.write_jsonl(jsonl_path)
        return result
