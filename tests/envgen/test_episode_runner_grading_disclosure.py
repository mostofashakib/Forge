"""The container episode runner must disclose how many verdicts a model issued.

Every step of a container episode is scored by :class:`ObjectiveScorer`, so this
path is LLM-graded regardless of reward preset. The run record has to say so,
and it must say so from what was observed rather than from what the preset
implies.
"""
from __future__ import annotations

import httpx

from forge.envgen.episode_runner import (
    ContainerEpisodeRunner,
    EpisodeConfig,
    EpisodeResult,
    TerminationMonitor,
)


class _CountingScorer:
    """Stands in for ObjectiveScorer; returns a score below every threshold."""

    def __init__(self, score: float = 0.1) -> None:
        self.calls = 0
        self._score = score

    def score(self, *args, **kwargs) -> float:
        self.calls += 1
        return self._score


class _FixedAgent:
    def act(self, state, objective, available_actions) -> dict:
        return {"endpoint": "/act", "payload": {}}


def _runner(max_steps: int, scorer: _CountingScorer) -> ContainerEpisodeRunner:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/forge/state":
            return httpx.Response(200, json={"n": 1})
        return httpx.Response(200, json={"ok": True})

    config = EpisodeConfig(
        base_url="http://c", objective="do it", max_steps=max_steps,
        consecutive_below_threshold=99,
    )
    runner = ContainerEpisodeRunner(config, scorer=scorer)
    runner._http = httpx.Client(
        base_url="http://c", transport=httpx.MockTransport(handler)
    )
    runner._actions = [{"endpoint": "/act", "method": "post", "params": {}}]
    return runner


def _run(max_steps: int, scorer: _CountingScorer) -> EpisodeResult:
    runner = _runner(max_steps, scorer)
    config = runner._cfg
    result = EpisodeResult(episode_id="cep_test", config=config)
    runner._run_steps(
        _FixedAgent(), config, result, TerminationMonitor(config),
        runner._actions, "cep_test", None, {"n": 0},
    )
    return result


def test_episode_counts_one_llm_verdict_per_scored_step():
    scorer = _CountingScorer()

    result = _run(max_steps=3, scorer=scorer)

    assert scorer.calls == 3
    assert result.llm_verdicts == 3


def test_episode_that_runs_no_steps_records_no_llm_verdicts():
    result = EpisodeResult(episode_id="cep_empty", config=EpisodeConfig(
        base_url="http://c", objective="do it"
    ))

    assert result.llm_verdicts == 0


def test_verdict_count_is_observed_not_assumed_from_step_count():
    """False-positive guard: the count must track real scorer calls.

    A count inferred from ``len(steps)`` would look correct here and would be
    wrong the moment a step is recorded without being scored.
    """
    scorer = _CountingScorer()

    result = _run(max_steps=2, scorer=scorer)
    result.steps.append(result.steps[-1])

    assert result.llm_verdicts == 2
    assert result.llm_verdicts != len(result.steps)


def test_episode_summary_discloses_the_verdict_count():
    result = _run(max_steps=1, scorer=_CountingScorer())

    assert result.summary()["llm_verdicts"] == 1
