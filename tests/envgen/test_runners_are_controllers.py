from __future__ import annotations

import inspect

import pytest

from forge.contracts import EpisodeController
from forge.envgen.browser_runner import BrowserEpisodeRunner
from forge.envgen.cli_runner import CliEpisodeRunner
from forge.envgen.episode_runner import ContainerEpisodeRunner

RUNNERS = [ContainerEpisodeRunner, CliEpisodeRunner, BrowserEpisodeRunner]


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda r: r.__name__)
def test_every_runner_is_an_episode_controller(runner):
    assert issubclass(runner, EpisodeController)


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda r: r.__name__)
def test_every_runner_accepts_the_same_keywords(runner):
    params = inspect.signature(runner.run_episode).parameters
    assert {"agent", "episode_id", "seed", "jsonl_path"} <= set(params)


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda r: r.__name__)
def test_seed_is_keyword_only(runner):
    # False-positive guard: uniform keywords must not become positional, or a
    # caller could pass jsonl_path into seed.
    params = inspect.signature(runner.run_episode).parameters
    assert params["seed"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda r: r.__name__)
def test_positional_seed_is_rejected(runner):
    # Negative case: Python itself must refuse a call that passes episode_id,
    # seed, and jsonl_path positionally after agent, not merely report them as
    # KEYWORD_ONLY in the abstract. This is the failure `seed`-as-positional
    # would let slip through: a caller passing jsonl_path into seed's slot.
    sig = inspect.signature(runner.run_episode)
    with pytest.raises(TypeError):
        sig.bind(object(), object(), "episode-1", 7, "trajectory.jsonl")
