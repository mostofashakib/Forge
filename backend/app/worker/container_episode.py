"""Run one agent episode attempt against a containerized environment.

Each environment type has its own runner:
  "cli"     → CliEpisodeRunner (docker exec shell)
  "browser" → BrowserEpisodeRunner (Playwright CDP)
  otherwise → ContainerEpisodeRunner (HTTP FastAPI, general and premade apps)

Runner classes are imported where they are used, so tests can patch them at
their source modules and the worker starts without loading Playwright.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from backend.app.worker.env_artifacts import load_manifest, load_personas
from forge.settings import experiment_seed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContainerEpisode:
    """Everything one episode needs, read from the run and sandbox rows up front."""

    run_id: str
    episode_id: str
    episode_index: int
    seed: int
    num_episodes: int
    env_name: str
    env_type: str
    env_dir: Path
    jsonl_path: Path
    agent_id: str
    objective: str
    max_steps: int
    dead_end_patience: int
    success_threshold: float
    container_port: int | None = None
    cdp_port: int | None = None

    def run_attempt(self, attempt: int, attempt_seed: int):
        """Run one attempt. The reliability executor calls this once per retry."""
        from forge.envgen.container import ContainerRuntime

        runner = _RUNNERS.get(self.env_type, _run_http_attempt)
        return runner(self, ContainerRuntime(), attempt_seed)

    def limits(self) -> dict:
        """Objective and stopping rules shared by every runner config."""
        return {
            "objective": self.objective,
            "max_steps": self.max_steps,
            "dead_end_patience": self.dead_end_patience,
            "success_threshold": self.success_threshold,
        }

    @property
    def is_last(self) -> bool:
        return self.episode_index + 1 >= self.num_episodes


def _episode_snapshot(runtime, env_name: str, seed: int) -> str | None:
    """The seeded initial snapshot, else the seed-0 one, else None."""
    from forge.envgen.docker_images import initial_snapshot_tag

    for candidate_seed in (seed, 0):
        tag = initial_snapshot_tag(env_name, candidate_seed)
        if runtime.image_exists(tag):
            return tag
    return None


def _cli_agent(episode: ContainerEpisode, attempt_seed: int):
    """Replay a recorded synthetic trajectory when the env has one, else build the agent."""
    from forge.envgen.agents.cli_agent import ReplayCliAgent, make_cli_agent

    replay_path = episode.env_dir / "synthetic_replay.json"
    if replay_path.exists():
        trajectories = json.loads(replay_path.read_text(encoding="utf-8")).get("episodes", [])
        if trajectories:
            index = attempt_seed % len(trajectories)
            logger.info(
                "[container-ep] using replay agent seed=%d → trajectory %d (%d commands)",
                attempt_seed, index, len(trajectories[index]),
            )
            return ReplayCliAgent(trajectories[index])
    return make_cli_agent(episode.agent_id, seed=experiment_seed(attempt_seed))


def _cli_reward_engine(env_name: str):
    from backend.app.services.reward_config import load_reward_config
    from forge.envgen.tiered_reward import TieredRewardConfig, TieredRewardEngine

    reward_cfg = load_reward_config(env_name)
    return TieredRewardEngine(
        config=TieredRewardConfig.from_preset(
            reward_cfg.reward_preset,
            partial_credit_methods=reward_cfg.scoring_methods,
        )
    )


def _run_cli_attempt(episode: ContainerEpisode, runtime, attempt_seed: int):
    """Fork the run's shell snapshot, run in the fork, and drop the snapshot after the last episode."""
    from forge.envgen.cli_runner import CliEpisodeConfig, CliEpisodeRunner
    from forge.envgen.docker_images import cli_snapshot_tag

    reward_engine = _cli_reward_engine(episode.env_name)
    agent = _cli_agent(episode, attempt_seed)
    snapshot = cli_snapshot_tag(episode.env_name, episode.run_id)
    with runtime.cli_episode(
        episode.env_name, snapshot, refill=not episode.is_last,
    ) as episode_container:
        config = CliEpisodeConfig(container_id=episode_container, **episode.limits())
        result = CliEpisodeRunner(config, reward_engine=reward_engine).run_episode(
            agent, episode_id=episode.episode_id, jsonl_path=episode.jsonl_path
        )
    if episode.is_last:
        runtime.discard_cli_snapshot(snapshot)
    return result


def _run_browser_attempt(episode: ContainerEpisode, runtime, attempt_seed: int):
    """Run in a clone of the initial snapshot, or the live browser when there is none."""
    from forge.envgen.agents.browser_agent import make_browser_agent

    agent = make_browser_agent(episode.agent_id, seed=experiment_seed(attempt_seed))
    snapshot = _episode_snapshot(runtime, episode.env_name, episode.seed)
    if snapshot is None:
        return _run_browser_on(episode, agent, episode.cdp_port)
    with runtime.cloned_episode(
        episode.env_name, snapshot, env_type="browser", episode_id=episode.episode_id,
    ) as (_, port):
        return _run_browser_on(episode, agent, port)


def _run_browser_on(episode: ContainerEpisode, agent, cdp_port: int | None):
    from forge.envgen.browser_runner import BrowserEpisodeConfig, BrowserEpisodeRunner

    config = BrowserEpisodeConfig(cdp_url=f"http://localhost:{cdp_port}", **episode.limits())
    return BrowserEpisodeRunner(config).run_episode(
        agent, episode_id=episode.episode_id, jsonl_path=episode.jsonl_path
    )


def _run_http_attempt(episode: ContainerEpisode, runtime, attempt_seed: int):
    """Run in a clone of the initial snapshot, or the live app when there is none."""
    from forge.envgen.agents.container_agent import make_container_agent

    agent = make_container_agent(episode.agent_id, seed=experiment_seed(attempt_seed))
    snapshot = _episode_snapshot(runtime, episode.env_name, episode.seed)
    if snapshot is None:
        if episode.container_port is None:
            raise RuntimeError(f"General sandbox {episode.env_name} has no container_port")
        return _run_http_on(episode, agent, episode.container_port)
    with runtime.cloned_episode(
        episode.env_name, snapshot, env_type="general", episode_id=episode.episode_id,
    ) as (_, port):
        return _run_http_on(episode, agent, port)


def _run_http_on(episode: ContainerEpisode, agent, port: int):
    from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig

    config = EpisodeConfig(
        base_url=f"http://localhost:{port}",
        personas=load_personas(episode.env_dir),
        **episode.limits(),
    )
    with ContainerEpisodeRunner(config, manifest=load_manifest(episode.env_dir)) as runner:
        return runner.run_episode(agent, episode_id=episode.episode_id, jsonl_path=episode.jsonl_path)


_RUNNERS = {"cli": _run_cli_attempt, "browser": _run_browser_attempt}


def browser_cdp_port(container_id: str) -> int:
    """The gateway's published DevTools port for a browser container."""
    import docker

    from forge.envgen.container import ContainerRuntime

    client = docker.from_env()
    try:
        container = client.containers.get(container_id)
        container.reload()
        port = ContainerRuntime(client).host_port(container, 9222)
    finally:
        client.close()
    if not port:
        raise RuntimeError(
            "The browser's DevTools port is not published through its gateway. "
            "Restart the environment to recreate it behind the gateway."
        )
    return port
