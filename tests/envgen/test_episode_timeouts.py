"""Episode-path timeouts are generous fixed budgets, not tight guesses.

A timeout that a busy host can hit turns the same action into a different
outcome: a failed step, an empty tool manifest, or an unreachable container.
Each budget below is fixed and far above normal latency, so load changes how
long an episode takes but not what happens in it.
"""
from __future__ import annotations

import inspect

import httpx

from forge.contracts.transport import DEFAULT_TIMEOUT_S
from forge.envgen.browser_runner import BrowserEpisodeRunner
from forge.envgen.cli_runner import CliEpisodeConfig
from forge.envgen.container_env_base import ContainerEnvBase
from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig
from forge.envgen.tiered_reward import TieredRewardConfig
from forge.runtime.http_state import HttpStateManager
from forge.runtime.rest_transport import RestTransport
from forge.runtime.tools import OpenAPIToolProvider


def test_the_shared_http_budget_is_generous():
    assert DEFAULT_TIMEOUT_S >= 60.0


def test_every_http_collaborator_defaults_to_the_shared_budget():
    budgets = {
        "episode_config": EpisodeConfig(base_url="http://app", objective="o").http_timeout,
        "container_env": ContainerEnvBase("http://app").client.timeout.read,
        "rest_transport": RestTransport("http://app")._client.timeout.read,
        "state_manager": HttpStateManager("http://app")._client.timeout.read,
        "tool_discovery": OpenAPIToolProvider(httpx.Client())._timeout,
    }
    assert budgets == dict.fromkeys(budgets, DEFAULT_TIMEOUT_S)


def test_shell_commands_and_their_assertions_get_two_minutes():
    assert CliEpisodeConfig(container_id="c", objective="o").command_timeout == 120.0
    assert TieredRewardConfig().assertion_timeout == 120.0


def _startup_budget_s(wait_fn) -> float:
    params = inspect.signature(wait_fn).parameters
    return params["max_retries"].default * params["delay"].default


def test_containers_get_two_minutes_to_come_up():
    assert _startup_budget_s(ContainerEpisodeRunner.wait_for_health) >= 120.0
    assert _startup_budget_s(BrowserEpisodeRunner._wait_for_cdp) >= 120.0


def test_a_hung_container_still_fails_the_step_instead_of_blocking():
    # Generous is not infinite: a request that never answers still comes back
    # as an in-band failure the runner can record.
    from forge.contracts import TransportRequest

    def hang(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no response", request=request)

    transport = RestTransport(
        "http://app", client=httpx.Client(transport=httpx.MockTransport(hang))
    )
    response = transport.call(TransportRequest(method="POST", target="/act"))

    assert response.status == 0
    assert response.error is not None


def test_an_explicit_timeout_is_not_overridden_by_the_default():
    assert EpisodeConfig(base_url="http://app", objective="o", http_timeout=5.0).http_timeout == 5.0
    assert ContainerEnvBase("http://app", timeout=5.0).client.timeout.read == 5.0
