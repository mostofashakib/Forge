"""App containers cannot reach the internet, but Forge can still reach them.

A custom or premade app sits on an `internal` Docker network, which has no
route out. Docker cannot publish a port from such a network, so each app
gets a small gateway container: it publishes the loopback port and forwards
only to its own app. Every lifecycle step treats the pair as one unit.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from forge.envgen.container import (
    FORGE_GATEWAY_IMAGE,
    ContainerRuntime,
    sandbox_network_name,
)
from tests.envgen.fake_docker import FakeDocker

APP_IMAGE = "forge-env-mail:latest"


@pytest.fixture
def daemon():
    fake = FakeDocker()
    with patch("forge.envgen.container.docker.from_env", return_value=fake), \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        yield fake


def test_the_app_joins_only_the_internal_network_and_publishes_nothing(daemon):
    app_id, _port = ContainerRuntime().run("mail", APP_IMAGE)

    app = daemon.containers.get(app_id)
    assert app.networks == [sandbox_network_name("mail")]
    assert not app.kwargs.get("ports")
    assert daemon.networks.get(sandbox_network_name("mail")).kwargs["internal"] is True


def test_the_gateway_publishes_the_port_and_forwards_only_to_its_app(daemon):
    _app_id, port = ContainerRuntime().run("mail", APP_IMAGE)

    gateway = daemon.named("forge-mail-gw")
    assert gateway.kwargs["image"] == FORGE_GATEWAY_IMAGE
    assert gateway.kwargs["ports"] == {"8000/tcp": ("127.0.0.1", None)}
    assert "TCP-LISTEN:8000" in " ".join(gateway.kwargs["command"])
    assert "TCP:forge-mail:8000" in " ".join(gateway.kwargs["command"])
    assert sandbox_network_name("mail") in gateway.networks
    assert port == int(gateway.ports["8000/tcp"][0]["HostPort"])


def test_each_environment_gets_its_own_internal_network(daemon):
    # A shared network would let one environment's agent reach another
    # environment's app, control plane included.
    runtime = ContainerRuntime()
    mail_id, _ = runtime.run("mail", APP_IMAGE)
    chat_id, _ = runtime.run("chat", "forge-env-chat:latest")

    assert sandbox_network_name("mail") != sandbox_network_name("chat")
    assert daemon.containers.get(mail_id).networks == [sandbox_network_name("mail")]
    assert daemon.containers.get(chat_id).networks == [sandbox_network_name("chat")]


def test_a_rerun_reuses_the_environments_network(daemon):
    runtime = ContainerRuntime()
    runtime.run("mail", APP_IMAGE)
    network = daemon.networks.get(sandbox_network_name("mail"))
    runtime._remove_existing("mail")
    runtime.run("mail", APP_IMAGE)

    assert daemon.networks.get(sandbox_network_name("mail")) is network


def test_host_port_reads_the_gateway_for_apps_and_browsers(daemon):
    runtime = ContainerRuntime()
    app_id, app_port = runtime.run("mail", APP_IMAGE)
    browser_id, ui_port = runtime.run_browser("web")

    assert runtime.host_port(daemon.containers.get(app_id)) == app_port
    assert runtime.host_port(daemon.containers.get(browser_id)) == ui_port


# ---------------------------------------------------------------------------
# Browser: no internet either
# ---------------------------------------------------------------------------

def test_the_browser_has_no_route_out_and_publishes_nothing_itself(daemon):
    browser_id, _ = ContainerRuntime().run_browser("web")

    browser = daemon.containers.get(browser_id)
    assert browser.networks == [sandbox_network_name("web")]
    assert not browser.kwargs.get("ports")
    assert daemon.networks.get(sandbox_network_name("web")).kwargs["internal"] is True


def test_the_browser_gateway_forwards_the_ui_and_devtools_ports(daemon):
    runtime = ContainerRuntime()
    browser_id, ui_port = runtime.run_browser("web")

    gateway = daemon.named("forge-web-gw")
    assert set(gateway.kwargs["ports"]) == {"3000/tcp", "9222/tcp"}
    script = " ".join(gateway.kwargs["command"])
    assert "TCP-LISTEN:3000" in script and "TCP:forge-web:3000" in script
    # DevTools goes through the in-namespace relay, since Chromium only
    # listens on its own loopback.
    assert "TCP-LISTEN:9222" in script and "TCP:forge-web:9223" in script
    assert ui_port == int(gateway.ports["3000/tcp"][0]["HostPort"])
    cdp = runtime.host_port(daemon.containers.get(browser_id), 9222)
    assert cdp == int(gateway.ports["9222/tcp"][0]["HostPort"])


def test_removing_an_environment_removes_its_network(daemon):
    runtime = ContainerRuntime()
    app_id, _ = runtime.run("mail", APP_IMAGE)

    runtime.remove(app_id, APP_IMAGE)

    with pytest.raises(Exception):
        daemon.networks.get(sandbox_network_name("mail"))


def test_an_app_without_its_gateway_has_no_host_port(daemon):
    # An app started before this change, or whose gateway was removed, is
    # unreachable. Reporting no port lets the UI offer a restart.
    runtime = ContainerRuntime()
    app_id, _ = runtime.run("mail", APP_IMAGE)
    daemon.named("forge-mail-gw").remove(force=True)

    assert runtime.host_port(daemon.containers.get(app_id)) is None


def test_stop_and_remove_take_the_gateway_with_the_app(daemon):
    runtime = ContainerRuntime()
    app_id, _ = runtime.run("mail", APP_IMAGE)

    runtime.stop(app_id)
    assert daemon.named("forge-mail-gw").status == "exited"

    runtime.remove(app_id, APP_IMAGE)
    assert {"forge-mail", "forge-mail-gw"} <= set(daemon.removed)


def test_start_restarts_the_pair_and_returns_the_gateway_port(daemon):
    runtime = ContainerRuntime()
    app_id, _ = runtime.run("mail", APP_IMAGE)
    runtime.stop(app_id)

    cid, port = runtime.start("mail", app_id, APP_IMAGE)

    gateway = daemon.named("forge-mail-gw")
    assert cid == app_id
    assert gateway.status == "running"
    assert port == int(gateway.ports["8000/tcp"][0]["HostPort"])


def test_start_runs_a_fresh_pair_when_the_gateway_is_missing(daemon):
    runtime = ContainerRuntime()
    old_id, _ = runtime.run("mail", APP_IMAGE)
    daemon.named("forge-mail-gw").remove(force=True)

    cid, port = runtime.start("mail", old_id, APP_IMAGE)

    assert cid != old_id
    assert port == int(daemon.named("forge-mail-gw").ports["8000/tcp"][0]["HostPort"])


def test_reattach_reports_the_app_with_its_gateway_port_and_never_the_gateway(daemon):
    runtime = ContainerRuntime()
    app_id, port = runtime.run("mail", APP_IMAGE)

    assert runtime.reattach_all() == [("mail", app_id, port)]


def test_a_rerun_replaces_both_containers_instead_of_conflicting(daemon):
    runtime = ContainerRuntime()
    runtime.run("mail", APP_IMAGE)

    runtime._remove_existing("mail")
    app_id, _ = runtime.run("mail", APP_IMAGE)

    assert daemon.containers.get(app_id).name == "forge-mail"


def test_the_browser_image_keeps_its_own_init_as_pid_1(daemon):
    # The linuxserver image boots through s6-overlay, which exits with
    # "can only run as pid 1" if Docker inserts tini ahead of it. s6 reaps
    # zombies itself, so the browser must not get Docker's init.
    browser_id, _ = ContainerRuntime().run_browser("web")

    assert not daemon.containers.get(browser_id).kwargs.get("init")


def test_a_devtools_relay_shares_the_browsers_network_namespace(daemon):
    # Chromium binds DevTools to 127.0.0.1 whatever flags it gets, so only a
    # process inside its network namespace can reach it. The relay lives
    # there, so it has no route out either.
    browser_id, _ = ContainerRuntime().run_browser("web")

    relay = daemon.named("forge-web-cdp")
    assert relay.kwargs["image"] == FORGE_GATEWAY_IMAGE
    assert relay.kwargs["network_mode"] == f"container:{browser_id}"
    script = " ".join(relay.kwargs["command"])
    assert "TCP-LISTEN:9223" in script and "TCP:127.0.0.1:9222" in script
    assert not relay.kwargs.get("ports")


def test_stop_start_and_remove_handle_the_devtools_relay(daemon):
    runtime = ContainerRuntime()
    browser_id, _ = runtime.run_browser("web")

    runtime.stop(browser_id)
    assert daemon.named("forge-web-cdp").status == "exited"
    runtime.start("web", browser_id, "builtin:browser")
    assert daemon.named("forge-web-cdp").status == "running"
    runtime.remove(browser_id, "builtin:browser")
    assert {"forge-web", "forge-web-gw", "forge-web-cdp"} <= set(daemon.removed)
