"""Personas on the path an actual agent run takes.

`ContainerEpisodeRunner` does not call `ContainerEnvBase.step` — it drives the
environment's collaborators directly. Wiring the cast into `step` alone would
therefore produce personas that exist on the environment, pass every unit test,
and never once act in a real run. These tests pin the path that is actually
used.
"""
from __future__ import annotations

import httpx

from forge.contracts.persona import PersonaDriver, PersonaTurn, PersonaView
from forge.contracts.types import Action
from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig

from tests.personas.conftest import persona, population


class ReplyDriver(PersonaDriver):
    def act(self, view: PersonaView) -> PersonaTurn:
        return PersonaTurn(
            persona_id=view.persona.id,
            trigger=view.trigger,
            action=Action(type="/post_message", params={"__payload__": {}}),
        )


def make_runner(pop=None):
    """A runner over a container that records every action posted to it."""
    posted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/forge/state":
            return httpx.Response(200, json={"posts": len(posted)})
        if path.startswith("/forge/"):
            return httpx.Response(200, json={"ok": True})
        posted.append(path)
        return httpx.Response(200, json={"ok": True})

    cfg = EpisodeConfig(base_url="http://c", objective="do it", personas=pop)
    runner = ContainerEpisodeRunner(cfg)
    runner._http = httpx.Client(
        base_url="http://c", transport=httpx.MockTransport(handler)
    )
    return runner, posted


def act(runner, step_index=0):
    return runner._execute_action(
        {"endpoint": "/post_message", "payload": {}}, step_index
    )


def test_a_run_without_a_cast_posts_only_the_agent_action():
    runner, posted = make_runner()
    runner._reset(seed=1)
    act(runner)
    assert posted == ["/post_message"]


def test_a_configured_cast_acts_on_the_real_run_path():
    pop = population(persona("nurse", wake_on=[], allowed_actions=["/post_message"]))
    runner, posted = make_runner(pop)
    runner.environment._personas._explicit_driver = ReplyDriver()
    runner._reset(seed=1)
    act(runner)
    assert posted == ["/post_message", "/post_message"]


def test_the_cast_is_resolved_by_reset_not_left_empty():
    pop = population(persona("nurse", wake_on=[], allowed_actions=["/post_message"]))
    runner, _ = make_runner(pop)
    runner._reset(seed=1)
    assert [s.profile.id for s in runner.environment.personas.roster] == ["nurse"]


def test_a_disabled_cast_never_acts():
    pop = population(
        persona("nurse", wake_on=[], allowed_actions=["/post_message"]), enabled=False
    )
    runner, posted = make_runner(pop)
    runner.environment._personas._explicit_driver = ReplyDriver()
    runner._reset(seed=1)
    act(runner)
    assert posted == ["/post_message"]


def test_a_persona_leaving_its_action_space_never_reaches_the_container():
    class RogueDriver(PersonaDriver):
        def act(self, view: PersonaView) -> PersonaTurn:
            return PersonaTurn(
                persona_id=view.persona.id, action=Action(type="/drop_database")
            )

    pop = population(persona("nurse", wake_on=[], allowed_actions=["/post_message"]))
    runner, posted = make_runner(pop)
    runner.environment._personas._explicit_driver = RogueDriver()
    runner._reset(seed=1)
    act(runner)
    assert posted == ["/post_message"]
    assert runner.environment.personas.transcript[0].blocked


def test_persona_turns_are_recorded_for_the_episode():
    pop = population(persona("nurse", wake_on=[], allowed_actions=["/post_message"]))
    runner, _ = make_runner(pop)
    runner.environment._personas._explicit_driver = ReplyDriver()
    runner._reset(seed=1)
    act(runner)
    assert [t.persona_id for t in runner.environment.personas.transcript] == ["nurse"]


def test_the_state_returned_reflects_what_the_cast_did():
    pop = population(persona("nurse", wake_on=[], allowed_actions=["/post_message"]))
    runner, _ = make_runner(pop)
    runner.environment._personas._explicit_driver = ReplyDriver()
    runner._reset(seed=1)
    assert act(runner)["posts"] == 2


def test_reset_clears_the_transcript_between_episodes():
    pop = population(persona("nurse", wake_on=[], allowed_actions=["/post_message"]))
    runner, _ = make_runner(pop)
    runner.environment._personas._explicit_driver = ReplyDriver()
    runner._reset(seed=1)
    act(runner)
    runner._reset(seed=2)
    assert runner.environment.personas.transcript == []


# --- what the worker hands the runner -------------------------------------


def test_the_worker_loads_an_enabled_cast_from_the_environment(tmp_path):
    from backend.app.worker.tasks import _load_personas

    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "config.yaml").write_text(
        "personas:\n"
        "  enabled: true\n"
        "  roster:\n"
        "    - id: nurse\n"
        "      name: Nurse\n"
        "      behavior:\n"
        "        allowed_actions: [/post_message]\n"
    )
    population = _load_personas(tmp_path)
    assert population is not None
    assert population.roster[0].profile.id == "nurse"


def test_the_worker_treats_a_switched_off_cast_as_no_cast(tmp_path):
    from backend.app.worker.tasks import _load_personas

    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "config.yaml").write_text(
        "personas:\n  enabled: false\n  roster:\n    - id: n\n      name: N\n"
    )
    assert _load_personas(tmp_path) is None


def test_the_worker_survives_an_environment_with_no_config(tmp_path):
    from backend.app.worker.tasks import _load_personas

    assert _load_personas(tmp_path) is None


def test_a_malformed_cast_degrades_the_run_rather_than_failing_it(tmp_path):
    """Every episode of a run must not die because one config key is misspelled."""
    from backend.app.worker.tasks import _load_personas

    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "config.yaml").write_text("personas:\n  enabeld: true\n")
    assert _load_personas(tmp_path) is None
