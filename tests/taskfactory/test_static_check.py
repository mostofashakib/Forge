"""The static check rejects drafts that cannot run, before any LLM review.

It checks a draft against the environment's real tools and state, the step
floor for its difficulty, and the shape each check kind needs.
"""
from __future__ import annotations

from forge.taskfactory.profile import EnvironmentProfile, ToolInfo
from forge.taskfactory.schemas import Check, GoldenStep, ReflectionPoint, TaskDraft, TaskPage, TaskSeed
from forge.taskfactory.slots import Slot
from forge.taskfactory.static_check import static_problems

STATE_PROFILE = EnvironmentProfile(
    env_name="mail",
    env_type="premade:gmail",
    family="state",
    tools=(ToolInfo(name="/archive_email"), ToolInfo(name="/emails/{email_id}/star"), ToolInfo(name="/delete_email")),
    state_sample={"emails": [{"id": "e001", "folder": "inbox"}], "contacts": [], "inbox_unread": 0},
)
CLI_PROFILE = EnvironmentProfile(env_name="shell", env_type="cli", family="cli")
BROWSER_PROFILE = EnvironmentProfile(env_name="web", env_type="browser", family="browser")


def _slot(difficulty: int) -> Slot:
    return Slot(index=0, category="triage", difficulty=difficulty)


def _state_draft(**overrides) -> TaskDraft:
    fields = dict(
        slot=0,
        title="Archive the invoice",
        objective="Archive the invoice email from finance.",
        seed=TaskSeed(records={"emails": [{"id": "e010", "folder": "inbox", "subject": "Invoice"}]}),
        golden=[GoldenStep(tool="/archive_email", args={"email_id": "e010"})],
        checks=[Check(kind="record_exists", collection="emails", match={"id": "e010"}, expect={"folder": "archive"})],
    )
    fields.update(overrides)
    return TaskDraft(**fields)


def test_a_well_formed_state_draft_has_no_problems():
    assert static_problems(_state_draft(), _slot(1), STATE_PROFILE) == []


def test_templated_endpoints_match_filled_paths():
    draft = _state_draft(golden=[GoldenStep(tool="/emails/e010/star")])

    assert static_problems(draft, _slot(1), STATE_PROFILE) == []


def test_a_tool_the_environment_lacks_is_rejected():
    draft = _state_draft(golden=[GoldenStep(tool="/forward_email", args={})])

    problems = static_problems(draft, _slot(1), STATE_PROFILE)
    assert any("/forward_email" in p for p in problems)


def test_a_control_plane_tool_is_rejected_even_if_listed():
    profile = EnvironmentProfile(
        env_name="mail", env_type="general", family="state",
        tools=(ToolInfo(name="/forge/restore-state"),), state_sample={},
    )
    draft = _state_draft(
        seed=TaskSeed(),
        golden=[GoldenStep(tool="/forge/restore-state")],
        checks=[Check(kind="value", path="x", op="==", value=1)],
    )

    assert any("control" in p for p in static_problems(draft, _slot(1), profile))


def test_seeding_a_collection_the_environment_lacks_is_rejected():
    draft = _state_draft(seed=TaskSeed(records={"calendar": [{"id": "c1"}]}))

    assert any("calendar" in p for p in static_problems(draft, _slot(1), STATE_PROFILE))


def test_checking_a_collection_the_state_lacks_is_rejected():
    draft = _state_draft(checks=[Check(kind="record_absent", collection="archive", match={"id": "e010"})])

    assert any("archive" in p for p in static_problems(draft, _slot(1), STATE_PROFILE))


def test_a_problem_repeated_by_several_checks_is_listed_once():
    draft = _state_draft(checks=[
        Check(kind="record_absent", collection="archive", match={"id": "e010"}),
        Check(kind="record_absent", collection="archive", match={"id": "e011"}),
    ])

    problems = static_problems(draft, _slot(1), STATE_PROFILE)
    assert len([p for p in problems if "'archive'" in p]) == 1


def test_a_draft_without_an_outcome_check_is_rejected():
    # Only "which actions ran" checks: nothing proves the world changed.
    draft = _state_draft(checks=[Check(kind="actions_called", tools=["/archive_email"])])

    assert any("outcome" in p for p in static_problems(draft, _slot(1), STATE_PROFILE))


def test_a_check_missing_a_field_its_kind_needs_is_rejected():
    draft = _state_draft(checks=[Check(kind="record_exists", match={"id": "e010"})])

    assert any("collection" in p for p in static_problems(draft, _slot(1), STATE_PROFILE))


def test_a_check_kind_from_another_family_is_rejected():
    draft = _state_draft(checks=[
        Check(kind="record_exists", collection="emails", match={"id": "e010"}, expect={"folder": "archive"}),
        Check(kind="shell", command="test -f /tmp/x"),
    ])

    assert any("shell" in p for p in static_problems(draft, _slot(1), STATE_PROFILE))


def _long_golden(n: int) -> list[GoldenStep]:
    return [GoldenStep(tool="/archive_email", args={"email_id": f"e{i:03d}"}) for i in range(n)]


def _reflections(*steps: int) -> list[ReflectionPoint]:
    return [ReflectionPoint(step=s, kind="verify", note="confirm before going on") for s in steps]


def test_a_hard_task_below_the_thirty_step_floor_is_rejected():
    draft = _state_draft(golden=_long_golden(29), reflection_points=_reflections(3, 10, 20))

    assert any("30" in p for p in static_problems(draft, _slot(4), STATE_PROFILE))


def test_a_hard_task_at_the_floor_with_reflections_passes():
    draft = _state_draft(golden=_long_golden(30), reflection_points=_reflections(3, 10, 20))

    assert static_problems(draft, _slot(5), STATE_PROFILE) == []


def test_a_hard_task_needs_three_reflection_points():
    draft = _state_draft(golden=_long_golden(30), reflection_points=_reflections(3, 10))

    assert any("reflection" in p for p in static_problems(draft, _slot(4), STATE_PROFILE))


def test_a_reflection_point_outside_the_golden_solution_is_rejected():
    draft = _state_draft(golden=_long_golden(30), reflection_points=_reflections(3, 10, 30))

    assert any("reflection" in p for p in static_problems(draft, _slot(4), STATE_PROFILE))


def test_an_easy_task_longer_than_its_ceiling_is_rejected():
    draft = _state_draft(golden=_long_golden(6))

    assert any("1 to 5" in p for p in static_problems(draft, _slot(1), STATE_PROFILE))


def test_a_cli_draft_needs_shell_steps_and_shell_checks():
    good = TaskDraft(
        slot=0, title="Count lines", objective="Write the line count of /data/a.txt to /tmp/n.",
        seed=TaskSeed(setup=["mkdir -p /data", "printf 'a\\nb\\n' > /data/a.txt"]),
        golden=[GoldenStep(tool="shell", args={"command": "wc -l < /data/a.txt > /tmp/n"})],
        checks=[Check(kind="shell", command="grep -qx 2 /tmp/n")],
    )
    bad = good.model_copy(update={"golden": [GoldenStep(tool="/archive_email")]})

    assert static_problems(good, _slot(1), CLI_PROFILE) == []
    assert static_problems(bad, _slot(1), CLI_PROFILE) != []


def _browser_draft(**overrides) -> TaskDraft:
    fields = dict(
        slot=0, title="Submit the form", objective="Submit the contact form with your name.",
        seed=TaskSeed(
            pages=[TaskPage(path="/form", html="<form><input id='name'><button id='go'>Go</button></form><p id='status'></p>")],
            start_path="/form",
        ),
        golden=[
            GoldenStep(tool="fill", args={"selector": "#name", "value": "Sam"}),
            GoldenStep(tool="click", args={"selector": "#go"}),
        ],
        checks=[Check(kind="dom", selector="#status", prop="text", op="equals", value="Submitted")],
    )
    fields.update(overrides)
    return TaskDraft(**fields)


def test_a_well_formed_browser_draft_has_no_problems():
    assert static_problems(_browser_draft(), _slot(1), BROWSER_PROFILE) == []


def test_a_browser_start_page_that_does_not_exist_is_rejected():
    draft = _browser_draft(seed=TaskSeed(pages=[TaskPage(path="/form", html="<p></p>")], start_path="/home"))

    assert any("/home" in p for p in static_problems(draft, _slot(1), BROWSER_PROFILE))


def test_a_browser_goto_to_an_unknown_page_is_rejected():
    draft = _browser_draft(golden=[GoldenStep(tool="goto", args={"path": "/missing"})])

    assert any("/missing" in p for p in static_problems(draft, _slot(1), BROWSER_PROFILE))


def test_a_browser_action_missing_its_selector_is_rejected():
    draft = _browser_draft(golden=[GoldenStep(tool="click", args={})])

    assert any("selector" in p for p in static_problems(draft, _slot(1), BROWSER_PROFILE))
