"""Container episodes graded by the environment's own compiled conditions.

Until now the held-out evaluation asked an LLM, every step, how close the state
looked to the objective — and that score decided both the reward and whether the
episode succeeded. The compiled success conditions the environment was built
with were loaded, counted for difficulty, and thrown away. This grades with
them.
"""
from __future__ import annotations

import pytest

from forge.benchmark._eval import structural_verdict
from forge.benchmark.compiled_tasks import task_from_template
from forge.extraction.schemas import SuccessCondition, TaskTemplate


def _template(name="finish", conditions=None, failures=None) -> TaskTemplate:
    return TaskTemplate(
        name=name,
        description="complete the task",
        success_conditions=conditions if conditions is not None else [
            SuccessCondition(type="state_check", expression="done == True"),
        ],
        failure_conditions=failures or [],
    )


class _Step:
    def __init__(self, state_after, action=None):
        self.state_after = state_after
        self.action = action or {}


class _Result:
    def __init__(self, steps, termination_reason="max_steps"):
        self.steps = steps
        self.termination_reason = termination_reason
        self.events: list[dict] = []


# ---------------------------------------------------------------------------
# The compiled ground truth survives into the benchmark task
# ---------------------------------------------------------------------------

def test_a_benchmark_task_carries_its_compiled_template():
    task = task_from_template(_template(), "env_a")

    assert task.template is not None
    assert task.template.success_conditions[0].expression == "done == True"


def test_difficulty_is_still_derived_from_the_conditions():
    """The template must be carried in addition to, not instead of, difficulty."""
    template = _template(conditions=[
        SuccessCondition(type="state_check", expression="a == 1"),
        SuccessCondition(type="state_check", expression="b == 2"),
    ])

    assert task_from_template(template, "env_a").difficulty == 2


# ---------------------------------------------------------------------------
# Structural grading
# ---------------------------------------------------------------------------

def test_a_satisfied_state_condition_passes_regardless_of_termination_reason():
    """The verifier decides, not the LLM-driven termination monitor."""
    task = task_from_template(_template(), "env_a")
    result = _Result([_Step({"done": True})], termination_reason="max_steps")

    assert structural_verdict(result, task, "full_layered_partial") is True


def test_an_unsatisfied_state_condition_fails_even_when_the_run_said_success():
    """The LLM calling an episode 'success' cannot override the state check."""
    task = task_from_template(_template(), "env_a")
    result = _Result([_Step({"done": False})], termination_reason="success")

    assert structural_verdict(result, task, "full_layered_partial") is False


def test_a_forbidden_action_fails_the_episode_despite_the_right_final_state():
    """Reaching the goal via a forbidden route is not success."""
    task = task_from_template(_template(
        failures=[SuccessCondition(type="policy_check", expression="delete_all")],
    ), "env_a")
    result = _Result([_Step({"done": True}, action={"type": "delete_all"})])

    assert structural_verdict(result, task, "full_layered_partial") is False


def test_a_permitted_route_to_the_same_final_state_passes():
    """False-positive guard: the forbidden-action check must not reject everything."""
    task = task_from_template(_template(
        failures=[SuccessCondition(type="policy_check", expression="delete_all")],
    ), "env_a")
    result = _Result([_Step({"done": True}, action={"type": "complete_task"})])

    assert structural_verdict(result, task, "full_layered_partial") is True


def test_a_forbidden_event_fails_the_episode():
    task = task_from_template(_template(
        failures=[SuccessCondition(type="negative_check", expression="data_deleted")],
    ), "env_a")
    result = _Result([_Step({"done": True})])
    result.events = [{"type": "data_deleted"}]

    assert structural_verdict(result, task, "full_layered_partial") is False


def test_an_episode_with_no_steps_cannot_satisfy_its_conditions():
    task = task_from_template(_template(), "env_a")

    assert structural_verdict(_Result([]), task, "full_layered_partial") is False


def test_a_task_without_compiled_conditions_yields_no_structural_verdict():
    """Absent ground truth is unknown, not success — the caller must decide."""
    task = task_from_template(_template(conditions=[]), "env_a")
    result = _Result([_Step({"done": True})])

    assert structural_verdict(result, task, "full_layered_partial") is None


def test_a_task_with_no_template_yields_no_structural_verdict():
    from forge.benchmark.task_suite import Task

    task = Task(
        name="t", domain="d", objective="o",
        success_fn=lambda _s: True, difficulty=1,
    )

    assert structural_verdict(_Result([_Step({})]), task, "full_layered_partial") is None


def test_grading_the_same_episode_twice_gives_the_same_verdict():
    """The point of structural grading: it is a function, not an opinion."""
    task = task_from_template(_template(), "env_a")
    result = _Result([_Step({"done": True})])

    first = structural_verdict(result, task, "full_layered_partial")
    second = structural_verdict(result, task, "full_layered_partial")

    assert first == second is True


def test_binary_final_state_preset_grades_on_the_state_check_alone():
    task = task_from_template(_template(), "env_a")
    result = _Result([_Step({"done": True})], termination_reason="dead_end")

    assert structural_verdict(result, task, "binary_final_state") is True


def test_a_judge_only_preset_yields_no_structural_verdict():
    """False-positive guard: judge_only has no structural checks to run."""
    task = task_from_template(_template(), "env_a")
    result = _Result([_Step({"done": True})])

    assert structural_verdict(result, task, "judge_only") is None


# ---------------------------------------------------------------------------
# Verdict resolution: structural first, LLM only when there is no ground truth
# ---------------------------------------------------------------------------

def _task():
    return task_from_template(_template(), "env_a")


def test_structural_ground_truth_overrides_the_llm_termination_reason():
    from forge.benchmark._eval import resolve_verdict

    result = _Result([_Step({"done": False})], termination_reason="success")
    verdict = resolve_verdict(result, _task(), "full_layered_partial")

    assert verdict.passed is False
    assert verdict.indeterminate is False
    assert verdict.source == "structural"


def test_without_ground_truth_the_verdict_falls_back_to_the_run_outcome():
    from forge.benchmark._eval import resolve_verdict
    from forge.benchmark.task_suite import Task

    bare = Task(name="t", domain="d", objective="o", success_fn=lambda _s: True, difficulty=1)
    result = _Result([_Step({})], termination_reason="success")
    verdict = resolve_verdict(result, bare, "full_layered_partial")

    assert verdict.passed is True
    assert verdict.source == "termination_reason"


def test_a_fallback_verdict_is_marked_as_llm_derived():
    """The fallback reads an LLM-driven signal, and the record must say so."""
    from forge.benchmark._eval import resolve_verdict
    from forge.benchmark.task_suite import Task

    bare = Task(name="t", domain="d", objective="o", success_fn=lambda _s: True, difficulty=1)
    verdict = resolve_verdict(_Result([_Step({})]), bare, "full_layered_partial")

    assert verdict.llm_derived is True


def test_a_structural_verdict_is_not_marked_as_llm_derived():
    from forge.benchmark._eval import resolve_verdict

    verdict = resolve_verdict(_Result([_Step({"done": True})]), _task(), "full_layered_partial")

    assert verdict.llm_derived is False


def test_a_disagreeing_verdict_jury_returns_an_indeterminate_verdict():
    from forge.benchmark._eval import resolve_verdict
    from forge.validation.jury import Jury
    from forge.validation.member import MemberVerdict

    class _Member:
        def __init__(self, mid, fam, passed):
            self.member_id, self.family, self._p = mid, fam, passed

        def evaluate(self, subject):
            return MemberVerdict(member_id=self.member_id, family=self.family, passed=self._p)

    jury = Jury(
        members=[_Member("gpt-4o", "gpt", True), _Member("gemini-2.0", "gemini", False)],
        generator_families=("claude",),
    )
    verdict = resolve_verdict(
        _Result([_Step({"done": True})]), _task(), "full_layered_partial", jury=jury
    )

    assert verdict.indeterminate is True
    assert verdict.passed is False


def test_an_agreeing_verdict_jury_decides_the_episode():
    from forge.benchmark._eval import resolve_verdict
    from forge.validation.jury import Jury
    from forge.validation.member import MemberVerdict

    class _Member:
        def __init__(self, mid, fam, passed):
            self.member_id, self.family, self._p = mid, fam, passed

        def evaluate(self, subject):
            return MemberVerdict(member_id=self.member_id, family=self.family, passed=self._p)

    jury = Jury(
        members=[_Member("gpt-4o", "gpt", True), _Member("gemini-2.0", "gemini", True)],
        generator_families=("claude",),
    )
    verdict = resolve_verdict(
        _Result([_Step({"done": True})]), _task(), "full_layered_partial", jury=jury
    )

    assert verdict.indeterminate is False
    assert verdict.passed is True
