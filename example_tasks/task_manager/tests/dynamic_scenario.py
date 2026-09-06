"""Scenarios invented for the tests, sharing nothing with the bundled fixture.

The point of the engine is that a second Harbor task is a second `Scenario`, not
an edit to the world. A suite that could only test the engine through the
reassignment fixture would be testing the fixture; these exist so the engine is
exercised against tasks it has never seen.
"""

from __future__ import annotations

from task_sim.models import LatentDependency, LatentTask, ScenarioEvent, ScenarioRule
from task_sim.scenario import Scenario

#: A read releases a work item, which in turn unlocks a second one. Nothing here
#: resembles a handover.
DISCOVERY_SCENARIO = Scenario(
    events=(
        ScenarioEvent("audit_opened", "Someone looked at the security audit."),
        ScenarioEvent("finding_filed", "The audit produced a finding."),
        ScenarioEvent("triage_ready", "Both prerequisites are in place."),
    ),
    rules=(
        ScenarioRule("d10", "audit_opened", "observed", observed_ids=("TASK020",)),
        ScenarioRule(
            "d20", "finding_filed", "field_equals",
            table="tasks", row_id="TASK020", field="status", value="IN_PROGRESS",
            requires_activated=("audit_opened",),
        ),
        ScenarioRule(
            "d30", "triage_ready", "all_of",
            requires_activated=("audit_opened", "finding_filed"),
        ),
    ),
    latent_tasks=(
        LatentTask(
            task_id="TASK900", event_id="finding_filed",
            title="Remediate audit finding",
            description="Close the finding the audit produced.",
            creator_id="U006", assignee_id="U003", status="PENDING",
            project_id="P003", milestone_id=None, priority="URGENT",
            labels=("security", "audit"),
        ),
    ),
    latent_dependencies=(
        LatentDependency(
            dep_id="DEP900", event_id="finding_filed",
            task_id="TASK900", depends_on_task_id="TASK020",
        ),
    ),
    observation_only=("audit_opened", "triage_ready"),
)

#: Ordering expressed as a negative gate: the second event can only fire while
#: the first is still pending, so acting in the wrong order locks it out.
ORDERED_SCENARIO = Scenario(
    events=(
        ScenarioEvent("shipped", "The staging deploy went out."),
        ScenarioEvent("signed_off_early", "Sign-off happened before the deploy."),
    ),
    rules=(
        ScenarioRule(
            "o10", "shipped", "field_equals",
            table="tasks", row_id="TASK016", field="status", value="COMPLETED",
        ),
        ScenarioRule(
            "o20", "signed_off_early", "label_present",
            row_id="TASK017", labels=("signed-off",),
            requires_pending=("shipped",),
        ),
    ),
    observation_only=("shipped", "signed_off_early"),
)

#: A scenario with no events at all: a plain, static tracker.
EMPTY_SCENARIO = Scenario()

__all__ = ["DISCOVERY_SCENARIO", "EMPTY_SCENARIO", "ORDERED_SCENARIO"]
