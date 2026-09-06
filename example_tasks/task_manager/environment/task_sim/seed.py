"""Seed corpus and bundled scenario for the task-tracker world.

This module is the deterministic generator: it defines the rows written into
SQLite, which is the single source of truth for state. No parallel in-memory
state is built from these constants.

Everything the workspace starts with is here, including the two design tasks a
previous revision created from the Dockerfile after seeding. Building the
initial state in one place is what lets `seed_database` be the only path to a
workspace and lets two reseeds produce byte-identical snapshots.
"""

from __future__ import annotations

from task_sim.clock import ONE_DAY_MS, START_MS
from task_sim.models import ScenarioEvent, ScenarioRule
from task_sim.scenario import Scenario

# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

TRACKER_USERS = [
    ("U001", "Avery Chen",   "avery@example.local",  "admin",  "research",    "avery"),
    ("U002", "Morgan Patel", "morgan@example.local", "member", "product",     "morgan"),
    ("U003", "Riley Stone",  "riley@example.local",  "member", "engineering", "riley"),
    ("U004", "Jordan Kim",   "jordan@example.local", "member", "design",      "jordan"),
    ("U005", "Sam Wu",       "sam@example.local",    "member", "platform",    "sam"),
    ("U006", "Alex Rivera",  "alex@example.local",   "admin",  "engineering", "alex"),
]

# ---------------------------------------------------------------------------
# Projects  (project_id, name, description, owner_id, step, archived)
# ---------------------------------------------------------------------------

TRACKER_PROJECTS = [
    ("P001", "ML Platform v2",          "Next-gen ML training and serving infrastructure.",   "U001", 1, False),
    ("P002", "Agent Eval Framework",    "Deterministic evaluation framework for LLM agents.", "U002", 2, False),
    ("P003", "Infrastructure Overhaul", "Platform reliability and scalability improvements.", "U006", 3, False),
    ("P004", "Product Dashboard",       "User-facing analytics and reporting dashboard.",     "U002", 4, False),
]

# ---------------------------------------------------------------------------
# Milestones  (milestone_id, project_id, title, description, due_at_ms, step)
# ---------------------------------------------------------------------------

TRACKER_MILESTONES = [
    ("M001", "P001", "Alpha Release",    "First public alpha of ML Platform v2.",                  START_MS + 60 * ONE_DAY_MS, 5),
    ("M002", "P001", "Beta Release",     "Feature-complete beta with performance baseline.",       START_MS + 90 * ONE_DAY_MS, 6),
    ("M003", "P002", "v1.0 Launch",      "Production-ready eval framework release.",               START_MS + 45 * ONE_DAY_MS, 7),
    ("M004", "P003", "Phase 1 Complete", "Core infrastructure hardening -- already overdue.",      START_MS -  5 * ONE_DAY_MS, 8),
    ("M005", "P004", "Q1 Goals",         "All Q1 product deliverables shipped -- already overdue.", START_MS - 10 * ONE_DAY_MS, 9),
]

# ---------------------------------------------------------------------------
# Tasks
# (task_id, title, description, creator_id, assignee_id, status,
#  project_id, milestone_id, due_at_ms, priority, labels, step)
# ---------------------------------------------------------------------------

TRACKER_TASKS = [
    # -- Agent Eval Framework (P002) ----------------------------------------
    (
        "TASK001", "Draft benchmark plan",
        "Create a deterministic task matrix for provider comparison.",
        "U001", None, "PENDING",
        "P002", "M003", None, "HIGH", (), 1,
    ),
    (
        "TASK002", "Review tool schemas",
        "Confirm tool contracts are stable and versioned.",
        "U002", "U003", "IN_PROGRESS",
        "P002", "M003", None, "MEDIUM", (), 2,
    ),
    (
        "TASK006", "Write API documentation",
        "Document all REST endpoints in OpenAPI format.",
        "U002", "U004", "PENDING",
        "P002", "M003", None, "LOW", ("docs",), 6,
    ),
    (
        "TASK012", "Agent trajectory logger",
        "Log full ATIF-v1.7 trajectories to structured storage.",
        "U001", "U001", "COMPLETED",
        "P002", "M003", None, "HIGH", ("eval",), 12,
    ),
    (
        "TASK013", "Benchmark provider A",
        "Run full eval suite against provider A. Blocked until plan and logger are done.",
        "U001", "U003", "PENDING",
        "P002", "M003", None, "MEDIUM", ("eval", "benchmark"), 13,
    ),
    (
        "TASK014", "Benchmark provider B",
        "Run full eval suite against provider B. Blocked until plan and logger are done.",
        "U001", "U002", "PENDING",
        "P002", "M003", None, "MEDIUM", ("eval", "benchmark"), 14,
    ),
    (
        "TASK023", "Review tool schemas v2",
        "Second-pass schema review for agent tool contracts.",
        "U002", None, "PENDING",
        "P002", "M003", None, "LOW", ("docs",), 23,
    ),
    (
        "TASK029", "Update API contracts",
        "Align REST and gRPC contracts after schema changes.",
        "U002", "U003", "BLOCKED",
        "P002", "M003", None, "HIGH", ("backend", "api"), 29,
    ),
    # -- ML Platform v2 (P001) ----------------------------------------------
    (
        "TASK004", "Design database schema",
        "Model entity relationships for the ML platform.",
        "U001", "U003", "COMPLETED",
        "P001", "M001", None, "URGENT", ("backend", "database"), 4,
    ),
    (
        "TASK010", "ML model training pipeline",
        "Build distributed training pipeline with checkpointing.",
        "U001", "U003", "PENDING",
        "P001", "M001", None, "URGENT", ("ml", "backend"), 10,
    ),
    (
        "TASK015", "Data pipeline refactor",
        "Migrate ETL jobs from Airflow 1.x to Airflow 2.x.",
        "U001", "U005", "BLOCKED",
        "P001", "M002", None, "HIGH", ("backend", "data"), 15,
    ),
    (
        "TASK027", "Performance optimization",
        "Profile and reduce P99 inference latency by 30%.",
        "U001", "U005", "IN_PROGRESS",
        "P001", "M002", None, "HIGH", ("backend", "perf"), 27,
    ),
    # -- Infrastructure Overhaul (P003) -------------------------------------
    (
        "TASK003", "Set up CI pipeline",
        "Configure GitHub Actions for automated testing.",
        "U006", "U005", "COMPLETED",
        "P003", "M004", None, "HIGH", ("infra", "devops"), 3,
    ),
    (
        "TASK005", "Implement auth service",
        "Build JWT-based authentication with role management.",
        "U006", "U003", "IN_PROGRESS",
        "P003", "M004", START_MS - 3 * ONE_DAY_MS, "HIGH", ("backend", "security"), 5,
    ),
    (
        "TASK007", "Set up monitoring dashboards",
        "Create Grafana dashboards for service health. Blocked on auth service.",
        "U006", "U005", "BLOCKED",
        "P003", "M004", None, "MEDIUM", ("infra",), 7,
    ),
    (
        "TASK011", "Feature flag service",
        "Implement feature flag evaluation with targeting rules.",
        "U006", "U005", "IN_PROGRESS",
        "P003", None, None, "HIGH", ("backend", "platform"), 11,
    ),
    (
        "TASK016", "Deploy to staging",
        "Promote auth service and feature flags to staging environment.",
        "U006", "U005", "PENDING",
        "P003", "M004", START_MS - 2 * ONE_DAY_MS, "URGENT", ("infra", "devops"), 16,
    ),
    (
        "TASK017", "Load testing",
        "Run k6 load tests against staging with 10k concurrent users.",
        "U006", "U005", "PENDING",
        "P003", None, None, "MEDIUM", ("infra",), 17,
    ),
    (
        "TASK020", "Security audit",
        "Third-party pen test and code review for P003 services.",
        "U006", None, "PENDING",
        "P003", None, None, "URGENT", ("security",), 20,
    ),
    (
        "TASK021", "Notification service",
        "Push/email notification delivery pipeline.",
        "U006", "U005", "CANCELLED",
        "P003", None, None, "LOW", ("backend",), 21,
    ),
    (
        "TASK028", "Add rate limiting",
        "Implement per-user rate limiting on all API endpoints.",
        "U006", "U003", "PENDING",
        "P003", None, None, "MEDIUM", ("backend", "security"), 28,
    ),
    # -- Product Dashboard (P004) -------------------------------------------
    (
        "TASK008", "User research interviews",
        "Conduct 10 interviews with target users.",
        "U002", "U004", "COMPLETED",
        "P004", "M005", None, "MEDIUM", ("research", "ux"), 8,
    ),
    (
        "TASK009", "Create wireframes",
        "Design lo-fi and hi-fi wireframes for dashboard views.",
        "U004", "U004", "IN_PROGRESS",
        "P004", "M005", None, "HIGH", ("design", "ux"), 9,
    ),
    (
        "TASK018", "Product onboarding flow",
        "Implement step-by-step onboarding for new users.",
        "U002", "U003", "IN_PROGRESS",
        "P004", "M005", None, "HIGH", ("frontend", "ux"), 18,
    ),
    (
        "TASK019", "Analytics integration",
        "Integrate Mixpanel events into the dashboard backend.",
        "U002", "U002", "PENDING",
        "P004", None, None, "LOW", ("backend", "analytics"), 19,
    ),
    (
        "TASK022", "Mobile app setup",
        "Initialize React Native project with navigation and auth stubs.",
        "U004", None, "PENDING",
        "P004", None, None, "MEDIUM", ("mobile",), 22,
    ),
    (
        "TASK026", "Fix login redirect bug",
        "POST /login redirecting to 404 on first sign-in.",
        "U003", "U003", "COMPLETED",
        "P004", "M005", None, "HIGH", ("frontend", "bug"), 26,
    ),
    # Jordan's two design tasks. TASK031 is the only one of the five with no
    # milestone, which is what makes the instruction's branch a real decision
    # rather than a bulk update.
    (
        "TASK031", "Component design tokens",
        "Define and document the design token system for the dashboard component library.",
        "U004", "U004", "PENDING",
        "P004", None, None, "MEDIUM", ("design", "frontend"), 31,
    ),
    (
        "TASK032", "Brand refresh mockups",
        "High-fidelity mockups applying brand refresh guidelines to all dashboard views.",
        "U004", "U004", "PENDING",
        "P004", "M005", None, "MEDIUM", ("design",), 32,
    ),
    # -- Stale / closed tasks (distractors) ---------------------------------
    (
        "TASK024", "Draft benchmark plan (stale)",
        "Outdated copy of benchmark plan. Superseded by TASK001.",
        "U001", None, "ARCHIVED",
        "P002", None, None, "LOW", (), 24,
    ),
    (
        "TASK025", "Legacy data migration",
        "One-time migration of v1 data to new schema. Duplicate of existing effort.",
        "U001", None, "DUPLICATE",
        "P001", None, None, "LOW", ("data",), 25,
    ),
    # -- Cross-project / unassigned -----------------------------------------
    (
        "TASK030", "Quarterly review rollup",
        "Compile cross-team Q4 progress report for leadership.",
        "U001", None, "PENDING",
        None, None, None, "LOW", ("admin",), 30,
    ),
]

# ---------------------------------------------------------------------------
# Assignments  (assignment_id, task_id, user_id, assigned_by, step)
# ---------------------------------------------------------------------------

TRACKER_ASSIGNMENTS = [
    ("ASSIGN001", "TASK002", "U003", "U002",  3),
    ("ASSIGN002", "TASK003", "U005", "U006",  4),
    ("ASSIGN003", "TASK004", "U003", "U001",  5),
    ("ASSIGN004", "TASK005", "U003", "U006",  6),
    ("ASSIGN005", "TASK006", "U004", "U002",  7),
    ("ASSIGN006", "TASK007", "U005", "U006",  8),
    ("ASSIGN007", "TASK008", "U004", "U002",  9),
    ("ASSIGN008", "TASK009", "U004", "U004", 10),
    ("ASSIGN009", "TASK010", "U003", "U001", 11),
    ("ASSIGN010", "TASK011", "U005", "U006", 12),
    ("ASSIGN011", "TASK012", "U001", "U001", 13),
    ("ASSIGN012", "TASK013", "U003", "U001", 14),
    ("ASSIGN013", "TASK014", "U002", "U001", 15),
    ("ASSIGN014", "TASK015", "U005", "U001", 16),
    ("ASSIGN015", "TASK016", "U005", "U006", 17),
    ("ASSIGN016", "TASK017", "U005", "U006", 18),
    ("ASSIGN017", "TASK018", "U003", "U002", 19),
    ("ASSIGN018", "TASK019", "U002", "U002", 20),
    ("ASSIGN019", "TASK026", "U003", "U003", 27),
    ("ASSIGN020", "TASK027", "U005", "U001", 28),
    ("ASSIGN021", "TASK028", "U003", "U006", 29),
    ("ASSIGN022", "TASK029", "U003", "U002", 30),
    ("ASSIGN023", "TASK031", "U004", "U004", 31),
    ("ASSIGN024", "TASK032", "U004", "U004", 32),
]

# ---------------------------------------------------------------------------
# Dependencies  (dep_id, task_id, depends_on_task_id)
# task_id is blocked until depends_on_task_id is COMPLETED.
# ---------------------------------------------------------------------------

TRACKER_DEPENDENCIES = [
    ("DEP001", "TASK005", "TASK003"),   # auth service needs CI pipeline
    ("DEP002", "TASK007", "TASK005"),   # monitoring needs auth service
    ("DEP003", "TASK009", "TASK008"),   # wireframes need research interviews
    ("DEP004", "TASK013", "TASK001"),   # benchmark A needs plan
    ("DEP005", "TASK013", "TASK012"),   # benchmark A needs trajectory logger
    ("DEP006", "TASK014", "TASK001"),   # benchmark B needs plan
    ("DEP007", "TASK014", "TASK012"),   # benchmark B needs trajectory logger
    ("DEP008", "TASK016", "TASK005"),   # staging deploy needs auth service
    ("DEP009", "TASK016", "TASK011"),   # staging deploy needs feature flags
    ("DEP010", "TASK017", "TASK016"),   # load testing needs staging deploy
    ("DEP011", "TASK018", "TASK009"),   # onboarding flow needs wireframes
    ("DEP012", "TASK028", "TASK005"),   # rate limiting needs auth service
    ("DEP013", "TASK029", "TASK002"),   # API contracts need schema review
]


# ---------------------------------------------------------------------------
# Bundled scenario
# ---------------------------------------------------------------------------
#
# The reassignment fixture is a *static* workspace: nothing latent is released
# and no message arrives mid-episode. Its events therefore record what the
# agent reached rather than what the world answered with, which is what the
# milestones layer grades and what the RL loop shapes on. `tracker` supports
# latent rows in full; `tests/test_scenario_engine.py` exercises that half with
# scenarios invented in the test file, so the engine is covered without this
# fixture having to pretend it needs them.

#: The five tasks Jordan Kim holds at T0, and where each one has to land.
DEPARTING_USER = "U004"
MILESTONE_SUCCESSOR = "U002"
UNMILESTONED_SUCCESSOR = "U003"
TRIAGE_LABEL = "needs-triage"

#: Ordered as the instruction's branch sorts them: four carry a milestone, one
#: does not. A contract naming these is naming the task, not the world.
MILESTONE_TASKS = ("TASK006", "TASK008", "TASK009", "TASK032")
UNMILESTONED_TASKS = ("TASK031",)
REASSIGNED_TASKS = tuple(sorted(MILESTONE_TASKS + UNMILESTONED_TASKS))

REASSIGNMENT_EVENTS = (
    ScenarioEvent("roster_listed", "The agent enumerated every task Jordan Kim holds."),
    ScenarioEvent("branch_inspected", "The agent read the milestone field it has to branch on."),
    ScenarioEvent("task006_reassigned", "TASK006 (milestone M003) moved to Morgan Patel."),
    ScenarioEvent("task008_reassigned", "TASK008 (milestone M005) moved to Morgan Patel."),
    ScenarioEvent("task009_reassigned", "TASK009 (milestone M005) moved to Morgan Patel."),
    ScenarioEvent("task032_reassigned", "TASK032 (milestone M005) moved to Morgan Patel."),
    ScenarioEvent("task031_reassigned", "TASK031 (no milestone) moved to Riley Stone."),
    ScenarioEvent("task031_triaged", "TASK031 gained needs-triage without losing its own labels."),
    ScenarioEvent("handover_reported", "The agent stated which tasks it modified."),
    ScenarioEvent("reassignment_complete", "Every branch of the instruction was carried out."),
)

REASSIGNMENT_RULES = (
    # Seeing all five ids in one result is what "found the whole set" means.
    # Any route reaches it: the assignee filter, an unfiltered list, or the
    # project rollup -- the rule listens for the ids, not for one call.
    ScenarioRule(
        "r010_roster", "roster_listed", "observed",
        observed_ids=REASSIGNED_TASKS,
    ),
    # The branch is on milestone presence, and TASK031 is the only task where
    # reading it changes the answer.
    ScenarioRule(
        "r020_branch", "branch_inspected", "observed",
        observed_ids=("TASK031",), tools=("get_task", "get_project", "list_tasks"),
    ),
    ScenarioRule(
        "r030_task006", "task006_reassigned", "field_equals",
        table="tasks", row_id="TASK006", field="assignee_id", value=MILESTONE_SUCCESSOR,
    ),
    ScenarioRule(
        "r031_task008", "task008_reassigned", "field_equals",
        table="tasks", row_id="TASK008", field="assignee_id", value=MILESTONE_SUCCESSOR,
    ),
    ScenarioRule(
        "r032_task009", "task009_reassigned", "field_equals",
        table="tasks", row_id="TASK009", field="assignee_id", value=MILESTONE_SUCCESSOR,
    ),
    ScenarioRule(
        "r033_task032", "task032_reassigned", "field_equals",
        table="tasks", row_id="TASK032", field="assignee_id", value=MILESTONE_SUCCESSOR,
    ),
    ScenarioRule(
        "r034_task031", "task031_reassigned", "field_equals",
        table="tasks", row_id="TASK031", field="assignee_id", value=UNMILESTONED_SUCCESSOR,
    ),
    # Appending, not replacing: the rule requires the seeded labels alongside
    # the new one, so a label list that overwrote them does not activate it.
    ScenarioRule(
        "r035_triage", "task031_triaged", "label_present",
        row_id="TASK031", labels=("design", "frontend", TRIAGE_LABEL),
    ),
    ScenarioRule(
        "r040_report", "handover_reported", "tool_called",
        tools=("submit_handover_report",),
    ),
    # A pure closure: it resolves in the same call that completed its last
    # prerequisite rather than waiting for an unrelated action to nudge it.
    ScenarioRule(
        "r050_complete", "reassignment_complete", "all_of",
        requires_activated=(
            "roster_listed", "branch_inspected",
            "task006_reassigned", "task008_reassigned", "task009_reassigned",
            "task032_reassigned", "task031_reassigned", "task031_triaged",
            "handover_reported",
        ),
    ),
)

REASSIGNMENT_SCENARIO = Scenario(
    events=REASSIGNMENT_EVENTS,
    rules=REASSIGNMENT_RULES,
    observation_only=tuple(event.event_id for event in REASSIGNMENT_EVENTS),
)
