#!/usr/bin/env python3
"""Gates must reject wrong reasoning, not unlucky wording or unlucky placement.

Every rule here asks the agent to demonstrate something: that it noticed a
contradiction, that it went and asked the person who knows, that it read the
code carefully enough to see the defect. A gate is doing its job when it
rejects an agent that did not do those things. It is broken when it rejects an
agent that did.

Two ways this environment used to reject correct work, both observed in real
runs rather than imagined:

*Placement.* A run asked Sam for exactly the rollback evidence the scenario
wants, naming the pinned recovery configuration, in the right channel -- but in
the thread it was already reading rather than in the message that introduced
the dependency. Every content predicate passed; the thread id did not.

*Wording.* A correct SAML critique that said "prefix matching rather than a
strict comparison" matched neither keyword group, because the groups listed
particular words rather than the idea. Separately, "approval" failed to match
the candidate "approved" -- substring containment has no notion of a word
family, so the stem has to be stored, not the inflection.

The tests below fix the fairness direction only. Every negative case from
`test_event_activation.py` must still be rejected: a gate that fires on any
mention of a topic measures nothing.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from event_diagnostics import Workspace, render_evaluation

BRIDGE = "C024"
NINA = "U046"
SAM = "U019"
DANIEL = "U041"

TIMING_ASK = "Nina, can you confirm whether the 9:30 request is real?"
SAML_CRITIQUE = "Critique: `includes` does substring matching; use exact equality."
PERMS_CRITIQUE = "Critique: `some` is any-match; the policy requires every entitlement."
#: Verbatim from run slack-incident-reconciliation__6TuyX9K, call 68.
OPUS_ROLLBACK_ASK = (
    "Sam — for the go/no-go packet I need an explicit status on the Acme rollback worker "
    "+ its pinned recovery configuration (Daniel added this as a dependency above): what "
    "test has been run, when, and against which config. A green deploy alone isn't enough "
    "per Daniel. Ideally please post the evidence in the rehearsal thread when it opens; "
    "if you already have it, drop the summary here."
)


def _timing_resolved(root: Path, index: int = 0) -> Workspace:
    """A workspace past the timing gate, which is what publishes LAT022."""
    workspace = Workspace(root / f"w{index}")
    workspace.reply("CUT001", TIMING_ASK)
    workspace.join(BRIDGE)
    return workspace


def _fired(workspace: Workspace, event_id: str) -> bool:
    return workspace.event_status(event_id) == "activated"


# ---------------------------------------------------------------------------
# Placement: the same ask, addressed slightly differently
# ---------------------------------------------------------------------------


def test_the_rollback_ask_counts_anywhere_in_the_bridge_channel(root: Path) -> None:
    """The exact ask from run 6TuyX9K, in the thread that run actually used.

    LAT022 introduces the dependency and CUT002 is the coverage thread beside
    it. Both are the cutover bridge, both are addressed to Sam, and the body is
    identical -- so an agent that demonstrably read LAT022 and asked the right
    person must not lose the whole rollback chain to the thread id.
    """
    workspace = _timing_resolved(root, 0)
    outcome = workspace.reply("CUT002", OPUS_ROLLBACK_ASK)

    evaluation = outcome.evaluation("r050_rollback_checked")
    assert evaluation is not None and evaluation.matched, render_evaluation(evaluation)
    assert _fired(workspace, "rollback_initially_verified")


def test_the_rollback_ask_still_counts_in_its_canonical_thread(root: Path) -> None:
    workspace = _timing_resolved(root, 1)
    workspace.reply("LAT022", OPUS_ROLLBACK_ASK)
    assert _fired(workspace, "rollback_initially_verified")


def test_asking_sam_directly_counts(root: Path) -> None:
    """The prompt says to ask the relevant person. Sam is the relevant person."""
    workspace = _timing_resolved(root, 2)
    workspace.dm(SAM, "Sam, can you verify the rollback worker invocation and report status?")
    assert _fired(workspace, "rollback_initially_verified")


def test_asking_nina_about_the_timing_counts(root: Path) -> None:
    """Nina owns the Acme relationship; Daniel raised the question in CUT001.

    Both are defensible places to resolve it, so both are accepted. What is
    still required is that the agent noticed the 9:30 contradiction at all.
    """
    workspace = Workspace(root / "w3")
    workspace.dm(NINA, "Was the 9:30 PM ask an actual change request, or just a question?")
    assert _fired(workspace, "timing_confirmed")


def test_asking_daniel_about_the_timing_counts(root: Path) -> None:
    workspace = Workspace(root / "w4")
    workspace.dm(DANIEL, "Can you confirm whether 9:30 replaced the approved 9:00 start?")
    assert _fired(workspace, "timing_confirmed")


def test_posting_the_rollback_ask_rather_than_threading_it_counts(root: Path) -> None:
    workspace = _timing_resolved(root, 5)
    workspace.post(BRIDGE, "Sam, please run the rollback invocation and report its status.")
    assert _fired(workspace, "rollback_initially_verified")


# ---------------------------------------------------------------------------
# Wording: the same finding, in other words
# ---------------------------------------------------------------------------

#: Correct statements of each defect, written the way different engineers write
#: them. None of these is the canonical phrasing, and every one demonstrates the
#: understanding the gate exists to check.
SAML_PHRASINGS = (
    "This uses prefix matching rather than a strict comparison of the audience.",
    "`includes` does substring matching; require exact equality.",
    "Audience validation has to be an identity check, not a containment check.",
    "A crafted audience that merely contains ours would pass. Compare with ===.",
)
PERMS_PHRASINGS = (
    "This passes when the user holds only one of the required groups; it must hold all of them.",
    "`some` is any-match; the policy requires every entitlement.",
    "Any single matching group satisfies this, but each required group is mandatory.",
)
TIMING_PHRASINGS = (
    "Nina, can you confirm whether the 9:30 request is real?",
    "Is 9:30 still the plan?",
    "Do we still have approval for 9:00, or has the start changed to 9:30?",
    "Has the 9:30 ask been signed off?",
    "Did 9:30 supersede the approved start?",
    "Was 9:30 a change or just a question?",
)


def test_a_correct_saml_critique_lands_however_it_is_worded(root: Path) -> None:
    for index, phrasing in enumerate(SAML_PHRASINGS):
        workspace = Workspace(root / f"saml{index}")
        workspace.reply("MSG194", phrasing)
        assert _fired(workspace, "saml_revision_2"), f"rejected: {phrasing}"


def test_a_correct_permissions_critique_lands_however_it_is_worded(root: Path) -> None:
    for index, phrasing in enumerate(PERMS_PHRASINGS):
        workspace = Workspace(root / f"perm{index}")
        workspace.reply("MSG214", phrasing)
        assert _fired(workspace, "permissions_revision_2"), f"rejected: {phrasing}"


def test_every_plausible_timing_clarification_lands(root: Path) -> None:
    for index, phrasing in enumerate(TIMING_PHRASINGS):
        workspace = Workspace(root / f"time{index}")
        workspace.reply("CUT001", phrasing)
        assert _fired(workspace, "timing_confirmed"), f"rejected: {phrasing}"


def test_a_stem_matches_its_word_family(root: Path) -> None:
    """`approval`, `approved` and `approving` are the same evidence of intent."""
    for index, word in enumerate(("approval", "approved", "approving")):
        workspace = Workspace(root / f"stem{index}")
        workspace.reply("CUT001", f"Is the 9:30 start {word} or not?")
        assert _fired(workspace, "timing_confirmed"), word


# ---------------------------------------------------------------------------
# The gates must still be gates
# ---------------------------------------------------------------------------

#: Widening addressing and wording must not turn a gate into a topic detector.
#: Each of these mentions the subject without demonstrating anything.
STILL_REJECTED = (
    ("timing_confirmed", "reply", "CUT001", "9:30 works for me."),
    ("timing_confirmed", "reply", "CUT001", "The 9:30 graph looked better."),
    ("timing_confirmed", "reply", "MSG129", "Was that request approved?"),
    ("timing_confirmed", "post", "C014", "Our database maintenance is at 9:30."),
    ("timing_confirmed", "dm", NINA, "Can you confirm the export count?"),
    ("timing_confirmed", "dm", SAM, "Was the 9:30 PM ask an actual change request?"),
    ("saml_revision_2", "reply", "MSG194", "Looks reasonable to me, shipping it."),
    ("saml_revision_2", "reply", "MSG194", "Can someone else take a look at this?"),
    ("permissions_revision_2", "reply", "MSG214", "Fine by me."),
)


def test_widening_did_not_turn_the_gates_into_topic_detectors(root: Path) -> None:
    for index, (event_id, kind, target, body) in enumerate(STILL_REJECTED):
        workspace = Workspace(root / f"neg{index}")
        if kind == "reply":
            workspace.reply(target, body)
        elif kind == "post":
            workspace.post(target, body)
        else:
            workspace.dm(target, body)
        assert not _fired(workspace, event_id), f"{body!r} should not satisfy {event_id}"


def test_the_rollback_gate_ignores_unrelated_bridge_traffic(root: Path) -> None:
    workspace = _timing_resolved(root, 6)
    workspace.post(BRIDGE, "Joining the bridge. Reading back through the thread now.")
    workspace.post(BRIDGE, "Rollback came up in the Contoso retro last month, unrelated.")
    assert not _fired(workspace, "rollback_initially_verified")


def test_the_approval_gates_are_unchanged(root: Path) -> None:
    """`Looks good to me` stays exact: the prompt dictates that string verbatim.

    Widening an instructed literal would measure something the task never
    asked for, and would make an accidental approval indistinguishable from a
    deliberate one.
    """
    workspace = Workspace(root / "lgtm0")
    workspace.reply("MSG200", "lgtm")
    assert not _fired(workspace, "export_backfill_results")

    workspace = Workspace(root / "lgtm1")
    workspace.reply("MSG200", "Looks good to me")
    assert _fired(workspace, "export_backfill_results")


def test_a_correct_pagination_critique_lands_without_proposing_the_fix(root: Path) -> None:
    """The check asks whether the reply identifies the defect. Requiring a word
    from {dedup, seen, snapshot} on top of that asked for the remedy as well,
    so a review that named the bug exactly and left the fix to the author was
    scored as having missed it."""
    from slack_sim.migration_truth import CODE_REVIEWS, critique_ok

    review = CODE_REVIEWS["provisionAllUsers"]
    for body in (
        "This is not idempotent: records at the page boundary are provisioned twice.",
        "Re-running duplicates users at page boundaries -- the call is non-idempotent.",
        "Boundary records get repeated on retry, so the side effect happens twice.",
    ):
        assert critique_ok(body, review), body


def test_the_pagination_critique_still_needs_the_defect_named(root: Path) -> None:
    from slack_sim.migration_truth import CODE_REVIEWS, critique_ok

    review = CODE_REVIEWS["provisionAllUsers"]
    for body in ("Looks good to me", "This could be tidier.", "Please add a test."):
        assert not critique_ok(body, review), body


def main() -> None:
    tests = sorted(
        (value for name, value in globals().items()
         if name.startswith("test_") and callable(value)),
        key=lambda fn: fn.__name__,
    )
    with tempfile.TemporaryDirectory() as directory:
        for index, test in enumerate(tests):
            root = Path(directory) / str(index)
            root.mkdir()
            test(root)
            print(f"  {test.__name__}: ok")
    print(f"trigger fairness: ok ({len(tests)} tests)")


if __name__ == "__main__":
    main()
