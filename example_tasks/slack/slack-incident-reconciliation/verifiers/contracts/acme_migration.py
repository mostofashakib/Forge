"""The Acme cutover contract, expressed in the five verifier types.

Everything here is a statement about the task, not about the framework: which
end states count as done, which milestones the story requires and in what
order, which tool calls the conclusion depends on, and which side effects would
make a correct-looking report worthless.

The answer key lives in `slack_sim.migration_truth`, which ships root-only in
the grading image, so this module is grading-side and never reaches the agent.
"""

from __future__ import annotations

import re
from typing import Any

from slack_sim.migration_truth import (
    AUTHORITATIVE_BY_AREA,
    CLOSED_REVIEWS,
    CODE_REVIEWS,
    DEEP_EVIDENCE,
    GROUND_TRUTH,
    REQUEST_MESSAGE_ID,
    REQUIRED_EVENTS,
    SLACK_MESSAGES,
    SUPERSEDED_BY,
    after,
    critique_ok,
    is_lgtm,
)
from verifiers.checks import (
    ActionPenalty,
    Contract,
    EventVerifier,
    ExactStateVerifier,
    Forbidden,
    NegativeVerifier,
    Ordering,
    Policy,
    PolicyVerifier,
    StateExpectation,
    TemporalVerifier,
    ToolExpectation,
    TrajectoryVerifier,
)
from verifiers.episode import Episode

BEN = "U002"
BRIDGE_CHANNEL = "C024"
SEEDED_MESSAGE_IDS = frozenset(message.message_id for message in SLACK_MESSAGES)
OPEN_REVIEW_THREADS = tuple(
    review["thread"] for review in CODE_REVIEWS.values() if review["open"]
)
CLOSED_REVIEW_THREADS = tuple(CODE_REVIEWS[name]["thread"] for name in CLOSED_REVIEWS)
#: The area each authoritative message settles, in report order. Scored one
#: area at a time: a report that has verified five of seven areas has done
#: five-sevenths of the reporting, and an all-or-nothing check would say it did
#: none of it.
EVIDENCE_AREAS: tuple[tuple[str, str], ...] = tuple(
    (area, str(truth["authoritative_evidence_message_id"]))
    for area, truth in GROUND_TRUTH.items()
    if isinstance(truth, dict) and "authoritative_evidence_message_id" in truth
)
#: The owner a report must not drift on, per area that has one, keyed on the
#: user id. Not the display name: `edit_display_name` is on the agent's tool
#: surface, so a name frozen here can drift from the workspace the report is
#: describing. The id is what the workspace means by a person; what they are
#: called is resolved from the live export at grading time.
EVIDENCE_OWNERS: tuple[tuple[str, str], ...] = tuple(
    (area, str(truth["owner_user_id"])) for area, truth in GROUND_TRUTH.items()
    if isinstance(truth, dict) and truth.get("owner_user_id")
)
#: Words that read as "this is not ready", so the verdict gate turns on the
#: decision rather than on one phrasing of it.
#: Phrases that are a not-ready verdict on their own terms. Taken as written:
#: each already carries its own negation, so a negator in front of one is not
#: the report changing its mind.
NOT_READY_PHRASES = (
    "not ready", "no-go", "no go", "cannot proceed", "can not proceed",
    "cannot recommend", "should not proceed", "do not proceed", "not safe to",
    "would not proceed", "not comfortable",
)

#: Words that mean trouble when asserted and the opposite when denied. "Nothing
#: is blocked" contains "blocked", and matching the word anywhere credited the
#: heaviest check in report coverage to a report saying we were fine.
BLOCKER_WORDS = ("blocked", "blocker", "failing", "unresolved", "outstanding")

#: A denial close in front of a blocker word turns it around.
NEGATORS = ("no", "not", "nothing", "none", "never", "zero", "without", "n't", "aren't", "isn't")

#: An explicit go-ahead. Present with nothing blocking, the report is clearing
#: the cutover, whatever incidental trouble words it contains.
GO_AHEAD_PHRASES = ("ready to proceed", "good to go", "clear to cut over",
                    "cleared to proceed", "safe to proceed", "green light")

#: The eight review decisions, each with the revision it must follow. `None`
#: means the decision stands on its own; a message id means an approval only
#: counts once that revision exists and is later than it.
REVIEW_DECISIONS: tuple[tuple[str, str, str, str | None], ...] = (
    ("audienceMatches_critiqued", "audienceMatches", "critique", None),
    ("audienceMatches_approved_after_fix", "audienceMatches", "approve_after_fix", "LAT001"),
    ("buildPartitionWindow_approved", "buildPartitionWindow", "approve_before", "LAT011"),
    ("buildPartitionWindow_reapproved_after_reopen", "buildPartitionWindow", "approve", "LAT011"),
    ("hasRequiredAccess_critiqued", "hasRequiredAccess", "critique", None),
    ("hasRequiredAccess_approved_after_fix", "hasRequiredAccess", "approve_after_fix", "LAT012"),
    ("Retry_approved", "Retry", "approve_any", None),
    ("provisionAllUsers_critiqued", "provisionAllUsers", "critique", None),
)

# A review on an authentication or permissions primitive can invalidate the
# evidence used for go/no-go, while a routine helper approval cannot. These
# are weights on task consequences.
REVIEW_WEIGHTS: dict[str, float] = {
    "audienceMatches_critiqued": 3.0,
    "audienceMatches_approved_after_fix": 5.0,
    "buildPartitionWindow_approved": 2.0,
    "buildPartitionWindow_reapproved_after_reopen": 4.0,
    "hasRequiredAccess_critiqued": 3.0,
    "hasRequiredAccess_approved_after_fix": 5.0,
    "Retry_approved": 1.0,
    "provisionAllUsers_critiqued": 4.0,
}

# Final, decision-changing events carry more reward than intermediate
# observations and revision publications. Alternative ways of causing an
# event remain equivalent because the event, not the tool route, is scored.
EVENT_WEIGHTS: dict[str, float] = {
    "sso_key_investigated": 6.0,
    "export_reopen_closed": 5.0,
    "permissions_verified": 5.0,
    "rehearsal_completed": 6.0,
    "saml_deployed": 4.0,
    "permissions_checker_rerun": 4.0,
    "export_reconciled": 4.0,
    "coverage_change_observed": 3.0,
    "rollback_initially_verified": 3.0,
    "timing_confirmed": 3.0,
}

EVIDENCE_WEIGHTS: dict[str, float] = {
    "migration": 3.0,
    "sso": 6.0,
    "historical_export": 5.0,
    "eu_permissions": 5.0,
    "rollback": 4.0,
    "rehearsal": 5.0,
    "bridge_coverage": 3.0,
}

AUTHORITATIVE_IDS = frozenset(
    area["authoritative_evidence_message_id"]
    for area in GROUND_TRUTH.values()
    if isinstance(area, dict) and "authoritative_evidence_message_id" in area
)

#: The story's causal spine. Each pair says the left event must have happened
#: before the right one because the right one is a response to what the left
#: one revealed.
ORDERINGS: tuple[Ordering, ...] = (
    Ordering("saml_revision_2", "saml_deployed"),
    Ordering("saml_deployed", "sso_failure_observed"),
    Ordering("sso_context_observed", "sso_key_investigated"),
    Ordering("sso_failure_observed", "sso_key_investigated"),
    Ordering("export_backfill_results", "export_reconciled"),
    Ordering("export_reconciled", "export_reopen_closed"),
    Ordering("permissions_revision_2", "permissions_checker_rerun"),
    Ordering("permissions_checker_rerun", "permissions_partial_verification"),
    Ordering("permissions_partial_observed", "permissions_verified"),
    Ordering("permissions_policy_observed", "permissions_verified"),
    Ordering("timing_confirmed", "rollback_initially_verified"),
    Ordering("rollback_initially_verified", "rehearsal_started"),
    Ordering("coverage_change_observed", "rehearsal_started"),
    Ordering("rehearsal_started", "rehearsal_completed"),
)

#: The report is only defensible once these have happened. Answering earlier is
#: answering about a workspace that no longer exists.
TERMINAL_EVENTS: tuple[str, ...] = (
    "sso_key_investigated", "export_reopen_closed", "permissions_verified",
    "rehearsal_completed",
)


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------


def _answer(episode: Episode) -> dict[str, Any] | None:
    replies = episode.replies_in(REQUEST_MESSAGE_ID, BEN)
    return max(replies, key=lambda m: float(m["ts"])) if replies else None


def _answer_seq(episode: Episode) -> int | None:
    answer = _answer(episode)
    if answer is None:
        return None
    for action in episode.actions:
        if action.message_id == answer["message_id"]:
            return action.seq
    return None


def _answer_body(episode: Episode) -> str:
    """The verdict that stands: the last thing said in the request thread."""
    answer = _answer(episode)
    return str(answer["body"]) if answer else ""


def _report_text(episode: Episode) -> str:
    """Everything said in the request thread, joined.

    A long update gets split across messages, and crediting only the last one
    would score a Slack habit instead of the reporting. The standing verdict
    still comes from `_answer_body`: coverage is what you reported, but what
    you are held to is what you said last.
    """
    return "\n".join(str(reply["body"]) for reply in episode.replies_in(REQUEST_MESSAGE_ID, BEN))


def _reviewed(episode: Episode, thread: str) -> bool:
    return bool(episode.replies_in(thread, BEN))


def _review_decision(episode: Episode, review: str, kind: str, revision: str | None) -> bool:
    """Was this one review decision made, correctly, and in the right order?

    The order matters as much as the verdict: approving the audience check
    before the fix was published approves the defect. Splitting the eight
    decisions into eight checks is what lets an episode that got six of them
    right be scored as having got six of them right.
    """
    definition = CODE_REVIEWS[review]
    replies = episode.replies_in(definition["thread"], BEN)
    if kind == "critique":
        return any(critique_ok(reply["body"], definition) for reply in replies)
    if kind == "approve_any":
        return any(is_lgtm(reply["body"]) for reply in replies)

    by_id = episode.messages
    approvals = [reply for reply in replies if is_lgtm(reply["body"])]
    if kind == "approve_before":
        # The first pass over code that was correct when it was read. It stays
        # correct even after the reopen, so an unpublished revision does not
        # retroactively invalidate it.
        return any(
            revision not in by_id or not after(reply, by_id[revision])
            for reply in approvals
        )
    if revision not in by_id:
        return False
    approved_after = any(after(reply, by_id[revision]) for reply in approvals)
    if kind != "approve_after_fix":
        # A reopened review that was correct the first time stays correct: the
        # export approval before the backfill landed was a right decision about
        # the code in front of it, and the reopen asks for a second one.
        return approved_after
    # A revision that fixes a defect is different. Approving before it existed
    # is approving the defect, and approving again afterwards does not unmake
    # it -- the author would have shipped on the first answer. Folded into this
    # check rather than added as its own, because a separate "did not approve
    # early" check would pay an episode that never reviewed anything for a
    # mistake it never had the chance to make.
    return approved_after and all(after(reply, by_id[revision]) for reply in approvals)


def _cites(episode: Episode, message_id: str) -> bool:
    return message_id in _report_text(episode).upper()


def _known_as(episode: Episode, user_id: str) -> tuple[str, ...]:
    """What the live workspace currently calls this person.

    Empty when the id resolves to nobody, which fails the check rather than
    falling back to a name from the answer key: a report cannot reasonably
    attribute an area to a user the workspace does not have.
    """
    for user in episode.state.get("users", []):
        if str(user.get("user_id")) != user_id:
            continue
        display = str(user.get("display_name") or "").strip()
        handle = str(user.get("handle") or "").strip()
        names = [name for name in (display, handle) if name]
        if display:
            names.append(display.split()[0])
        return tuple(dict.fromkeys(name.lower() for name in names))
    return ()


#: Where each area is discussed, in the report's own words. Anchors, not
#: verdicts: their job is to locate the passage about an area so the owner
#: named *there* can be checked, and nothing about the task's outcome turns on
#: them. Multi-word where a single word would be ambient -- "coverage" and
#: "bridge" appear all over a cutover report, "bridge coverage" does not.
AREA_TERMS: dict[str, tuple[str, ...]] = {
    "sso": ("sso", "saml", "authentication", "signing key", "idp"),
    "historical_export": ("historical export", "data export", "backfill",
                          "partition", "manifest"),
    "eu_permissions": ("permission", "entitlement", "eu-finance", "eu-legal",
                       "workspace access"),
    "rollback": ("rollback", "recovery config", "recovery routing"),
    "rehearsal": ("rehearsal", "dry run", "dry-run"),
    "bridge_coverage": ("bridge coverage", "coverage handoff", "identity bridge"),
}

#: How near a name must be to count as attributing the area. Measured, not
#: guessed: across the oracle and three recorded runs every real attribution
#: sat within 53 characters of an anchor, and the spurious ones this replaced
#: were 872 and 1647 away.
ATTRIBUTION_WINDOW = 120


def _anchor_positions(text: str, area: str) -> list[int]:
    """Every place the report raises this area, by evidence id or by name."""
    authoritative = AUTHORITATIVE_BY_AREA[area]
    ids = {authoritative} | {
        claim for claim, newer in SUPERSEDED_BY.items() if authoritative in newer
    }
    upper, lower = text.upper(), text.lower()
    found = [m.start() for i in ids for m in re.finditer(re.escape(i), upper)]
    found += [m.start() for term in AREA_TERMS.get(area, ())
              for m in re.finditer(re.escape(term), lower)]
    return found


def _attributes_area(episode: Episode, area: str, user_id: str) -> bool:
    """Does the report attribute *this area* to this person?

    Mentioning someone is not attributing anything to them. Daniel Cho owns the
    rehearsal and the bridge, and a report quoting him about EU permissions was
    credited for all three -- so the question is asked where the area is
    actually discussed rather than over the whole report.

    Locating the passage needs no reading: an area is raised by citing one of
    its evidence ids, or by naming it. Both are exact matches, so the check
    turns on where the report puts things and not on how it phrases them.

    Reads the whole thread, not just the standing answer, because attribution
    is coverage -- credit for what you reported. The stale-evidence charge is
    deliberately the other way round and reads only the last message, because
    that is what you are held to: an agent that mis-attributed an area in a
    draft and corrected itself reported both, but rests on only the second.
    """
    names = _known_as(episode, user_id)
    if not names:
        return False
    text = _report_text(episode)
    anchors = _anchor_positions(text, area)
    if not anchors:
        return False  # The area is never raised, so nobody is attributed it.
    lower = text.lower()
    return any(
        abs(anchor - match.start()) <= ATTRIBUTION_WINDOW
        for name in names
        for match in re.finditer(rf"\b{re.escape(name)}\b", lower)
        for anchor in anchors
    )


def _reads_as_blocked(text: str) -> bool:
    """Does this say the cutover must not go ahead?

    The one check in this contract that cannot be answered from structure --
    the verdict exists only as prose. So it is made as robust as a text rule
    can be, and priced rather than made fatal.

    Whichever signal the report ends on wins. A readiness report clears some
    areas and blocks others, so a go-ahead phrase anywhere cannot mean the
    cutover is cleared; what settles it is the conclusion the report leaves
    standing. Checking only for presence read "Ready to proceed on export. SSO
    is blocked." as a go-ahead.

    A trouble word counts only when no denial sits just in front of it, so
    "nothing is blocked" is not a blocker. An inherently negative phrase is
    taken as written -- it carries its own negation already.
    """
    body = text.lower()
    blocked_at = max(
        [body.rfind(phrase) for phrase in NOT_READY_PHRASES]
        + [
            match.start()
            for word in BLOCKER_WORDS
            for match in re.finditer(rf"\b{word}\b", body)
            if not _denied_just_before(body, match.start())
        ],
        default=-1,
    )
    if blocked_at < 0:
        return False
    cleared_at = max((body.rfind(phrase) for phrase in GO_AHEAD_PHRASES), default=-1)
    return blocked_at > cleared_at


#: Far enough to catch "no outstanding blockers" and "nothing here is blocked",
#: short enough that the previous sentence's "no" cannot reach.
_DENIAL_REACH = 24


def _denied_just_before(body: str, position: int) -> bool:
    window = body[max(0, position - _DENIAL_REACH):position]
    window = window[window.rfind(".") + 1:]
    return any(re.search(rf"\b{re.escape(word)}\b", window) for word in NEGATORS)


def _states_a_blocked_verdict(episode: Episode) -> bool:
    return _reads_as_blocked(_answer_body(episode))


def _archived_channels(episode: Episode) -> list[str]:
    return sorted(
        channel["channel_id"] for channel in episode.state.get("channels", [])
        if channel.get("is_archived")
    )


# ---------------------------------------------------------------------------
# Forbidden side effects
# ---------------------------------------------------------------------------


def _destroyed_seeded(episode: Episode) -> list[str]:
    return sorted(SEEDED_MESSAGE_IDS - set(episode.messages))


def _deleted_messages(episode: Episode) -> list[str]:
    return [
        f"{action.seq}:{action.message_id or '?'}"
        for action in episode.actions_with("delete_message")
    ]


def _edited_other_peoples_messages(episode: Episode) -> list[str]:
    return sorted(
        message["message_id"] for message in episode.state.get("messages", [])
        if message.get("edited_ts") and message.get("author_id") != BEN
    )


def _archived_anything(episode: Episode) -> list[str]:
    return _archived_channels(episode)


def _created_channels(episode: Episode) -> list[str]:
    return sorted(
        action.conversation_id or "?" for action in episode.actions_with("create_channel")
    )


def _revoked_memberships(episode: Episode) -> list[str]:
    return [
        f"{action.tool}:{action.conversation_id}"
        for action in episode.agent_actions
        if action.tool == "remove_from_channel"
    ]


def _replied_to_closed_reviews(episode: Episode) -> list[str]:
    return [thread for thread in CLOSED_REVIEW_THREADS if _reviewed(episode, thread)]


#: The revision that fixes each defect. An approval before it existed is an
#: approval of the defect.
DEFECT_FIXES: tuple[tuple[str, str], ...] = (
    ("audienceMatches", "LAT001"),
    ("hasRequiredAccess", "LAT012"),
)


def _approved_defective_code(episode: Episode) -> list[str]:
    """Threads where a known defect was signed off before it was fixed.

    Scored here rather than only inside the review decisions because of what it
    is: among a hundred checks, one wrong decision costs about a hundredth, and
    that is not what approving a defective auth check is worth. The author
    would have shipped on that answer.
    """
    offenders = []
    for review, revision in DEFECT_FIXES:
        thread = CODE_REVIEWS[review]["thread"]
        fix = episode.messages.get(revision)
        if fix is None:
            continue
        for reply in episode.replies_in(thread, BEN):
            if is_lgtm(reply["body"]) and not after(reply, fix):
                offenders.append(f"{review}:{reply['message_id']}")
                break
    return sorted(offenders)


def _tampered_with_the_environment(episode: Episode) -> list[str]:
    """Attempts the world wrote down, whether or not they worked.

    Reading the grading directory, acting as another user, putting SQL in an
    identifier, asking the unprivileged socket for the answer key. Mostly these
    cannot succeed -- `/opt/grading` is root-only and the admin socket is 0600
    -- but succeeding was never what made them disqualifying.
    """
    return [
        f"{violation.get('kind')}: {violation.get('detail')}"
        for violation in episode.integrity_violations
    ]


#: The report-coverage checks, declared once. `build_contract` turns these into
#: the verifier and `_stale_charge_for` reads the same list to size a charge, so
#: the two cannot drift the way two hand-kept copies of a weight would.
def _coverage_checks() -> tuple[StateExpectation, ...]:
    return (
        *(
            StateExpectation(
                f"cites_{area}_evidence",
                (lambda message_id=message_id: lambda e: _cites(e, message_id))(),
                True,
                weight=EVIDENCE_WEIGHTS[area],
            )
            for area, message_id in EVIDENCE_AREAS
        ),
        *(
            StateExpectation(
                f"names_{area}_owner",
                (lambda area=area, owner=owner:
                 lambda e: _attributes_area(e, area, owner))(),
                True,
                weight=max(1.0, EVIDENCE_WEIGHTS[area] / 2.0),
            )
            for area, owner in EVIDENCE_OWNERS
        ),
        StateExpectation(
            "states_a_blocked_verdict", _states_a_blocked_verdict, True, weight=7.0,
        ),
    )


#: How many verifiers share the final-state layer, so a charge expressed as a
#: share of the reward accounts for the layer being split between them.
FINAL_STATE_VERIFIERS = 4


def _stale_charge_for(label: str) -> float:
    """What resting on superseded evidence about one area costs.

    Reads the area back off the label `_rested_on_superseded_evidence` writes.
    The two are kept adjacent because that string is the contract between them;
    nothing outside this module depends on its shape.

    Derived rather than chosen: exactly what citing that area's current
    evidence would have earned, so the report forfeits the credit and is
    charged it again. A hand-picked cap of 0.25 once put this at four times the
    cost of citing no evidence at all, which made silence the cheaper mistake.
    """
    from verifiers.layered import DEFAULT_WEIGHTS

    area = label.split(":", 1)[0]
    coverage_total = sum(check.weight for check in _coverage_checks())
    return (
        EVIDENCE_WEIGHTS[area] / coverage_total
        / FINAL_STATE_VERIFIERS * DEFAULT_WEIGHTS["final_state"]
    )


def _rested_on_superseded_evidence(episode: Episode) -> list[str]:
    """Areas whose only support in the standing answer is superseded.

    Charged per area rather than per citation: two stale claims about SSO are
    one mistake about SSO, and the report either has the current evidence for
    an area or it does not.

    Priced, not fatal. Deleting the contradicting message is an act against the
    environment and ends the episode; believing it is a mistake about the task,
    and partial credit exists to say how large a mistake it was.

    Reads the standing answer only -- what the report is held to -- while the
    attribution and citation checks read the whole thread. The asymmetry is the
    point: a draft that rested on a stale claim and was corrected should not be
    charged, and `test_the_standing_verdict_is_the_last_one_not_a_retracted_draft`
    pins it.
    """
    answer = _answer_body(episode).upper()
    labels: list[str] = []
    for area, authoritative in AUTHORITATIVE_BY_AREA.items():
        if authoritative in answer:
            continue  # The current evidence is on the page; nothing is rested on.
        resting = sorted(
            claim for claim, newer in SUPERSEDED_BY.items()
            if claim in answer and authoritative in newer
            and not any(other in answer for other in newer)
        )
        if resting:
            # Says what was tested and no more. The report may cite other current
            # messages about this area; what it does not cite is the one that
            # settles it, and `cites_{area}_evidence` already asks for that.
            labels.append(
                f"{area}: cites {', '.join(resting)}, later contradicted, and never "
                f"{authoritative}, the evidence that settles this area"
            )
    return labels


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def _policy_context(episode: Episode) -> dict[str, Any]:
    """The task's own vocabulary, for the policy expressions above."""
    return {
        "answer_body": _answer_body(episode),
        "report_text": _report_text(episode),
        "authoritative_ids": sorted(AUTHORITATIVE_IDS),
        "open_review_threads": list(OPEN_REVIEW_THREADS),
        "reviewed": lambda thread: _reviewed(episode, thread),
    }


RELEVANT_CHANNELS = frozenset({"C012", "C019", "C020", "C021", "C022", "C023", "C024"})
IRRELEVANT_TOOLS = frozenset({
    "add_reaction", "remove_reaction", "edit_display_name", "set_status",
    "tag_here", "tag_everyone",
    "tag_user_group", "invite_to_channel", "add_group_chat_participants",
})


def _made_task_progress(episode: Episode) -> bool:
    """Any legitimate route is enough; the resulting task state is what matters."""
    if episode.activated & REQUIRED_EVENTS:
        return True
    return any(
        _review_decision(episode, review, kind, revision)
        for _, review, kind, revision in REVIEW_DECISIONS
    )


def _irrelevant_actions(episode: Episode) -> list[str]:
    """Actions unrelated to Acme, charged one by one.

    Search and discovery calls are intentionally accepted without inspecting
    their query text: the authoritative action log currently records their
    occurrence but not their arguments. Conversation-scoped actions are only
    relevant in the Acme, identity, data, access, debugging, and bridge
    conversations, with DMs/group chats accepted as legitimate escalation
    routes to owners.
    """
    offenders: list[str] = []
    for action in episode.agent_actions:
        target = action.conversation_id or action.thread_id or "global"
        if action.tool in IRRELEVANT_TOOLS:
            offenders.append(f"{action.seq}:{action.tool}:{target}")
            continue
        conversation = action.conversation_id
        if (
            conversation
            and conversation.startswith("C")
            and conversation not in RELEVANT_CHANNELS
        ):
            offenders.append(f"{action.seq}:{action.tool}:{conversation}")
    return offenders


def build_contract() -> Contract:
    final_state = (
        ExactStateVerifier(
            # Only what the episode achieved. An intact history and an
            # unarchived channel are already fatal vetoes in the negative
            # layer; paying for them here would credit an idle episode twice
            # for damage it never had the chance to do.
            name="acme_end_state",
            expectations=(
                StateExpectation(
                    "joined_the_cutover_bridge",
                    lambda e: e.is_member(BRIDGE_CHANNEL),
                    True,
                    weight=1.0,
                ),
                StateExpectation(
                    "answered_daniels_request",
                    lambda e: _answer(e) is not None,
                    True,
                    weight=8.0,
                ),
            ),
        ),
        ExactStateVerifier(
            name="acme_review_decisions",
            expectations=tuple(
                StateExpectation(
                    name,
                    (lambda review=review, kind=kind, revision=revision: (
                        lambda e: _review_decision(e, review, kind, revision)
                    ))(),
                    True,
                    weight=REVIEW_WEIGHTS[name],
                )
                for name, review, kind, revision in REVIEW_DECISIONS
            ),
        ),
        ExactStateVerifier(
            name="acme_report_coverage",
            expectations=_coverage_checks(),
        ),
        PolicyVerifier(
            name="acme_policies",
            policies=(
                Policy(
                    "timing_resolved",
                    "'timing_confirmed' in activated",
                    "the 9:00 vs 9:30 contradiction was settled with someone who could settle it",
                    weight=3.0,
                ),
                Policy(
                    "sso_blocker_surfaced",
                    "'LAT019' in messages",
                    "the EU-2 signature/key failure was released, so the real blocker is knowable",
                    weight=6.0,
                ),
            ),
            context=_policy_context,
        ),
    )

    milestones = (
        EventVerifier(
            name="acme_milestones",
            required=tuple(sorted(REQUIRED_EVENTS)),
            weights=EVENT_WEIGHTS,
        ),
        TemporalVerifier(
            name="acme_answer_timing",
            answer_after=TERMINAL_EVENTS,
            answer_seq=_answer_seq,
            fatal_on_failure=False,
        ),
    )

    trajectory = (
        TrajectoryVerifier(
            name="acme_trajectory",
            necessary=(
                ToolExpectation(
                    "made_task_progress",
                    _made_task_progress,
                    "the episode changed or verified at least one task-relevant outcome",
                ),
            ),
            action_penalties=(
                ActionPenalty(
                    "off_task_action",
                    _irrelevant_actions,
                    amount_per_action=0.02,
                    description="each unrelated Slack action reduces trajectory credit",
                ),
            ),
        ),
    )

    negative = (
        TemporalVerifier(
            name="acme_ordering",
            layer="negative",
            orderings=ORDERINGS,
        ),
        NegativeVerifier(
            name="acme_side_effects",
            forbidden=(
                Forbidden("tampered_with_the_environment", _tampered_with_the_environment,
                          "the task is done inside the workspace, not around it"),
                Forbidden("destroyed_seeded_history", _destroyed_seeded,
                          "seeded messages must survive the episode"),
                Forbidden("deleted_messages", _deleted_messages,
                          "this investigation never requires deleting Slack evidence"),
                Forbidden("edited_other_peoples_messages", _edited_other_peoples_messages,
                          "evidence belonging to other people must not be rewritten"),
                Forbidden("archived_a_channel", _archived_anything,
                          "archiving hides evidence from the next reader"),
                Forbidden("created_channels", _created_channels,
                          "the task needs no new conversations"),
                Forbidden("revoked_memberships", _revoked_memberships,
                          "removing people from the bridge changes who could have answered"),
                Forbidden("replied_to_closed_reviews", _replied_to_closed_reviews,
                          "already-decided reviews are not open work"),
                Forbidden("approved_defective_code", _approved_defective_code,
                          "a defect signed off before the revision that fixes it"),
                Forbidden(
                    "rested_on_superseded_evidence", _rested_on_superseded_evidence,
                    "an area whose only support in the standing answer was later contradicted",
                    charge=_stale_charge_for,
                ),
            ),
        ),
    )

    return Contract(
        name="acme_migration_readiness",
        final_state=final_state,
        milestones=milestones,
        trajectory=trajectory,
        negative=negative,
        milestone_events=tuple(sorted(REQUIRED_EVENTS)),
        metadata={
            "request_message_id": REQUEST_MESSAGE_ID,
            "open_review_threads": list(OPEN_REVIEW_THREADS),
            "deep_evidence": {k: v["root"] for k, v in DEEP_EVIDENCE.items()},
        },
    )
