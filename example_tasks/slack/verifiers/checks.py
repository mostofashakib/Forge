"""The five verifier types every contract is assembled from.

Each answers one question and reports its own checks, so a composed verdict can
always be read back to the assertion that produced it:

    ExactStateVerifier   did specific state fields end up with specific values
    PolicyVerifier       does an expression over the end state hold
    EventVerifier        did the required events appear at all
    TemporalVerifier     did they appear in an order the story permits
    NegativeVerifier     did anything forbidden happen
All five read world-side evidence and are deterministic: the same episode
grades the same way every time, on any machine, with no model in the loop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from verifiers.episode import Episode
from verifiers.results import CheckOutcome, Penalty, VerifierResult


class Verifier(Protocol):
    """Anything that can grade an episode on one layer."""

    name: str
    layer: str

    def verify(self, episode: Episode) -> VerifierResult: ...


def _score(checks: Sequence[CheckOutcome]) -> float:
    if not checks:
        return 1.0
    total = sum(max(0.0, check.weight) for check in checks)
    if total == 0.0:
        return 0.0
    earned = sum(max(0.0, check.weight) for check in checks if check.passed)
    return earned / total


# ---------------------------------------------------------------------------
# Final state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StateExpectation:
    """One field of the end state, and what it must contain.

    `select` pulls the value out of the episode rather than a path string: a
    callable is inspectable, testable and cannot silently mean nothing when a
    schema changes, which a mistyped dotted path can.
    """

    name: str
    select: Callable[[Episode], Any]
    expected: Any
    #: Compare with `==` by default; `contains` for "the answer is in there".
    comparison: str = "equals"
    #: Material requirements are worth more than bookkeeping. A task contract
    #: owns these weights because only the task knows which outcomes are on its
    #: critical path.
    weight: float = 1.0

    def evaluate(self, episode: Episode) -> CheckOutcome:
        try:
            actual = self.select(episode)
        except Exception as error:  # noqa: BLE001 - a broken selector is a failed check
            return CheckOutcome(
                self.name, False, {"error": f"{type(error).__name__}: {error}"}, self.weight
            )
        if self.comparison == "contains":
            passed = self.expected in (actual or [])
        elif self.comparison == "not_contains":
            passed = self.expected not in (actual or [])
        else:
            passed = actual == self.expected
        detail: dict[str, Any] = {"expected": self.expected, "comparison": self.comparison}
        detail["actual"] = sorted(actual) if isinstance(actual, set) else actual
        return CheckOutcome(self.name, passed, detail, self.weight)


@dataclass(frozen=True, slots=True)
class ExactStateVerifier:
    """Specific state field values."""

    expectations: tuple[StateExpectation, ...]
    name: str = "exact_state"
    layer: str = "final_state"
    fatal_on_failure: bool = False

    def verify(self, episode: Episode) -> VerifierResult:
        checks = tuple(expectation.evaluate(episode) for expectation in self.expectations)
        return VerifierResult(
            verifier=type(self).__name__, layer=self.layer, score=_score(checks),
            checks=checks, fatal_on_failure=self.fatal_on_failure,
        )


@dataclass(frozen=True, slots=True)
class Policy:
    name: str
    expression: str
    description: str = ""
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class PolicyVerifier:
    """Python expressions evaluated against the current state.

    Expressions are configuration written by the task author, never anything
    the agent can influence, and they are evaluated with builtins stripped so a
    typo fails as a check rather than reaching the filesystem.
    """

    policies: tuple[Policy, ...]
    name: str = "policy"
    layer: str = "final_state"
    fatal_on_failure: bool = False
    #: Extra names an expression may use, derived from the episode. Lets a
    #: contract write `all(reviewed(t) for t in open_reviews)` instead of
    #: unpacking rows inside a string, where a typo is invisible.
    context: Callable[[Episode], dict[str, Any]] | None = None

    def verify(self, episode: Episode) -> VerifierResult:
        namespace = episode.namespace()
        if self.context is not None:
            namespace.update(self.context(episode))
        checks = []
        for policy in self.policies:
            try:
                # One mapping, used as globals. Names inside a comprehension
                # resolve against globals rather than the locals argument, so
                # splitting them would make `all(f(x) for x in xs)` raise
                # NameError for every name the contract supplied.
                value = eval(policy.expression, {"__builtins__": {}, **namespace})  # noqa: S307
                passed = bool(value)
                detail: dict[str, Any] = {"expression": policy.expression}
            except Exception as error:  # noqa: BLE001
                passed = False
                detail = {
                    "expression": policy.expression,
                    "error": f"{type(error).__name__}: {error}",
                }
            if policy.description:
                detail["description"] = policy.description
            checks.append(CheckOutcome(policy.name, passed, detail, policy.weight))
        checks_tuple = tuple(checks)
        return VerifierResult(
            verifier=type(self).__name__, layer=self.layer, score=_score(checks_tuple),
            checks=checks_tuple, fatal_on_failure=self.fatal_on_failure,
        )


# ---------------------------------------------------------------------------
# Events and ordering
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventVerifier:
    """Required events appeared in the trajectory.

    Partial credit is the fraction reached, because an episode that completed
    nine of twenty milestones did more than one that completed none, and a
    reward signal that cannot say so teaches nothing.
    """

    required: tuple[str, ...]
    weights: dict[str, float] = field(default_factory=dict)
    name: str = "milestones"
    layer: str = "milestones"
    fatal_on_failure: bool = False

    def verify(self, episode: Episode) -> VerifierResult:
        activated = episode.activated
        checks = tuple(
            CheckOutcome(
                f"event:{event}",
                event in activated,
                {"status": episode.events.get(event, "absent")},
                self.weights.get(event, 1.0),
            )
            for event in self.required
        )
        return VerifierResult(
            verifier=type(self).__name__, layer=self.layer, score=_score(checks),
            checks=checks, fatal_on_failure=self.fatal_on_failure,
        )


@dataclass(frozen=True, slots=True)
class Ordering:
    """`earlier` must have activated before `later`, if `later` happened at all."""

    earlier: str
    later: str
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class TemporalVerifier:
    """Event ordering and timing constraints.

    Order is what separates a conclusion that was reached from one that was
    guessed: reporting the outcome before the evidence that produced it is a
    different episode from reporting it after, even when the end state matches.
    An ordering over two events that never happened is vacuous, and scored as
    such rather than counted as a pass.
    """

    orderings: tuple[Ordering, ...] = ()
    #: name -> (event that must precede the agent's answer)
    answer_after: tuple[str, ...] = ()
    answer_seq: Callable[[Episode], int | None] | None = None
    name: str = "ordering"
    layer: str = "milestones"
    fatal_on_failure: bool = True

    def verify(self, episode: Episode) -> VerifierResult:
        checks: list[CheckOutcome] = []
        for ordering in self.orderings:
            earlier = episode.activation_ts(ordering.earlier)
            later = episode.activation_ts(ordering.later)
            if later is None:
                continue  # nothing to order yet; EventVerifier already counts it missing
            passed = earlier is not None and float(earlier) <= float(later)
            checks.append(
                CheckOutcome(
                    f"order:{ordering.earlier}<{ordering.later}",
                    passed,
                    {"earlier_ts": earlier, "later_ts": later},
                    ordering.weight,
                )
            )
        if self.answer_after and self.answer_seq is not None:
            answered = self.answer_seq(episode)
            for event in self.answer_after:
                ts = episode.activation_ts(event)
                if answered is None:
                    checks.append(CheckOutcome(f"answered_after:{event}", False,
                                               {"reason": "no answer recorded"}))
                    continue
                seq_of_event = _seq_at_or_after(episode, ts)
                passed = ts is not None and seq_of_event is not None and seq_of_event <= answered
                checks.append(
                    CheckOutcome(f"answered_after:{event}", passed,
                                 {"event_ts": ts, "answer_seq": answered,
                                  "event_seq": seq_of_event})
                )
        checks_tuple = tuple(checks)
        if not checks_tuple:
            # Every constraint was vacuous, so nothing was measured. Saying
            # "passed" here would pay an episode for constraints it never got
            # close enough to violate.
            return VerifierResult(
                verifier=type(self).__name__, layer=self.layer, score=0.0,
                checks=(CheckOutcome("orderings", False,
                                     {"reason": "no constrained event occurred"}),),
                available=False,
            )
        return VerifierResult(
            verifier=type(self).__name__, layer=self.layer, score=_score(checks_tuple),
            checks=checks_tuple, fatal_on_failure=self.fatal_on_failure,
        )


def _seq_at_or_after(episode: Episode, ts: str | None) -> int | None:
    """The first recorded action at or after `ts`, which is when an event fired."""
    if ts is None:
        return None
    for action in episode.actions:
        if action.ts and float(action.ts) >= float(ts):
            return action.seq
    return None


# ---------------------------------------------------------------------------
# Forbidden behaviour
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Forbidden:
    """Something that must not have happened, and what it costs.

    `charge` is the whole of the difference between the two kinds of negative
    finding. Left as None -- the default -- an occurrence ends the episode: no
    amount of good work elsewhere settles a report produced by deleting the
    contradicting message. Given a function, the occurrence is priced instead,
    and the reward loses what that function says it is worth.

    One declaration rather than two, because a second class and a second
    verifier for the priced kind meant two ways of saying "this was wrong" and
    two places to look when a score needed explaining.
    """

    name: str
    detect: Callable[[Episode], list[str]]
    description: str = ""
    #: What one occurrence costs, from its label. None disqualifies.
    charge: Callable[[str], float] | None = None


@dataclass(frozen=True, slots=True)
class NegativeVerifier:
    """Everything that went wrong in a way credit must not launder.

    The only place a negative finding is decided. Each rule says for itself
    whether an occurrence is priced or fatal, and this reports the two
    outcomes separately -- `penalties` for what reduces the reward,
    `disqualifying_checks` for what ends the episode. The engine applies what
    it is told rather than re-deriving the same facts from the same checks by
    a second route.
    """

    forbidden: tuple[Forbidden, ...]
    name: str = "side_effects"
    layer: str = "negative"
    fatal_on_failure: bool = True

    def verify(self, episode: Episode) -> VerifierResult:
        checks: list[CheckOutcome] = []
        charges: list[Penalty] = []
        fatal: list[str] = []
        for rule in self.forbidden:
            try:
                offenders = list(rule.detect(episode))
            except Exception as error:  # noqa: BLE001 - a broken detector is a
                # failed check, never a clean episode and never an invented charge.
                offenders = [f"detector error: {type(error).__name__}: {error}"]
            detail: dict[str, Any] = {"offenders": offenders[:20], "count": len(offenders)}
            if rule.description:
                detail["description"] = rule.description
            if offenders and rule.charge is not None:
                detail["charged"] = round(sum(rule.charge(o) for o in offenders), 6)
                charges.extend(
                    Penalty(reason=f"{rule.name}: {offender}",
                            amount=rule.charge(offender), source=rule.name)
                    for offender in offenders
                )
            elif offenders:
                fatal.append(rule.name)
            checks.append(CheckOutcome(rule.name, not offenders, detail))
        checks_tuple = tuple(checks)
        return VerifierResult(
            verifier=type(self).__name__, layer=self.layer, score=_score(checks_tuple),
            checks=checks_tuple, fatal_on_failure=self.fatal_on_failure,
            disqualifying_checks=tuple(fatal),
            penalties=tuple(charges),
        )


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolExpectation:
    name: str
    detect: Callable[[Episode], bool]
    description: str = ""
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class ActionPenalty:
    """A task-specific detector for irrelevant actions.

    The detector returns one stable label per action. This makes the charge
    proportional to what the agent actually did instead of charging once for
    an entire category, however many detours it took.
    """

    name: str
    detect: Callable[[Episode], list[str]]
    amount_per_action: float
    description: str = ""


@dataclass(frozen=True, slots=True)
class TrajectoryVerifier:
    """Task progress was made without unnecessary detours.

    The task contract should express necessary outcomes rather than prescribe
    one tool route. Discouraged categories remain available for broad generic
    checks; `action_penalties` are the precise mechanism, charging once for
    every irrelevant action returned by a detector.
    """

    necessary: tuple[ToolExpectation, ...] = ()
    discouraged: tuple[ToolExpectation, ...] = ()
    action_penalties: tuple[ActionPenalty, ...] = ()
    name: str = "trajectory"
    layer: str = "trajectory"
    fatal_on_failure: bool = False

    def verify(self, episode: Episode) -> VerifierResult:
        if not episode.has_action_log:
            return VerifierResult(
                verifier=type(self).__name__, layer=self.layer, score=0.0,
                checks=(CheckOutcome("action_log", False,
                                     {"reason": "this episode was recorded without an "
                                                "action log; trajectory cannot be graded"}),),
                available=False,
            )
        checks: list[CheckOutcome] = []
        for expectation in self.necessary:
            passed = bool(expectation.detect(episode))
            checks.append(CheckOutcome(
                f"necessary:{expectation.name}", passed,
                {"description": expectation.description}, expectation.weight,
            ))
        for expectation in self.discouraged:
            tripped = bool(expectation.detect(episode))
            checks.append(CheckOutcome(f"unnecessary:{expectation.name}", not tripped,
                                       {"description": expectation.description}))
        action_penalty = 0.0
        for rule in self.action_penalties:
            offenders = list(rule.detect(episode))
            amount = max(0.0, rule.amount_per_action) * len(offenders)
            action_penalty += amount
            checks.append(CheckOutcome(
                f"irrelevant:{rule.name}",
                not offenders,
                {
                    "description": rule.description,
                    "offenders": offenders[:50],
                    "count": len(offenders),
                    "amount_per_action": rule.amount_per_action,
                    "penalty": amount,
                },
                0.0,
            ))
        checks_tuple = tuple(checks)
        earned = _score(checks_tuple[:len(self.necessary)])
        discouraged_checks = checks_tuple[
            len(self.necessary):len(self.necessary) + len(self.discouraged)
        ]
        deducted = sum(not check.passed for check in discouraged_checks)
        category_penalty = deducted / len(self.discouraged) if self.discouraged else 0.0
        return VerifierResult(
            verifier=type(self).__name__, layer=self.layer,
            score=max(0.0, earned - category_penalty - action_penalty),
            checks=checks_tuple, fatal_on_failure=self.fatal_on_failure,
        )


@dataclass(frozen=True, slots=True)
class Contract:
    """One task's verifiers, grouped by the layer each belongs to."""

    name: str
    final_state: tuple[Any, ...] = ()
    milestones: tuple[Any, ...] = ()
    trajectory: tuple[Any, ...] = ()
    negative: tuple[Any, ...] = ()
    milestone_events: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
