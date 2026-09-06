"""A composable, auditable verifier stack for Harbor episodes.

Grading an agent well means answering more than "is the final answer right".
An episode can reach the right end state by deleting the evidence, by skipping
the work, or by doing the steps in an order that could not have produced the
conclusion honestly. Each of those is a different question, so each is a
different layer here, and every layer reports its own checks:

    final_state   assertions over the recorded end state
    milestones    the required events happened, none skipped, in a valid order
    trajectory    the necessary tool calls were made, and no gratuitous ones
    negative      forbidden side effects did not occur

`LayeredVerifier` composes them into one verdict, `TieredRewardEngine` turns
that verdict into a `RewardBreakdown` you can read line by line, and
`RewardHackingAuditor` asks separately whether a pass was earned.

Every grading path reads world-side evidence: the state export and the
tracker's action log, both written by the environment in a container the agent
cannot reach.
"""

from verifiers.auditor import AuditFinding, AuditReport, RewardHackingAuditor
from verifiers.checks import (
    ActionPenalty,
    EventVerifier,
    ExactStateVerifier,
    NegativeVerifier,
    PolicyVerifier,
    TemporalVerifier,
)
from verifiers.episode import Action, Episode
from verifiers.harbor import blank_reward_dict, reward_dict, reward_keys
from verifiers.layered import LayeredVerifier
from verifiers.presets import (
    REWARD_PRESETS,
    RewardPreset,
    TieredRewardEngine,
    VerifierComposer,
    load_experiment,
)
from verifiers.results import CheckOutcome, EpisodeEvaluation, RewardBreakdown, VerifierResult

__all__ = [
    "Action", "ActionPenalty", "AuditFinding", "AuditReport", "CheckOutcome",
    "Episode",
    "EpisodeEvaluation", "EventVerifier", "ExactStateVerifier",
    "LayeredVerifier", "NegativeVerifier", "PolicyVerifier",
    "REWARD_PRESETS",
    "RewardBreakdown", "RewardHackingAuditor", "RewardPreset", "TemporalVerifier",
    "TieredRewardEngine", "VerifierComposer", "VerifierResult", "blank_reward_dict",
    "load_experiment", "reward_dict", "reward_keys",
]
