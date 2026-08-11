# Independent Quorum Validation

**Date:** 2026-08-11
**Status:** Approved (approach), in implementation

## Problem

Forge generates environments with an LLM and, until now, validated them with the
same LLM. A model marking its own homework is not evidence. Three specific
failures:

1. **Self-marking.** `ReviewerAgent` runs its semantic review on
   `get_client(capable=True)` — the same client that generated the code it is
   reviewing. The build gate is not independent.
2. **Wrong tool.** `backend/app/api/detect.py` asks an LLM for distribution
   drift, reward collapse, and anomalous patterns. These are statistics. An LLM
   is slower, costlier, non-reproducible, and worse at them.
3. **Single opinion.** Every validation verdict comes from one model. There is
   no measure of how contestable a verdict is.

`EnvironmentCorrectnessAgent` is the counter-example to follow: it is pure AST
analysis with zero LLM calls, and it is the most reliable gate in the system.

## Design: typed jury per decision

A `Jury` per decision kind, each with aggregation appropriate to its verdict
type. Members implement one small interface and may be LLM-backed *or*
deterministic, so a statistical drift detector and a Gemini judge are the same
kind of thing to the jury. Swapping an LLM member for a traditional-ML member is
a member substitution, not a subsystem rewrite.

Rejected alternatives:

- **`QuorumClient` as an `LLMClient` decorator** — cannot aggregate correctly.
  Majority-voting a `GenerationReview` carrying typed issues is a different
  operation from majority-voting a pass/fail, and a generic client-level
  decorator must pretend they are the same. Also leaves traditional-ML
  validators homeless.
- **Tiered validator ladder with runtime escalation** — most validation points
  know statically which method fits; the escalation machinery would be unused
  indirection.

### Components

```
forge/validation/
  member.py     MemberVerdict, ValidationMember protocol
  jury.py       Jury base: deliberate(), aggregation, abstention
  quorum.py     Quorum construction from config + independence enforcement
  juries/
    review.py   ReviewJury        — accept/reject generated artifacts
    episode.py  EpisodeVerdictJury — pass/fail for one episode
```

`MemberVerdict` carries `member_id`, `family`, `passed` (`None` = the member
abstained), optional `score`, and `detail`. `JuryOutcome` carries the
`decision` (`None` = indeterminate), every member's verdict, the `agreement`
fraction, and a `reason`.

A member is not required to be an LLM. `StructuralMember` wraps a
`LayeredVerifier` and returns a deterministic verdict; it never abstains.

### Aggregation and abstention

Majority decides. When agreement among non-abstaining members falls below
`agreement_threshold`, the outcome is **indeterminate** and the episode is
excluded from the denominator.

`agreement_threshold` defaults to `1.0` — unanimity among voting members — which
is the faithful reading of "abstain when confidence is low". Lowering it to
`0.67` accepts 2-1 majorities on a three-member jury.

**Mitigation for the known risk.** Excluding data from a denominator can bias the
sample invisibly. Therefore:

- `abstention_rate` is a first-class field in every run record, reported
  alongside `heldout_pass_rate`.
- A run whose abstention rate exceeds `max_abstention_rate` (default `0.2`)
  fails rather than reporting a metric computed from a gutted denominator. A
  jury that cannot agree on a fifth of its cases is a broken instrument, and
  the run record should say so instead of quietly publishing the remainder.

### Independence

A jury refuses construction when any member shares a family with a generation
model, reusing `model_family()` and `GraderContaminationError` from
`forge.grading_provenance`. Tier swaps do not count as independence — Haiku and
Sonnet are one family.

Generation stays on Anthropic. Quorum members are drawn from Ollama (already
supported), OpenAI, and Gemini (both new providers on `get_client()`).

Configuration: `FORGE_QUORUM_MODELS` as a comma-separated `provider:model` list.
Empty means no quorum — a single judge via `get_judge_client()`, as today.

## Phasing

**A — Independence.** Validation core (`member`, `jury`, `quorum`), OpenAI and
Gemini providers, independence enforcement, `ReviewJury` wired into the build
gate.

**B — Right tool for the job.** Replace LLM calls where a traditional method is
better suited:

| Today | Replacement |
|---|---|
| LLM distribution-drift detection | Population stability index / KS test over episode feature distributions |
| LLM reward-collapse detection | Changepoint detection on the reward series |
| LLM anomalous-pattern detection | Outlier detection (IQR / z-score) on per-episode feature vectors |
| LLM semantic equivalence checks | `SentenceEmbeddingScorer` (already in `ml_reward.py`) |
| `ObjectiveScorer` per-step progress | Structural verification against compiled success conditions, with embedding similarity as dense shaping |

The last row absorbs the previously separate issue #3 (routing container grading
through `VerifierComposer`/`LayeredVerifier` instead of per-step LLM scoring).
`task_from_template` currently discards `success_conditions`; it must carry them.

**C — Quorum on verdicts.** `EpisodeVerdictJury` on final episode pass/fail,
`abstention_rate` in `RunResult`, and the `max_abstention_rate` guard.

Quorum covers gates and final verdicts only. Per-step scoring is not made 3x
more expensive; it is replaced under B.

## Testing

Every jury behavior needs a negative case and a false-positive guard, per the
repo's diversity gate:

- Unanimous pass, unanimous fail, and split-to-indeterminate.
- A member sharing the generator family is refused at construction.
- A member that looks independent (different model id, same family) is *also*
  refused.
- Abstaining members are excluded from the agreement denominator, not counted
  as votes against.
- A run exceeding `max_abstention_rate` fails and writes no result record.

## Out of scope

Retraining or fine-tuning any judge. Human-in-the-loop review. Changing the
reward presets.
