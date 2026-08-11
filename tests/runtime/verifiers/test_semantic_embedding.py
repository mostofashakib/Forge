"""Semantic verification backed by embeddings rather than a language model.

Scoring how well a piece of text satisfies a rubric is a text-similarity
question. An embedding model answers it deterministically, offline, in
milliseconds, and with no provider to be independent of.
"""
from __future__ import annotations

import pytest

from forge.runtime.verifiers.semantic import EmbeddingSemanticClient, SemanticVerifier


class _FixedScorer:
    """Stands in for SentenceEmbeddingScorer without loading a model."""

    def __init__(self, score: float) -> None:
        self._score = score
        self.calls: list[tuple[str, str]] = []

    def score(self, reference: str, candidate: str) -> float:
        self.calls.append((reference, candidate))
        return self._score


def test_embedding_client_scores_text_against_the_rubric():
    scorer = _FixedScorer(0.9)
    client = EmbeddingSemanticClient(scorer=scorer)

    raw = client.judge("Rubric: polite reply\n\nText: Thank you kindly\n\n")

    assert float(raw) == pytest.approx(0.9)
    assert scorer.calls


def test_embedding_client_returns_a_parseable_score_for_the_verifier():
    """The verifier parses a float out of the response, so it must be one."""
    client = EmbeddingSemanticClient(scorer=_FixedScorer(0.42))

    assert 0.0 <= float(client.judge("Rubric: x\n\nText: y\n\n")) <= 1.0


def test_a_verifier_using_embeddings_passes_on_high_similarity():
    verifier = SemanticVerifier(
        rubric="a polite acknowledgement",
        state_field="reply",
        mode="live",
        llm_client=EmbeddingSemanticClient(scorer=_FixedScorer(0.95)),
    )

    result = verifier.check({"reply": "Thanks very much"}, None, {})

    assert result.passed is True


def test_a_verifier_using_embeddings_fails_on_low_similarity():
    verifier = SemanticVerifier(
        rubric="a polite acknowledgement",
        state_field="reply",
        mode="live",
        llm_client=EmbeddingSemanticClient(scorer=_FixedScorer(0.1)),
    )

    result = verifier.check({"reply": "go away"}, None, {})

    assert result.passed is False
    assert result.evidence is not None


def test_an_empty_field_does_not_score_as_similar():
    """False-positive guard: nothing must not satisfy a rubric."""
    client = EmbeddingSemanticClient(scorer=_FixedScorer(0.99))

    assert float(client.judge("Rubric: anything\n\nText: \n\n")) == 0.0


def test_a_malformed_prompt_scores_zero_rather_than_raising():
    client = EmbeddingSemanticClient(scorer=_FixedScorer(0.99))

    assert float(client.judge("no rubric or text markers here")) == 0.0


def test_the_scorer_is_not_consulted_when_there_is_no_text():
    scorer = _FixedScorer(0.99)
    EmbeddingSemanticClient(scorer=scorer).judge("Rubric: x\n\nText: \n\n")

    assert scorer.calls == []


def test_scoring_is_deterministic_across_repeated_calls():
    client = EmbeddingSemanticClient(scorer=_FixedScorer(0.73))
    prompt = "Rubric: a summary\n\nText: the report summarises Q3\n\n"

    assert client.judge(prompt) == client.judge(prompt)
