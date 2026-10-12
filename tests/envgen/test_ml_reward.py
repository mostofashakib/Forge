"""ML similarity scorers report real similarity and fail loudly when they cannot score."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import forge.envgen.ml_reward as ml_reward
from forge.envgen.ml_reward import NGramScorer, SentenceEmbeddingScorer


class _VectorModel:
    """Encodes each text to a fixed unit vector."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def encode(self, texts, normalize_embeddings: bool):
        return np.array([self._vectors[text] for text in texts], dtype=float)


class _BrokenModel:
    def encode(self, texts, normalize_embeddings: bool):
        raise RuntimeError("tokenizer crashed")


@pytest.fixture
def no_cached_model(monkeypatch):
    monkeypatch.setattr(ml_reward, "_embedding_model", None)


def _scorer(vectors: dict[str, list[float]]) -> SentenceEmbeddingScorer:
    return SentenceEmbeddingScorer(model=_VectorModel(vectors))


def test_unrelated_texts_score_zero_not_one_half():
    assert _scorer({"a": [1.0, 0.0], "b": [0.0, 1.0]}).score("a", "b") == pytest.approx(0.0)


def test_opposite_texts_score_zero():
    assert _scorer({"a": [1.0, 0.0], "b": [-1.0, 0.0]}).score("a", "b") == pytest.approx(0.0)


def test_identical_texts_score_one():
    assert _scorer({"a": [0.6, 0.8], "b": [0.6, 0.8]}).score("a", "b") == pytest.approx(1.0)


def test_partly_similar_texts_score_their_cosine():
    assert _scorer({"a": [1.0, 0.0], "b": [0.6, 0.8]}).score("a", "b") == pytest.approx(0.6)


def test_empty_text_has_no_similarity():
    assert _scorer({}).score("  ", "anything") == 0.0


def test_a_missing_embedding_library_raises_with_an_install_hint(monkeypatch, no_cached_model):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(ImportError, match="sentence-transformers"):
        SentenceEmbeddingScorer().score("reference", "candidate")


def test_an_encoding_failure_raises_instead_of_scoring_zero():
    with pytest.raises(RuntimeError, match="tokenizer crashed"):
        SentenceEmbeddingScorer(model=_BrokenModel()).score("reference", "candidate")


def test_a_missing_rouge_library_raises_instead_of_scoring_zero(monkeypatch):
    monkeypatch.setitem(sys.modules, "rouge_score", None)

    with pytest.raises(ImportError):
        NGramScorer("rougeL").score("the cat sat", "the cat sat")


def test_rouge_scores_real_overlap():
    assert NGramScorer("rougeL").score("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_reward_reevaluation_leaves_out_episodes_it_could_not_score(monkeypatch):
    from backend.app.api.evaluate import _run_reward_eval_ml

    class _Scorer:
        def score(self, reference, candidate):
            if "broken" in candidate:
                raise RuntimeError("model unavailable")
            return 0.7

    monkeypatch.setattr(ml_reward, "build_scorer", lambda method: _Scorer())
    episodes = [
        (SimpleNamespace(id="ok", total_reward=0.5), [{"command": "ls"}]),
        (SimpleNamespace(id="bad", total_reward=0.5), [{"command": "broken"}]),
    ]

    result = _run_reward_eval_ml("list files", episodes, "embeddings")

    assert [r.episode_id for r in result.reevaluations] == ["ok"]
    assert result.reevaluations[0].new_score == pytest.approx(0.7)
    assert "1 of 2 episodes could not be scored" in result.summary
