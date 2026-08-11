from __future__ import annotations
import hashlib
import os
import re
import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable
from forge.runtime.verification import CheckResult

_PASS_THRESHOLD = 0.5


@runtime_checkable
class SemanticLLMClient(Protocol):
    def judge(self, prompt: str) -> str: ...


class MockSemanticLLMClient:
    def __init__(self, score: float = 1.0) -> None:
        self._score = score

    def judge(self, prompt: str) -> str:
        return str(self._score)


class EmbeddingSemanticClient:
    """Score a rubric against text with sentence embeddings instead of an LLM.

    "How well does this text satisfy this rubric" is a text-similarity question.
    An embedding model answers it deterministically, offline, and in
    milliseconds — and, having no provider, it needs no independence guarantee
    from the model that generated the environment.

    Satisfies the :class:`SemanticLLMClient` protocol so it drops into an
    existing :class:`SemanticVerifier` unchanged.
    """

    def __init__(self, scorer: object | None = None) -> None:
        if scorer is None:
            from forge.envgen.ml_reward import SentenceEmbeddingScorer

            scorer = SentenceEmbeddingScorer()
        self._scorer = scorer

    def judge(self, prompt: str) -> str:
        rubric, text = _split_rubric_prompt(prompt)
        # An absent rubric or empty text is a score of zero, not a similarity
        # question: nothing can satisfy a rubric, however close the embeddings
        # of two empty strings happen to land.
        if not rubric or not text:
            return "0.0"
        return str(self._scorer.score(rubric, text))


def _split_rubric_prompt(prompt: str) -> tuple[str, str]:
    """Pull the rubric and text back out of the verifier's prompt format."""
    rubric_match = re.search(r"Rubric:\s*(.*?)\n\n", prompt, re.DOTALL)
    text_match = re.search(r"Text:\s*(.*?)\n\n", prompt, re.DOTALL)
    if rubric_match is None or text_match is None:
        return "", ""
    return rubric_match.group(1).strip(), text_match.group(1).strip()


class SemanticVerifier:
    def __init__(
        self,
        rubric: str,
        state_field: str,
        mode: str = "mock",
        cache_path: Path | None = None,
        llm_client: SemanticLLMClient | None = None,
    ) -> None:
        if mode not in ("mock", "cached", "live"):
            raise ValueError(f"Invalid mode: {mode!r}. Must be 'mock', 'cached', or 'live'.")
        if os.environ.get("FORGE_ENV") == "test":
            mode = "mock"
        self._rubric = rubric
        self._state_field = state_field
        self._mode = mode
        self._cache_path = cache_path
        self._llm_client = llm_client

    def check(self, state: dict, trajectory, task: dict) -> CheckResult:
        text = str(state.get(self._state_field, ""))
        name = f"semantic:{self._state_field}"

        if self._mode == "mock":
            score = 1.0
        elif self._mode == "cached":
            score = self._cached_score(text)
        else:
            score = self._live_score(text)

        passed = score >= _PASS_THRESHOLD
        return CheckResult(
            name=name,
            passed=passed,
            score=score,
            evidence=None if passed else f"Semantic score {score:.2f} below threshold {_PASS_THRESHOLD}",
        )

    def _live_score(self, text: str) -> float:
        if self._llm_client is None:
            return 0.0
        try:
            prompt = (
                f"Rubric: {self._rubric}\n\nText: {text}\n\n"
                "Score how well the text satisfies the rubric. "
                "Respond with a single float between 0.0 and 1.0."
            )
            raw = self._llm_client.judge(prompt)
            match = re.search(r"\b([01](?:\.\d+)?|\d*\.\d+)\b", raw)
            return min(1.0, max(0.0, float(match.group(1)))) if match else 0.0
        except Exception:
            return 0.0

    def _cached_score(self, text: str) -> float:
        rubric_hash = hashlib.sha256(self._rubric.encode()).hexdigest()
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_path = self._cache_path or Path("custom/.semantic_cache.db")
        con = sqlite3.connect(str(cache_path))
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS semantic_cache "
                "(rubric_hash TEXT NOT NULL, text_hash TEXT NOT NULL, score REAL NOT NULL, "
                "PRIMARY KEY (rubric_hash, text_hash))"
            )
            row = con.execute(
                "SELECT score FROM semantic_cache WHERE rubric_hash=? AND text_hash=?",
                (rubric_hash, text_hash),
            ).fetchone()
            if row is not None:
                return float(row[0])
            score = self._live_score(text)
            if self._llm_client is not None:
                con.execute(
                    "INSERT INTO semantic_cache VALUES (?, ?, ?)",
                    (rubric_hash, text_hash, score),
                )
                con.commit()
            return score
        finally:
            con.close()
