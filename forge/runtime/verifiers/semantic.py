from __future__ import annotations
import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from forge.runtime.errors import GradingError, VerifierConfigurationError
from forge.runtime.verification import CheckResult

_PASS_THRESHOLD = 0.5


@runtime_checkable
class SemanticLLMClient(Protocol):
    def judge(self, prompt: str) -> str: ...


class _RubricScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)


_JUDGE_SYSTEM = (
    "You grade whether a text satisfies a rubric. "
    "Return a score between 0.0 (not at all) and 1.0 (fully)."
)


class ConfiguredJudgeClient:
    """Scores rubrics with the grading model set by FORGE_JUDGE_PROVIDER and FORGE_JUDGE_MODEL.

    The model client is built on first use, so a generated verifier module
    imports without a provider key and fails, naming the key, only when it grades.
    """

    def __init__(self, client: object | None = None) -> None:
        self._client = client

    def judge(self, prompt: str) -> str:
        if self._client is None:
            from forge.extraction import llm_client

            self._client = llm_client.get_judge_client()
        return str(self._client.extract(system=_JUDGE_SYSTEM, user=prompt, schema=_RubricScore).score)


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
        mode: str = "live",
        cache_path: Path | None = None,
        llm_client: SemanticLLMClient | None = None,
    ) -> None:
        if mode not in ("cached", "live"):
            raise ValueError(f"Invalid mode: {mode!r}. Must be 'cached' or 'live'.")
        if llm_client is None:
            raise VerifierConfigurationError(
                f"SemanticVerifier for {state_field!r} needs a judge client to score its rubric"
            )
        self._rubric = rubric
        self._state_field = state_field
        self._mode = mode
        self._cache_path = cache_path
        self._llm_client = llm_client

    def check(self, state: dict, trajectory, task: dict) -> CheckResult:
        text = str(state.get(self._state_field, ""))
        name = f"semantic:{self._state_field}"

        if self._mode == "cached":
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
        prompt = (
            f"Rubric: {self._rubric}\n\nText: {text}\n\n"
            "Score how well the text satisfies the rubric. "
            "Respond with a single float between 0.0 and 1.0."
        )
        raw = self._llm_client.judge(prompt)
        match = re.search(r"\b([01](?:\.\d+)?|\d*\.\d+)\b", raw)
        if match is None:
            raise GradingError(f"semantic judge returned no score for {self._state_field!r}: {raw!r}")
        return min(1.0, max(0.0, float(match.group(1))))

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
            con.execute(
                "INSERT INTO semantic_cache VALUES (?, ?, ?)",
                (rubric_hash, text_hash, score),
            )
            con.commit()
            return score
        finally:
            con.close()
