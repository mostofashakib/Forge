import sqlite3
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from forge.runtime.errors import GradingError, VerifierConfigurationError
from forge.runtime.verifiers.semantic import ConfiguredJudgeClient, SemanticVerifier


class MockSemanticLLMClient:
    """Answers every rubric with a fixed score."""

    def __init__(self, score: float = 1.0) -> None:
        self._score = score

    def judge(self, prompt: str) -> str:
        return str(self._score)


class _RawClient:
    def __init__(self, raw: str) -> None:
        self._raw = raw

    def judge(self, prompt: str) -> str:
        return self._raw


class _BrokenClient:
    def judge(self, prompt: str) -> str:
        raise ConnectionError("judge unreachable")


@dataclass
class _FakeTraj:
    steps: list = field(default_factory=list)
    events: list = field(default_factory=list)


def test_mock_mode_no_longer_exists():
    with pytest.raises(ValueError, match="Invalid mode"):
        SemanticVerifier(rubric="polite", state_field="reply", mode="mock", llm_client=MockSemanticLLMClient())


def test_a_verifier_without_a_judge_cannot_be_built():
    with pytest.raises(VerifierConfigurationError, match="judge"):
        SemanticVerifier(rubric="polite", state_field="reply", llm_client=None)


def test_live_is_the_default_mode():
    v = SemanticVerifier(rubric="polite", state_field="reply", llm_client=MockSemanticLLMClient(score=0.2))
    assert not v.check({"reply": "rude"}, _FakeTraj(), {}).passed


def test_live_mode_calls_llm_client():
    client = MockSemanticLLMClient(score=0.8)
    v = SemanticVerifier(rubric="Must be polite", state_field="reply", mode="live", llm_client=client)
    result = v.check({"reply": "hello"}, _FakeTraj(), {})
    assert result.passed
    assert abs(result.score - 0.8) < 0.01


def test_live_mode_fails_when_score_below_threshold():
    client = MockSemanticLLMClient(score=0.3)
    v = SemanticVerifier(rubric="Must be polite", state_field="reply", mode="live", llm_client=client)
    result = v.check({"reply": "bad text"}, _FakeTraj(), {})
    assert not result.passed
    assert "0.30" in result.evidence


def test_a_judge_failure_raises_instead_of_scoring_zero():
    v = SemanticVerifier(rubric="polite", state_field="reply", llm_client=_BrokenClient())
    with pytest.raises(ConnectionError):
        v.check({"reply": "hello"}, _FakeTraj(), {})


def test_a_judge_answer_without_a_score_raises():
    v = SemanticVerifier(rubric="polite", state_field="reply", llm_client=_RawClient("I think it is fine"))
    with pytest.raises(GradingError, match="I think it is fine"):
        v.check({"reply": "hello"}, _FakeTraj(), {})


def test_cached_mode_reads_from_sqlite(tmp_path):
    db_path = tmp_path / "cache.db"
    client = MockSemanticLLMClient(score=0.9)
    v = SemanticVerifier(
        rubric="polite", state_field="reply", mode="cached",
        cache_path=db_path, llm_client=client,
    )
    result = v.check({"reply": "nice text"}, _FakeTraj(), {})
    assert result.passed

    # Second call: client not called again (same rubric+text)
    client2 = MockSemanticLLMClient(score=0.0)
    v2 = SemanticVerifier(
        rubric="polite", state_field="reply", mode="cached",
        cache_path=db_path, llm_client=client2,
    )
    result2 = v2.check({"reply": "nice text"}, _FakeTraj(), {})
    # Score is cached from first call — should NOT be 0.0
    assert result2.score > 0.5


def test_cached_mode_creates_sqlite_table(tmp_path):
    db_path = tmp_path / "cache.db"
    client = MockSemanticLLMClient(score=0.7)
    v = SemanticVerifier(
        rubric="polite", state_field="reply", mode="cached",
        cache_path=db_path, llm_client=client,
    )
    v.check({"reply": "hello"}, _FakeTraj(), {})
    con = sqlite3.connect(str(db_path))
    rows = con.execute("SELECT * FROM semantic_cache").fetchall()
    assert len(rows) == 1
    con.close()


def test_forge_env_test_does_not_change_the_verdict(monkeypatch):
    monkeypatch.setenv("FORGE_ENV", "test")
    v = SemanticVerifier(rubric="polite", state_field="reply", mode="live", llm_client=MockSemanticLLMClient(score=0.0))
    result = v.check({"reply": "rude"}, _FakeTraj(), {})
    assert not result.passed
    assert result.score == 0.0


def test_name_includes_state_field():
    v = SemanticVerifier(rubric="polite", state_field="my_field", llm_client=MockSemanticLLMClient())
    result = v.check({"my_field": "hello"}, _FakeTraj(), {})
    assert result.name == "semantic:my_field"


def test_missing_state_field_uses_empty_string():
    client = MockSemanticLLMClient(score=0.6)
    v = SemanticVerifier(rubric="polite", state_field="missing", mode="live", llm_client=client)
    result = v.check({}, _FakeTraj(), {})
    assert result.passed  # score 0.6 >= 0.5


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Invalid mode"):
        SemanticVerifier(rubric="x", state_field="y", mode="bad_mode", llm_client=MockSemanticLLMClient())


class _RecordingLLM:
    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = []

    def extract(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        self.calls.append((system, user))
        return schema(score=self.score)


def test_the_configured_judge_scores_the_prompt_with_the_grading_model():
    llm = _RecordingLLM(0.75)

    assert ConfiguredJudgeClient(llm).judge("Rubric: polite\n\nText: thanks\n\n") == "0.75"
    assert "Rubric: polite" in llm.calls[0][1]


def test_the_configured_judge_uses_the_judge_client_by_default(monkeypatch):
    import forge.extraction.llm_client as llm_client

    llm = _RecordingLLM(0.4)
    monkeypatch.setattr(llm_client, "get_judge_client", lambda: llm)

    assert ConfiguredJudgeClient().judge("Rubric: x\n\nText: y\n\n") == "0.4"
