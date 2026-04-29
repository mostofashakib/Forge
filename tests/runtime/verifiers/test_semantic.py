import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from forge.runtime.verifiers.semantic import SemanticVerifier, MockSemanticLLMClient


@dataclass
class _FakeTraj:
    steps: list = field(default_factory=list)
    events: list = field(default_factory=list)


def test_mock_mode_always_returns_score_one():
    v = SemanticVerifier(rubric="Must be polite", state_field="reply", mode="mock")
    result = v.check({"reply": "hello"}, _FakeTraj(), {})
    assert result.passed
    assert result.score == 1.0


def test_mock_mode_ignores_llm_client():
    v = SemanticVerifier(rubric="polite", state_field="reply", mode="mock", llm_client=None)
    result = v.check({"reply": "rude response"}, _FakeTraj(), {})
    assert result.passed


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


def test_live_mode_returns_zero_when_no_client():
    v = SemanticVerifier(rubric="polite", state_field="reply", mode="live", llm_client=None)
    result = v.check({"reply": "hello"}, _FakeTraj(), {})
    assert not result.passed
    assert result.score == 0.0


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


def test_forge_env_test_overrides_to_mock(monkeypatch):
    monkeypatch.setenv("FORGE_ENV", "test")
    client = MockSemanticLLMClient(score=0.0)
    v = SemanticVerifier(rubric="polite", state_field="reply", mode="live", llm_client=client)
    # FORGE_ENV=test forces mock mode — client is NOT called
    result = v.check({"reply": "rude"}, _FakeTraj(), {})
    assert result.passed  # mock always returns 1.0


def test_name_includes_state_field():
    v = SemanticVerifier(rubric="polite", state_field="my_field", mode="mock")
    result = v.check({"my_field": "hello"}, _FakeTraj(), {})
    assert result.name == "semantic:my_field"


def test_missing_state_field_uses_empty_string():
    client = MockSemanticLLMClient(score=0.6)
    v = SemanticVerifier(rubric="polite", state_field="missing", mode="live", llm_client=client)
    result = v.check({}, _FakeTraj(), {})
    assert result.passed  # score 0.6 >= 0.5


def test_invalid_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="Invalid mode"):
        SemanticVerifier(rubric="x", state_field="y", mode="bad_mode")
