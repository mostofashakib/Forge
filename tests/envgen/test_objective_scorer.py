import pytest
from forge.envgen.objective import ObjectiveScorer


class _AlwaysOneClient:
    """Returns score=1.0 and captures the last user prompt."""
    last_user: str = ""

    def extract(self, system: str, user: str, schema):
        _AlwaysOneClient.last_user = user
        return schema(score=1.0, reasoning="ok")


def test_score_without_extra_context_still_works():
    scorer = ObjectiveScorer(client=_AlwaysOneClient())
    result = scorer.score({"inbox_count": 3}, "Read an email")
    assert result == 1.0


def test_score_includes_derived_diff_in_prompt():
    scorer = ObjectiveScorer(client=_AlwaysOneClient())
    derived = {"search_results": {"before": [], "after": [{"id": 1}]}}
    scorer.score(
        {"inbox_count": 3, "search_results": [{"id": 1}]},
        "Search for emails",
        derived_diff=derived,
        action_taken={"endpoint": "/search", "payload": {"query": "invoice"}},
    )
    assert "search_results" in _AlwaysOneClient.last_user
    assert "before" in _AlwaysOneClient.last_user


def test_score_without_derived_diff_prompt_unchanged():
    scorer = ObjectiveScorer(client=_AlwaysOneClient())
    scorer.score({"inbox_count": 3}, "Read an email")
    assert "Derived field changes" not in _AlwaysOneClient.last_user


def test_score_clamps_to_0_1():
    class _OverScorer:
        def extract(self, system, user, schema):
            return schema(score=2.5, reasoning="over")
    scorer = ObjectiveScorer(client=_OverScorer())
    assert scorer.score({}, "anything") == 1.0
