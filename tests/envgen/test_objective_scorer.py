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


# ---------------------------------------------------------------------------
# Grading routes through the judge client, not the generation client
# ---------------------------------------------------------------------------

def test_scorer_defaults_to_the_judge_model_not_the_generation_model(monkeypatch):
    monkeypatch.setenv("FORGE_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("FORGE_LLM_MODEL", "gemma4:26b")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "llama3.1:8b")

    scorer = ObjectiveScorer()

    assert scorer._client._model == "llama3.1:8b"
    assert scorer._client._model != "gemma4:26b"


# ---------------------------------------------------------------------------
# A judge that fails must not be recorded as a real verdict
# ---------------------------------------------------------------------------

class _FailingClient:
    def extract(self, system, user, schema):
        raise ConnectionError("judge unreachable")

    def extract_with_image(self, system, user, image_b64, schema):
        raise ConnectionError("judge unreachable")


def test_failed_judge_call_raises_instead_of_inventing_a_score():
    from forge.runtime.errors import GradingError

    scorer = ObjectiveScorer(client=_FailingClient())
    with pytest.raises(GradingError, match="judge unreachable"):
        scorer.score({"inbox_count": 3}, "Read an email")


def test_failed_visual_judge_call_raises_instead_of_inventing_a_score():
    from forge.runtime.errors import GradingError

    scorer = ObjectiveScorer(client=_FailingClient())
    with pytest.raises(GradingError):
        scorer.score_with_image("aGk=", "http://app", "Open settings")


def test_visual_scoring_uses_the_client_the_scorer_was_given():
    class _ImageClient:
        calls = 0

        def extract_with_image(self, system, user, image_b64, schema):
            _ImageClient.calls += 1
            return schema(score=0.7, reasoning="ok")

    scorer = ObjectiveScorer(client=_ImageClient())
    assert scorer.score_with_image("aGk=", "http://app", "Open settings") == 0.7
    assert _ImageClient.calls == 1


def test_container_episode_with_a_failed_judge_is_not_scored():
    import httpx
    from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig, EpisodeResult
    from forge.runtime.errors import GradingError

    config = EpisodeConfig(base_url="http://c", objective="do it", max_steps=1)
    runner = ContainerEpisodeRunner(config, scorer=ObjectiveScorer(client=_FailingClient()))
    runner._http = httpx.Client(base_url="http://c", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"n": 1})
    ))
    result = EpisodeResult(episode_id="cep_fail", config=config)

    with pytest.raises(GradingError):
        runner._finalize_result(result, {"n": 1}, {"endpoint": "/act", "payload": {}})
    assert result.total_reward == 0.0
    assert result.llm_verdicts == 0
