from forge.envgen.config import EnvGenConfig


def test_envgen_config_reads_centralized_overrides(monkeypatch):
    monkeypatch.setenv("FORGE_ENVGEN_CAPABLE_TOKENS", "12000")
    monkeypatch.setenv("FORGE_SPECIALIST_CONTEXT_CHARS", "6000")
    monkeypatch.setenv("FORGE_RESEARCH_HTTP_TIMEOUT", "4.5")

    config = EnvGenConfig.from_env()

    assert config.capable_llm_tokens == 12000
    assert config.specialist_context_chars == 6000
    assert config.research_http_timeout == 4.5


def test_envgen_config_rejects_non_positive_limits(monkeypatch):
    monkeypatch.setenv("FORGE_RESEARCH_SEARCH_RESULTS", "0")

    try:
        EnvGenConfig.from_env()
    except ValueError as exc:
        assert "FORGE_RESEARCH_SEARCH_RESULTS" in str(exc)
    else:
        raise AssertionError("Expected a non-positive configuration to be rejected")
