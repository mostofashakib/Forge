from __future__ import annotations

import os
from dataclasses import dataclass


def _integer(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _number(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class EnvGenConfig:
    """All tunable generation limits and LLM budgets in one place."""

    capable_llm_tokens: int = 8192
    telemetry_llm_tokens: int = 32768
    standard_llm_tokens: int = 4096
    fast_llm_tokens: int = 2048
    action_llm_tokens: int = 512
    cli_llm_tokens: int = 256
    grading_llm_tokens: int = 1024
    research_search_results: int = 3
    research_http_timeout: float = 8.0
    research_document_chars: int = 20_000
    research_total_source_chars: int = 24_000
    specialist_context_chars: int = 12_000
    specialist_items_per_section: int = 8
    generated_file_review_chars: int = 16_000
    state_bridge_input_chars: int = 2_000

    @classmethod
    def from_env(cls) -> EnvGenConfig:
        return cls(
            capable_llm_tokens=_integer("FORGE_ENVGEN_CAPABLE_TOKENS", 8192),
            telemetry_llm_tokens=_integer("FORGE_ENVGEN_TELEMETRY_TOKENS", 32768),
            standard_llm_tokens=_integer("FORGE_ENVGEN_STANDARD_TOKENS", 4096),
            fast_llm_tokens=_integer("FORGE_ENVGEN_FAST_TOKENS", 2048),
            action_llm_tokens=_integer("FORGE_ENVGEN_ACTION_TOKENS", 512),
            cli_llm_tokens=_integer("FORGE_ENVGEN_CLI_TOKENS", 256),
            grading_llm_tokens=_integer("FORGE_ENVGEN_GRADING_TOKENS", 1024),
            research_search_results=_integer("FORGE_RESEARCH_SEARCH_RESULTS", 3),
            research_http_timeout=_number("FORGE_RESEARCH_HTTP_TIMEOUT", 8.0),
            research_document_chars=_integer("FORGE_RESEARCH_DOCUMENT_CHARS", 20_000),
            research_total_source_chars=_integer("FORGE_RESEARCH_SOURCE_CHARS", 24_000),
            specialist_context_chars=_integer("FORGE_SPECIALIST_CONTEXT_CHARS", 12_000),
            specialist_items_per_section=_integer("FORGE_SPECIALIST_CONTEXT_ITEMS", 8),
            generated_file_review_chars=_integer("FORGE_REVIEW_FILE_CHARS", 16_000),
            state_bridge_input_chars=_integer("FORGE_STATE_BRIDGE_INPUT_CHARS", 2_000),
        )


def envgen_config() -> EnvGenConfig:
    return EnvGenConfig.from_env()
