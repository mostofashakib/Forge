"""Every platform setting the Settings page edits, and the rules a save must pass.

API keys, database and file locations, and the values Forge injects into
containers are left out on purpose: keys stay in backend/.env only, and a
typo in a location breaks the app with no way back through the page.

Most settings are read once when the API and workers start, so a saved
change waits for a restart. `live` marks the ones each job reads fresh.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields

from forge.envgen.config import EnvGenConfig
from forge.envgen.container import (
    DEFAULT_BROWSER_IMAGE,
    DEFAULT_BROWSER_MEMORY,
    DEFAULT_CLI_IMAGE,
    DEFAULT_CLI_MEMORY,
    DEFAULT_CONTAINER_MEMORY,
    DEFAULT_CONTAINER_NANO_CPUS,
    DEFAULT_CONTAINER_PIDS,
    DEFAULT_PYTHON_BASE_IMAGE,
)
from forge.extraction.llm_client import _ANTHROPIC_CAPABLE, _ANTHROPIC_DEFAULT, _PROVIDER_DEFAULTS
from forge.grading_provenance import model_family
from forge.settings import DEFAULT_SANDBOX_LIMIT
from forge.taskfactory.model_settings import (
    SUPPORTED_PROVIDERS,
    VALIDATOR_MODEL_VAR,
    VALIDATOR_PROVIDER_VAR,
    TaskFactoryConfigError,
    resolve_models,
    writer_spec,
)
from forge.validation.quorum import parse_quorum_models

DEFAULT_OLLAMA_URL = "http://localhost:11434"

_MEMORY = re.compile(r"[1-9][0-9]*[kmg]?")
_URL = re.compile(r"https?://[A-Za-z0-9.-]+(:[0-9]+)?(/[A-Za-z0-9._/-]*)?")


class SettingsError(ValueError):
    """A value or a combination of values the platform cannot run with."""


@dataclass(frozen=True)
class Setting:
    key: str
    group: str
    label: str
    # text, choice, integer, number, memory, url, quorum
    kind: str
    default: str
    help: str = ""
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    # Empty is allowed and means "off" or "not configured".
    optional: bool = False
    # Read when each job starts. Everything else waits for a restart.
    live: bool = False


def _budget(key: str, label: str, field: str, minimum: float = 1, help: str = "") -> Setting:
    default = {f.name: f.default for f in fields(EnvGenConfig)}[field]
    kind = "number" if isinstance(default, float) else "integer"
    return Setting(key, "budgets", label, kind, str(default), help=help, minimum=minimum)


SETTINGS: tuple[Setting, ...] = (
    Setting("FORGE_LLM_PROVIDER", "models", "Generator provider", "choice", "anthropic",
            "Writes environments and synthetic tasks.", choices=SUPPORTED_PROVIDERS),
    Setting("FORGE_LLM_MODEL", "models", "Generator model", "text", _ANTHROPIC_DEFAULT,
            "Policies, scenarios and rewards."),
    Setting("FORGE_LLM_MODEL_CAPABLE", "models", "Capable generator model", "text", _ANTHROPIC_CAPABLE,
            "App code, reviews and task writing."),
    Setting("FORGE_JUDGE_PROVIDER", "models", "Judge provider", "choice", "",
            "Grades agent runs. Empty means the generator provider.", choices=SUPPORTED_PROVIDERS, optional=True),
    Setting("FORGE_JUDGE_MODEL", "models", "Judge model", "text", "",
            "Must come from another family than the generator. Empty means the generator grades itself.",
            optional=True),
    Setting("FORGE_QUORUM_MODELS", "models", "Validation quorum", "quorum", "",
            "provider:model list, one per family, none from the generator's family. Empty means no quorum.",
            optional=True),
    Setting(VALIDATOR_PROVIDER_VAR, "models", "Task validator provider", "choice", "",
            "Reviews generated tasks.", choices=SUPPORTED_PROVIDERS, optional=True, live=True),
    Setting(VALIDATOR_MODEL_VAR, "models", "Task validator model", "text", "",
            "Must come from another family than the capable generator.", optional=True, live=True),
    Setting("OLLAMA_BASE_URL", "models", "Ollama server", "url", DEFAULT_OLLAMA_URL),

    Setting("FORGE_DETERMINISM", "runtime", "Determinism", "choice", "on",
            "Off drops seeds for the determinism ablation.", choices=("on", "off")),
    Setting("FORGE_SANDBOX_LIMIT", "runtime", "Sandbox limit", "integer", str(DEFAULT_SANDBOX_LIMIT),
            "Most sandboxes running at once.", minimum=1),
    Setting("FORGE_DISABLE_PREWARM", "runtime", "Skip image prewarm", "choice", "0",
            "1 skips pulling base images when a worker starts.", choices=("0", "1")),

    Setting("FORGE_PYTHON_BASE_IMAGE", "containers", "Python base image", "text", DEFAULT_PYTHON_BASE_IMAGE),
    Setting("FORGE_CLI_IMAGE", "containers", "CLI image", "text", DEFAULT_CLI_IMAGE),
    Setting("FORGE_BROWSER_IMAGE", "containers", "Browser image", "text", DEFAULT_BROWSER_IMAGE),
    Setting("FORGE_CONTAINER_MEMORY", "containers", "App memory", "memory", DEFAULT_CONTAINER_MEMORY,
            "Like 512m or 2g."),
    Setting("FORGE_BROWSER_MEMORY", "containers", "Browser memory", "memory", DEFAULT_BROWSER_MEMORY),
    Setting("FORGE_CLI_MEMORY", "containers", "CLI memory", "memory", DEFAULT_CLI_MEMORY),
    Setting("FORGE_CONTAINER_NANO_CPUS", "containers", "CPU (nano-CPUs)", "integer",
            str(DEFAULT_CONTAINER_NANO_CPUS), "1000000000 is one CPU.", minimum=1),
    Setting("FORGE_CONTAINER_PIDS", "containers", "Process limit", "integer", str(DEFAULT_CONTAINER_PIDS),
            minimum=1),

    _budget("FORGE_ENVGEN_CAPABLE_TOKENS", "Capable tokens", "capable_llm_tokens"),
    _budget("FORGE_ENVGEN_TELEMETRY_TOKENS", "Telemetry tokens", "telemetry_llm_tokens"),
    _budget("FORGE_ENVGEN_STANDARD_TOKENS", "Standard tokens", "standard_llm_tokens"),
    _budget("FORGE_ENVGEN_FAST_TOKENS", "Fast tokens", "fast_llm_tokens"),
    _budget("FORGE_ENVGEN_ACTION_TOKENS", "Action tokens", "action_llm_tokens"),
    _budget("FORGE_ENVGEN_CLI_TOKENS", "CLI tokens", "cli_llm_tokens"),
    _budget("FORGE_ENVGEN_GRADING_TOKENS", "Grading tokens", "grading_llm_tokens"),
    _budget("FORGE_ENVGEN_MAX_REPAIR_ROUNDS", "Repair rounds", "max_repair_rounds", minimum=0,
            help="Reviewer repair attempts. 0 fails on the first rejection."),
    _budget("FORGE_RESEARCH_SEARCH_RESULTS", "Search results", "research_search_results"),
    _budget("FORGE_RESEARCH_HTTP_TIMEOUT", "Research timeout (s)", "research_http_timeout", minimum=0.001),
    _budget("FORGE_RESEARCH_DOCUMENT_CHARS", "Chars per document", "research_document_chars"),
    _budget("FORGE_RESEARCH_SOURCE_CHARS", "Chars across sources", "research_total_source_chars"),
    _budget("FORGE_REVIEW_FILE_CHARS", "Chars per reviewed file", "generated_file_review_chars"),
    _budget("FORGE_SPECIALIST_CONTEXT_CHARS", "Specialist context chars", "specialist_context_chars"),
    _budget("FORGE_SPECIALIST_CONTEXT_ITEMS", "Specialist items per section", "specialist_items_per_section"),
    _budget("FORGE_STATE_BRIDGE_INPUT_CHARS", "State bridge input chars", "state_bridge_input_chars"),
)

_BY_KEY = {s.key: s for s in SETTINGS}
KEYS: tuple[str, ...] = tuple(_BY_KEY)


def setting(key: str) -> Setting:
    if key not in _BY_KEY:
        raise SettingsError(f"{key} is not a setting this page edits")
    return _BY_KEY[key]


def check_value(spec: Setting, raw: str) -> str:
    """The value as it is saved, or SettingsError naming the setting."""
    value = raw.strip()
    if spec.kind == "choice":
        value = value.lower()
    if not value:
        if spec.optional:
            return ""
        raise SettingsError(f"{spec.key} cannot be empty")
    if spec.kind == "choice" and value not in spec.choices:
        raise SettingsError(f"{spec.key} must be one of {', '.join(spec.choices)}")
    if spec.kind in ("integer", "number"):
        _check_number(spec, value)
    if spec.kind == "memory" and not _MEMORY.fullmatch(value.lower()):
        raise SettingsError(f"{spec.key} must be a size like 512m or 2g")
    if spec.kind == "url" and not _URL.fullmatch(value):
        raise SettingsError(f"{spec.key} must be an http or https URL")
    if spec.kind == "quorum":
        try:
            parse_quorum_models(value)
        except ValueError as exc:
            raise SettingsError(f"{spec.key}: {exc}") from exc
    return value.lower() if spec.kind == "memory" else value


def _check_number(spec: Setting, value: str) -> None:
    try:
        number = int(value) if spec.kind == "integer" else float(value)
    except ValueError:
        raise SettingsError(f"{spec.key} must be {'a whole number' if spec.kind == 'integer' else 'a number'}") from None
    if spec.minimum is not None and number < spec.minimum:
        raise SettingsError(f"{spec.key} must be at least {spec.minimum:g}")


def check_models(env: Mapping[str, str]) -> None:
    """Refuse a judge, quorum member or task validator from the generator's family."""
    provider = (env.get("FORGE_LLM_PROVIDER") or "anthropic").lower()
    # The same fallbacks get_client uses when a model is unset.
    standard = env.get("FORGE_LLM_MODEL") or _PROVIDER_DEFAULTS[provider]
    generator = {model_family(standard), writer_spec(env).family}
    judge = (env.get("FORGE_JUDGE_MODEL") or "").strip()
    if env.get("FORGE_JUDGE_PROVIDER") and not judge:
        raise SettingsError("FORGE_JUDGE_PROVIDER is set, so FORGE_JUDGE_MODEL needs a model too")
    if judge and model_family(judge) in generator:
        raise SettingsError(
            f"FORGE_JUDGE_MODEL={judge!r} is from the {model_family(judge)!r} family, the same as the generator. "
            "A judge must come from another family."
        )
    try:
        quorum = parse_quorum_models(env.get("FORGE_QUORUM_MODELS") or "")
    except ValueError as exc:
        raise SettingsError(f"FORGE_QUORUM_MODELS: {exc}") from exc
    shared = sorted({spec.model for spec in quorum if spec.family in generator})
    if shared:
        raise SettingsError(f"FORGE_QUORUM_MODELS members {shared} share the generator's family")
    if env.get(VALIDATOR_PROVIDER_VAR) or env.get(VALIDATOR_MODEL_VAR):
        try:
            resolve_models(env)
        except TaskFactoryConfigError as exc:
            raise SettingsError(str(exc)) from exc
