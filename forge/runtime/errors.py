from __future__ import annotations


class ForgeError(Exception):
    """Base for all Forge errors.

    Every error carries `detail` (what happened), `code` (machine-readable
    kind), `origin` (which component raised it: environment, agent, verifier,
    builder, replay, determinism), and an optional chained `cause`. Granular
    subclasses co-inherit the builtin type they replaced (ValueError,
    RuntimeError) so pre-hierarchy callers keep working.
    """

    label = "FORGE_ERROR"          # legacy "error" key in to_dict payloads
    default_code = "FORGE_ERROR"
    origin = "forge"

    def __init__(
        self,
        detail: str,
        *,
        code: str | None = None,
        origin: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.code = code or self.default_code
        if origin is not None:
            self.origin = origin
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> dict:
        payload = {
            "error": self.label,
            "code": self.code,
            "detail": self.detail,
            "origin": self.origin,
        }
        if self.__cause__ is not None:
            payload["cause"] = repr(self.__cause__)
        return payload


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class InvalidActionError(ForgeError):
    """An agent action the environment cannot accept."""

    label = "INVALID_ACTION"
    default_code = "INVALID_ACTION"
    origin = "environment"

    def __init__(self, detail: str, code: str | None = None) -> None:
        super().__init__(detail, code=code or self.default_code)


class ToolContractViolation(InvalidActionError):
    """Tool call breaking the ToolUseSchema contract."""

    default_code = "TOOL_CONTRACT_VIOLATION"


class ComputerContractViolation(InvalidActionError):
    """OS primitive breaking the ComputerUseSchema contract."""

    default_code = "COMPUTER_CONTRACT_VIOLATION"


class BrowserContractViolation(InvalidActionError):
    """Browser primitive breaking the BrowserUseSchema contract."""

    default_code = "BROWSER_CONTRACT_VIOLATION"


class ResetRequiredError(ForgeError, RuntimeError):
    """Environment used before reset() established an episode."""

    default_code = "RESET_REQUIRED"
    origin = "environment"


class DeterminismViolation(ForgeError, RuntimeError):
    """Environment code broke a rule in its DeterminismConfig."""

    default_code = "DETERMINISM_VIOLATION"
    origin = "environment"


# ---------------------------------------------------------------------------
# Determinism check
# ---------------------------------------------------------------------------

class DeterminismError(ForgeError, RuntimeError):
    """Two identically-seeded rollouts produced different observations."""

    default_code = "NONDETERMINISTIC_ENV"
    origin = "determinism"

    def __init__(
        self,
        seed: int,
        first_hash: str,
        second_hash: str,
        divergent_step: int | None = None,
    ) -> None:
        self.seed = seed
        self.first_hash = first_hash
        self.second_hash = second_hash
        self.divergent_step = divergent_step
        location = (
            f"first divergence at observation {divergent_step}"
            if divergent_step is not None
            else "observation counts differ"
        )
        super().__init__(
            f"Environment is not deterministic for seed {seed}: "
            f"{first_hash} != {second_hash} ({location})"
        )


# ---------------------------------------------------------------------------
# Builder, verifier, agent, replay
# ---------------------------------------------------------------------------

class EnvironmentBuildError(ForgeError, ValueError):
    """EnvBuilder cannot assemble an environment from its inputs."""

    default_code = "ENV_BUILD_ERROR"
    origin = "builder"


class VerifierConfigurationError(ForgeError, RuntimeError):
    """A verifier is misconfigured (e.g. judge checks without a judge client)."""

    default_code = "VERIFIER_MISCONFIGURED"
    origin = "verifier"


class AgentError(ForgeError, ValueError):
    """An agent adapter cannot be created or has failed structurally."""

    default_code = "AGENT_ERROR"
    origin = "agent"


class EpisodeNotFoundError(ForgeError, ValueError):
    """A recorded episode referenced for replay does not exist."""

    default_code = "EPISODE_NOT_FOUND"
    origin = "replay"
