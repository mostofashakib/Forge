# tests/runtime/test_errors.py
import pytest
from forge.runtime.errors import (
    AgentError,
    BrowserContractViolation,
    ComputerContractViolation,
    DeterminismError,
    DeterminismViolation,
    EnvironmentBuildError,
    EpisodeNotFoundError,
    ForgeError,
    InvalidActionError,
    ResetRequiredError,
    ToolContractViolation,
    VerifierConfigurationError,
)


def test_every_error_descends_from_forge_error():
    for exc_type in [
        InvalidActionError, ToolContractViolation, ComputerContractViolation,
        BrowserContractViolation, DeterminismViolation, EnvironmentBuildError,
        VerifierConfigurationError, EpisodeNotFoundError, AgentError, ResetRequiredError,
    ]:
        assert issubclass(exc_type, ForgeError)


def test_errors_carry_origin_code_and_detail():
    err = ToolContractViolation("missing param 'thread_id'")
    assert err.origin == "environment"
    assert err.code == "TOOL_CONTRACT_VIOLATION"
    assert err.detail == "missing param 'thread_id'"

    assert ComputerContractViolation("x").code == "COMPUTER_CONTRACT_VIOLATION"
    assert BrowserContractViolation("x").code == "BROWSER_CONTRACT_VIOLATION"
    assert VerifierConfigurationError("x").origin == "verifier"
    assert AgentError("x").origin == "agent"
    assert EnvironmentBuildError("x").origin == "builder"
    assert EpisodeNotFoundError("x").origin == "replay"


def test_to_dict_includes_origin_and_preserves_legacy_keys():
    err = InvalidActionError("bad action", code="UNKNOWN_ACTION_TYPE")
    payload = err.to_dict()
    assert payload["error"] == "INVALID_ACTION"
    assert payload["code"] == "UNKNOWN_ACTION_TYPE"
    assert payload["detail"] == "bad action"
    assert payload["origin"] == "environment"


def test_cause_is_chained_and_reported():
    root = KeyError("thread_id")
    err = ForgeError("transition crashed", cause=root)
    assert err.__cause__ is root
    assert "thread_id" in err.to_dict()["cause"]


def test_contract_violations_are_catchable_as_invalid_action():
    # Callers that catch InvalidActionError keep working for all contracts.
    for exc_type in [ToolContractViolation, ComputerContractViolation, BrowserContractViolation]:
        assert issubclass(exc_type, InvalidActionError)


def test_backwards_compatible_with_builtin_exception_types():
    # Pre-hierarchy callers caught builtins; granular types still satisfy them.
    assert issubclass(EpisodeNotFoundError, ValueError)
    assert issubclass(AgentError, ValueError)
    assert issubclass(EnvironmentBuildError, ValueError)
    assert issubclass(VerifierConfigurationError, RuntimeError)
    assert issubclass(ResetRequiredError, RuntimeError)


# ---------------------------------------------------------------------------
# Raise sites actually use the granular types
# ---------------------------------------------------------------------------

def test_interaction_contracts_raise_granular_violations():
    from forge.runtime.interaction import (
        BrowserUse, BrowserUseSchema, ComputerUse, ComputerUseSchema, ToolUse, ToolUseSchema,
    )
    from forge.runtime.snapshot import ToolSpec

    with pytest.raises(ToolContractViolation):
        ToolUse(schema=ToolUseSchema(tools=[ToolSpec(name="a")]), executor=lambda a: None).execute({"type": "b"})
    with pytest.raises(ComputerContractViolation):
        ComputerUse(schema=ComputerUseSchema(), executor=lambda a: None).execute({"action_type": "exec"})
    with pytest.raises(BrowserContractViolation):
        BrowserUse(schema=BrowserUseSchema(), executor=lambda a: None).execute({"action_type": "warp"})


def test_env_builder_raises_build_error():
    from forge.runtime.env_builder import EnvBuilder

    with pytest.raises(EnvironmentBuildError):
        EnvBuilder("x", domain="test").build()


def test_layered_verifier_raises_verifier_configuration_error():
    from forge.runtime.layered_verifier import LayeredVerifier

    v = LayeredVerifier("t")
    v.add_llm_judge("quality", rubric="r")
    with pytest.raises(VerifierConfigurationError):
        v({}, None, {})


def test_agent_factory_raises_agent_error():
    from forge.runtime.agents.factory import make_agent

    with pytest.raises(AgentError):
        make_agent("quantum:gpt-12")


def test_env_step_before_reset_raises_reset_required():
    from forge.runtime.env import ForgeEnv
    from forge.runtime.reward import RewardEngine
    from forge.runtime.snapshot import EnvironmentSpec
    from forge.runtime.transition import TransitionEngine
    from forge.runtime.verifier import VerifierEngine

    te = TransitionEngine()
    te.register("noop", lambda s, a, c: None)
    env = ForgeEnv(
        env_spec=EnvironmentSpec(name="e", domain="t"),
        initial_state_factory=None,
        transition_engine=te,
        verifier_engine=VerifierEngine(),
        reward_engine=RewardEngine(),
    )
    with pytest.raises(ResetRequiredError):
        env.step({"type": "noop"})


def test_replay_service_raises_episode_not_found():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.database import Base
    from forge.runtime.replay import ReplayService

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    with pytest.raises(EpisodeNotFoundError):
        ReplayService().load_episode("ep_missing", db)
