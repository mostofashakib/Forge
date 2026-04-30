from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentPrompt:
    """
    Bundles all text used when prompting an LLM agent to select an action.
    Having the output contract here makes it clear that the agent must respond
    via a tool call — not with free-form text.
    """
    system: str
    observation_template: str   # {observation} placeholder → JSON-encoded env state
    action_description_template: str  # {action} placeholder → one tool per action type
    output_contract: str        # human-readable note on the required response format


FORGE_AGENT_PROMPT = AgentPrompt(
    system=(
        "You are an agent operating inside a simulated workflow environment.\n"
        "Each turn you receive the current environment state as JSON and must choose\n"
        "exactly one action by calling the corresponding tool.\n"
        "Do not produce free-form text — always respond with a tool call."
    ),
    observation_template=(
        "Current environment state:\n{observation}\n\n"
        "Select the action that best advances the workflow goal."
    ),
    action_description_template=(
        "Execute the '{action}' step in the current workflow state."
    ),
    output_contract=(
        "Response must be a single tool call matching one of the available action tools. "
        "Tool input fields must conform to the action's parameter schema."
    ),
)
