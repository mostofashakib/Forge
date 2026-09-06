"""The system prompt for a run.

The roles live in the environment (``slack_sim.prompts.PromptInterface``), where
they belong: which operator persona a rollout is run under is part of the task's
contract, not of the client that happens to be driving it. This module is the
lookup, plus the fallback for the case where the agent is being used away from a
checkout and the world's modules are not importable -- the agent's container
deliberately does not contain them.
"""

from __future__ import annotations

DEFAULT_PROMPT_ID = "incident_coordinator"

FALLBACK_SYSTEM = (
    "You are operating a Slack workspace through the tools you have been "
    "given. Read what is actually there before you act, follow the evidence "
    "where it leads rather than assuming, and post only what the request asks "
    "for. New messages appear in response to what you do, so re-read a thread "
    "after acting on it. Report what you found when you are done."
)

# Appended to whichever role prompt is in force. It is about the shape of the
# conversation rather than the task: which surface exists, what "finished" means,
# and that saying so is how the episode ends. The task itself arrives as the
# user instruction and is not restated here.
OPERATING_RULES = """
How this session works:
- Every action goes through a tool call. There is no shell and no file to edit.
- Tool results are JSON. `{"ok": true, ...}` succeeded; `{"ok": false, ...}`
  was refused, and the `error` explains what to ask differently. A refusal is
  information, not a dead end.
- Call one or a few tools per turn and read what comes back before the next.
- When the work is done, answer in plain text with no tool call. That ends the
  episode, so do not do it until the workspace is in the state you intend.
"""


def system_prompt(prompt_id: str = DEFAULT_PROMPT_ID, *, context: dict | None = None) -> str:
    """The system message for ``prompt_id``, with the operating rules appended."""
    try:
        from slack_sim.prompts import get_prompt_interface
    except ImportError:
        base = FALLBACK_SYSTEM
    else:
        interface = get_prompt_interface()
        try:
            base = interface.render(prompt_id, user_instruction="", context=context)["system"]
        except KeyError:
            base = interface.render(
                DEFAULT_PROMPT_ID, user_instruction="", context=context
            )["system"]
    return f"{base}\n{OPERATING_RULES}"
