from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext


class EnvGenAgent(ABC):
    agent_id: str = "agent"
    depends_on: list[str] = []
    optional_depends_on: list[str] = []
    produces: list[str] = []

    @abstractmethod
    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None: ...


def render_correction_context(bus: Any, agent_id: str) -> str | None:
    """Render the repair-loop correction envelope for ``agent_id`` as a prompt block.

    Returns ``None`` when no correction is pending, so specialists can append the
    block to their LLM prompt only during a repair re-run. ``bus`` may be the raw
    :class:`ArtifactBus` or a scoped ``AgentChannel`` — both expose ``get``. The
    key format mirrors ``forge.envgen.repair.correction_key`` (inlined to avoid an
    import cycle).
    """
    try:
        correction = bus.get(f"correction:{agent_id}")
    except PermissionError:
        # Not in the agent's read scope — no correction is pending for it.
        return None
    if not correction:
        return None

    lines = [
        "A previous version of your output was rejected in review. "
        "Regenerate it so every finding below is resolved while keeping what "
        "already works:",
    ]
    for finding in correction.get("findings", []):
        artifact = finding.get("artifact")
        location = f" [{artifact}]" if artifact else ""
        lines.append(f"- ({finding.get('category', 'issue')}){location} {finding.get('message', '')}")

    criteria = correction.get("acceptance_criteria") or []
    if criteria:
        lines.append("")
        lines.append("Acceptance criteria that must hold:")
        lines.extend(f"- {item}" for item in criteria)

    prior = correction.get("prior_output")
    if prior:
        lines.append("")
        lines.append("Your previous output (revise it, do not start from scratch):")
        lines.append(_render_prior(prior))

    return "\n".join(lines)


def with_correction(bus: Any, agent_id: str, user: str) -> str:
    """Append the pending correction block (if any) to a specialist's LLM prompt."""
    block = render_correction_context(bus, agent_id)
    return f"{user}\n\n{block}" if block else user


def _render_prior(prior: Any) -> str:
    if isinstance(prior, dict):
        return "\n\n".join(
            f"=== {path} ===\n{content}" for path, content in prior.items()
        )
    return str(prior)
