"""Load a trained checkpoint into the runtime agent that serves it."""
from __future__ import annotations

from pathlib import Path

from forge.runtime.agents.vllm_agent import vLLMAgent
from forge.training.checkpoint import PolicyCheckpoint


def load_policy_agent(checkpoint_dir: Path, client=None, environment=None, task=None) -> vLLMAgent:
    """Serve the checkpoint's model through Forge's vLLM adapter."""
    checkpoint = PolicyCheckpoint.load(checkpoint_dir)
    if environment is None:
        return vLLMAgent(model=checkpoint.model_path, client=client)
    from forge.runtime.agents.factory import make_agent

    return make_agent(
        f"vllm:{checkpoint.model_path}",
        client=client,
        environment=environment,
        task=task,
    )
