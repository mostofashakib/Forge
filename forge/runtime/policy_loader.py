"""Load a trained checkpoint into the runtime agent that serves it."""
from __future__ import annotations

from pathlib import Path

from forge.runtime.agents.vllm_agent import vLLMAgent
from forge.training.checkpoint import PolicyCheckpoint


def load_policy_agent(checkpoint_dir: Path, client=None) -> vLLMAgent:
    """Serve the checkpoint's model through Forge's vLLM adapter."""
    checkpoint = PolicyCheckpoint.load(checkpoint_dir)
    return vLLMAgent(model=checkpoint.model_path, client=client)
