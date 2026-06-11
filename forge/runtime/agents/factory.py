from forge.runtime.agents.random_agent import RandomAgent
from forge.runtime.agents.scripted_agent import ScriptedAgent
from forge.runtime.errors import AgentError


def make_agent(agent_id: str, **kwargs):
    if agent_id == "random":
        return RandomAgent()
    if agent_id.startswith("scripted:"):
        path = agent_id[len("scripted:"):]
        return ScriptedAgent(path)
    if agent_id.startswith("anthropic:"):
        from forge.runtime.agents.anthropic_agent import AnthropicAgent
        model = agent_id[len("anthropic:"):]
        return AnthropicAgent(model=model, **kwargs)
    if agent_id.startswith("openai:"):
        from forge.runtime.agents.openai_agent import OpenAIAgent
        model = agent_id[len("openai:"):]
        return OpenAIAgent(model=model, **kwargs)
    if agent_id.startswith("vllm:"):
        from forge.runtime.agents.vllm_agent import vLLMAgent
        model = agent_id[len("vllm:"):]
        return vLLMAgent(model=model, **kwargs)
    raise AgentError(f"Unknown agent_id: {agent_id!r}")
