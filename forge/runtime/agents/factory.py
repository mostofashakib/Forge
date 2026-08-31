from forge.runtime.agents.random_agent import RandomAgent
from forge.runtime.agents.scripted_agent import ScriptedAgent
from forge.runtime.errors import AgentError
from forge.contracts import normalize_task


def make_agent(agent_id: str, *, environment=None, task=None, task_id=None, **kwargs):
    """Construct an agent, optionally binding an Environment's model context.

    LLM adapters receive the environment's prompt template, selected task, and
    complete tool schemas automatically. Non-LLM policies remain environment
    agnostic and keep their existing construction behavior.
    """
    if agent_id == "random":
        return RandomAgent()
    if agent_id.startswith("scripted:"):
        path = agent_id[len("scripted:"):]
        return ScriptedAgent(path)
    if agent_id.startswith("anthropic:"):
        from forge.runtime.agents.anthropic_agent import AnthropicAgent
        model = agent_id[len("anthropic:"):]
        return AnthropicAgent(model=model, **_bound_kwargs(environment, task, task_id, kwargs))
    if agent_id.startswith("openai:"):
        from forge.runtime.agents.openai_agent import OpenAIAgent
        model = agent_id[len("openai:"):]
        return OpenAIAgent(model=model, **_bound_kwargs(environment, task, task_id, kwargs))
    if agent_id.startswith("vllm:"):
        from forge.runtime.agents.vllm_agent import vLLMAgent
        model = agent_id[len("vllm:"):]
        return vLLMAgent(model=model, **_bound_kwargs(environment, task, task_id, kwargs))
    raise AgentError(f"Unknown agent_id: {agent_id!r}")


def _bound_kwargs(environment, task, task_id, kwargs: dict) -> dict:
    bound = dict(kwargs)
    if environment is None:
        if task is not None:
            bound.setdefault("task", normalize_task(task))
        return bound

    bound.setdefault("prompt_template", environment.prompt)
    if environment.tools is not None:
        bound.setdefault("tool_specs", list(environment.tools.tools()))

    selected = task
    if selected is None:
        selected = getattr(environment, "current_task", None)
    if selected is None and task_id is not None:
        selected = environment.task_source.get(str(task_id))
    if selected is None:
        available = environment.task_source.tasks()
        selected = available[0] if available else None
    if selected is not None:
        bound.setdefault("task", normalize_task(selected))
    return bound
