from forge.extraction.task_generator import TaskGenerator
from forge.extraction.prompts import TaskExtractionResult
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.schemas import TaskTemplate, SuccessCondition


def _mock_tasks() -> TaskExtractionResult:
    return TaskExtractionResult(tasks=[
        TaskTemplate(
            name="resolve_ticket",
            description="Resolve a support ticket",
            success_conditions=[
                SuccessCondition(type="state_check", expression="ticket.status == 'solved'")
            ],
        )
    ])


def test_task_generator_returns_task_list():
    client = MockLLMClient({"TaskExtractionResult": _mock_tasks()})
    gen = TaskGenerator(client)
    tasks = gen.extract("Resolve tickets", entities=[], actions=[], policies=[])
    assert len(tasks) == 1
    assert tasks[0].name == "resolve_ticket"


def test_empty_llm_result_yields_no_tasks():
    # False-positive guard: no extracted tasks must yield an empty list, never a
    # fabricated placeholder task.
    client = MockLLMClient({"TaskExtractionResult": TaskExtractionResult(tasks=[])})
    assert TaskGenerator(client).extract("idle", entities=[], actions=[], policies=[]) == []
