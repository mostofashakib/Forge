from forge.extraction.task_generator import TaskGenerator, _TaskExtractionResult
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.schemas import TaskTemplate, SuccessCondition


def _mock_tasks() -> _TaskExtractionResult:
    return _TaskExtractionResult(tasks=[
        TaskTemplate(
            name="resolve_ticket",
            description="Resolve a support ticket",
            success_conditions=[
                SuccessCondition(type="state_check", expression="ticket.status == 'solved'")
            ],
        )
    ])


def test_task_generator_returns_task_list():
    client = MockLLMClient({"_TaskExtractionResult": _mock_tasks()})
    gen = TaskGenerator(client)
    tasks = gen.extract("Resolve tickets", entities=[], actions=[], policies=[])
    assert len(tasks) == 1
    assert tasks[0].name == "resolve_ticket"
