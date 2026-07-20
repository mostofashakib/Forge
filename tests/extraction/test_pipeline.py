from forge.extraction.pipeline import ExtractionPipeline
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.prompts import (
    EntityExtractionResult,
    ActionExtractionResult,
    PolicyExtractionResult,
    TaskExtractionResult,
)
from forge.extraction.schemas import (
    EntityDef, FieldDef, ActionDef, ActionParam,
    PolicyRule, TaskTemplate, SuccessCondition,
)


def _make_mock_client() -> MockLLMClient:
    return MockLLMClient({
        "EntityExtractionResult": EntityExtractionResult(entities=[
            EntityDef(name="counter", fields=[
                FieldDef(name="id", type="string"),
                FieldDef(name="value", type="integer", default=0),
            ])
        ]),
        "ActionExtractionResult": ActionExtractionResult(actions=[
            ActionDef(name="increment", params=[
                ActionParam(name="counter_id", type="string")
            ], mutates=["counter.value"])
        ]),
        "PolicyExtractionResult": PolicyExtractionResult(policies=[]),
        "TaskExtractionResult": TaskExtractionResult(tasks=[
            TaskTemplate(
                name="reach_target",
                description="Reach target value",
                success_conditions=[
                    SuccessCondition(type="state_check", expression="counter.value >= target")
                ],
            )
        ]),
    })


def test_pipeline_returns_compiler_input():
    from forge.extraction.schemas import CompilerInput
    pipeline = ExtractionPipeline(_make_mock_client())
    result = pipeline.run(
        prompt="A counter environment",
        project_name="counter_env",
        domain="counter",
    )
    assert isinstance(result, CompilerInput)
    assert result.project_name == "counter_env"
    assert len(result.entities) == 1
    assert len(result.actions) == 1
    assert len(result.tasks) == 1


def test_pipeline_with_empty_extractions_yields_empty_input():
    # False-positive guard: when every stage returns nothing, the assembled
    # CompilerInput must be empty across the board, not partially fabricated.
    from forge.extraction.schemas import CompilerInput
    client = MockLLMClient({
        "EntityExtractionResult": EntityExtractionResult(entities=[]),
        "ActionExtractionResult": ActionExtractionResult(actions=[]),
        "PolicyExtractionResult": PolicyExtractionResult(policies=[]),
        "TaskExtractionResult": TaskExtractionResult(tasks=[]),
    })
    result = ExtractionPipeline(client).run(prompt="void", project_name="p", domain="d")
    assert isinstance(result, CompilerInput)
    assert result.entities == []
    assert result.actions == []
    assert result.tasks == []


def test_pipeline_passes_entities_to_action_inferencer():
    captured = {}

    class CapturingClient:
        def extract(self, system, user, schema):
            captured[schema.__name__] = user
            return _make_mock_client().extract(system, user, schema)

    ExtractionPipeline(CapturingClient()).run("desc", "proj", "dom")
    assert "ActionExtractionResult" in captured
    assert "counter" in captured["ActionExtractionResult"]
