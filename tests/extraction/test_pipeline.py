from forge.extraction.pipeline import ExtractionPipeline
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.entity_extractor import _EntityExtractionResult
from forge.extraction.action_inferencer import _ActionExtractionResult
from forge.extraction.policy_parser import _PolicyExtractionResult
from forge.extraction.task_generator import _TaskExtractionResult
from forge.extraction.schemas import (
    EntityDef, FieldDef, ActionDef, ActionParam,
    PolicyRule, TaskTemplate, SuccessCondition,
)


def _make_mock_client() -> MockLLMClient:
    return MockLLMClient({
        "_EntityExtractionResult": _EntityExtractionResult(entities=[
            EntityDef(name="counter", fields=[
                FieldDef(name="id", type="string"),
                FieldDef(name="value", type="integer", default=0),
            ])
        ]),
        "_ActionExtractionResult": _ActionExtractionResult(actions=[
            ActionDef(name="increment", params=[
                ActionParam(name="counter_id", type="string")
            ], mutates=["counter.value"])
        ]),
        "_PolicyExtractionResult": _PolicyExtractionResult(policies=[]),
        "_TaskExtractionResult": _TaskExtractionResult(tasks=[
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


def test_pipeline_passes_entities_to_action_inferencer():
    captured = {}

    class CapturingClient:
        def extract(self, system, user, schema):
            captured[schema.__name__] = user
            return _make_mock_client().extract(system, user, schema)

    ExtractionPipeline(CapturingClient()).run("desc", "proj", "dom")
    assert "_ActionExtractionResult" in captured
    assert "counter" in captured["_ActionExtractionResult"]
