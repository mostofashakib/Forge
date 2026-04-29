from forge.extraction.schemas import (
    CompilerInput, EntityDef, FieldDef, ActionDef, ActionParam,
    PolicyRule, SuccessCondition, TaskTemplate,
)


def _counter_input() -> CompilerInput:
    return CompilerInput(
        project_name="counter_env",
        domain="counter",
        entities=[
            EntityDef(
                name="counter",
                fields=[
                    FieldDef(name="id", type="string"),
                    FieldDef(name="value", type="integer", default=0),
                ],
            )
        ],
        actions=[
            ActionDef(
                name="increment",
                params=[ActionParam(name="counter_id", type="string")],
                mutates=["counter.value"],
            )
        ],
        tasks=[
            TaskTemplate(
                name="reach_target",
                description="Reach the target value",
                success_conditions=[
                    SuccessCondition(type="state_check", expression="counter.value >= target")
                ],
            )
        ],
    )


def test_compiler_input_roundtrips_json():
    ci = _counter_input()
    restored = CompilerInput.model_validate_json(ci.model_dump_json())
    assert restored == ci


def test_entity_def_primary_key_default():
    entity = EntityDef(name="ticket", fields=[])
    assert entity.primary_key == "id"


def test_field_def_enum_values():
    field = FieldDef(name="status", type="enum", values=["open", "closed"])
    assert field.values == ["open", "closed"]


def test_action_def_mutates_defaults_empty():
    action = ActionDef(name="noop", params=[])
    assert action.mutates == []


def test_task_template_failure_conditions_defaults_empty():
    task = TaskTemplate(name="t", description="d", success_conditions=[])
    assert task.failure_conditions == []
