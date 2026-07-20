from forge.extraction.action_inferencer import ActionInferencer
from forge.extraction.prompts import ActionExtractionResult
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.schemas import ActionDef, ActionParam, EntityDef


def _mock_actions() -> ActionExtractionResult:
    return ActionExtractionResult(actions=[
        ActionDef(name="close_ticket", params=[
            ActionParam(name="ticket_id", type="string")
        ], mutates=["ticket.status"])
    ])


def test_action_inferencer_returns_action_list():
    client = MockLLMClient({"ActionExtractionResult": _mock_actions()})
    inferencer = ActionInferencer(client)
    actions = inferencer.extract("A ticketing system", entities=[])
    assert len(actions) == 1
    assert actions[0].name == "close_ticket"


def test_empty_llm_result_yields_no_actions():
    # False-positive guard: when the model finds no actions, the inferencer must
    # return an empty list — never fabricate a phantom action.
    client = MockLLMClient({"ActionExtractionResult": ActionExtractionResult(actions=[])})
    assert ActionInferencer(client).extract("nothing actionable", entities=[]) == []
