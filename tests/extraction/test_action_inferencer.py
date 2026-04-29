from forge.extraction.action_inferencer import ActionInferencer, _ActionExtractionResult
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.schemas import ActionDef, ActionParam, EntityDef


def _mock_actions() -> _ActionExtractionResult:
    return _ActionExtractionResult(actions=[
        ActionDef(name="close_ticket", params=[
            ActionParam(name="ticket_id", type="string")
        ], mutates=["ticket.status"])
    ])


def test_action_inferencer_returns_action_list():
    client = MockLLMClient({"_ActionExtractionResult": _mock_actions()})
    inferencer = ActionInferencer(client)
    actions = inferencer.extract("A ticketing system", entities=[])
    assert len(actions) == 1
    assert actions[0].name == "close_ticket"
