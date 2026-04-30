from forge.extraction.policy_parser import PolicyParser
from forge.extraction.prompts import PolicyExtractionResult
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.schemas import PolicyRule


def _mock_policies() -> PolicyExtractionResult:
    return PolicyExtractionResult(policies=[
        PolicyRule(
            id="no_close_without_reply",
            condition="ticket.status == 'open'",
            forbidden_actions=["close_ticket_without_comment"],
        )
    ])


def test_policy_parser_returns_policy_list():
    client = MockLLMClient({"PolicyExtractionResult": _mock_policies()})
    parser = PolicyParser(client)
    policies = parser.extract("Agents must reply before closing", entities=[], actions=[])
    assert len(policies) == 1
    assert policies[0].id == "no_close_without_reply"
