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


def test_empty_llm_result_yields_no_policies():
    # False-positive guard: a domain with no stated rules must produce no
    # policies rather than a fabricated one.
    client = MockLLMClient({"PolicyExtractionResult": PolicyExtractionResult(policies=[])})
    assert PolicyParser(client).extract("anything goes", entities=[], actions=[]) == []
