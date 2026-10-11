"""The LLM judge flags harm, and any failure of the judge blocks the row."""
from pydantic import BaseModel

from forge.safety.classifier import LLMSafetyClassifier, SafetyVerdict


class FakeClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.seen = []

    def extract(self, system, user, schema):
        self.seen.append((system, user, schema))
        if self.error:
            raise self.error
        return self.result


def test_a_harmful_verdict_becomes_findings():
    client = FakeClient(SafetyVerdict(harmful=True, categories=["weapons"], reason="build guide"))
    findings = LLMSafetyClassifier(client).classify("text")
    assert [(f.rule, f.detail) for f in findings] == [("llm:weapons", "build guide")]
    assert client.seen[0][2] is SafetyVerdict
    assert "text" in client.seen[0][1]


def test_a_harmful_verdict_without_categories_still_blocks():
    findings = LLMSafetyClassifier(FakeClient(SafetyVerdict(harmful=True, categories=[], reason="r"))).classify("t")
    assert [f.rule for f in findings] == ["llm:harmful"]


def test_a_clean_verdict_has_no_findings():
    client = FakeClient(SafetyVerdict(harmful=False, categories=[], reason="fine"))
    assert LLMSafetyClassifier(client).classify("text") == []


def test_a_judge_error_blocks_the_row():
    findings = LLMSafetyClassifier(FakeClient(error=TimeoutError("slow"))).classify("text")
    assert [f.rule for f in findings] == ["classifier_error"]
    assert "slow" in findings[0].detail


def test_a_wrong_schema_from_the_judge_blocks_the_row():
    class Other(BaseModel):
        x: int = 1

    findings = LLMSafetyClassifier(FakeClient(Other())).classify("text")
    assert [f.rule for f in findings] == ["classifier_error"]
