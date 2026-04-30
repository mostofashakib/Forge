import pytest
from forge.extraction.pii_redactor import PIIRedactor
from forge.extraction.schemas import CompilerInput, EntityDef, FieldDef, ActionDef, TaskTemplate, SuccessCondition


def test_email_redacted():
    r = PIIRedactor()
    assert r.redact("Contact alice@example.com for help") == "Contact [REDACTED_EMAIL] for help"


def test_phone_redacted():
    r = PIIRedactor()
    result = r.redact("Call 555-867-5309 now")
    assert "[REDACTED_PHONE]" in result
    assert "555-867-5309" not in result


def test_ssn_redacted():
    r = PIIRedactor()
    assert r.redact("SSN: 123-45-6789") == "SSN: [REDACTED_SSN]"


def test_non_pii_unchanged():
    r = PIIRedactor()
    text = "The ticket is open and pending review"
    assert r.redact(text) == text


def test_multiple_patterns_in_one_string():
    r = PIIRedactor()
    text = "Email alice@example.com or call 555-867-5309"
    result = r.redact(text)
    assert "[REDACTED_EMAIL]" in result
    assert "[REDACTED_PHONE]" in result
    assert "alice@example.com" not in result


def test_compiler_input_redaction():
    r = PIIRedactor()
    ci = CompilerInput(
        project_name="test",
        domain="Test domain with alice@example.com",
        entities=[
            EntityDef(
                name="customer",
                fields=[FieldDef(name="email", type="string", default="bob@test.com")],
            )
        ],
        actions=[ActionDef(name="reply", params=[])],
        tasks=[
            TaskTemplate(
                name="task1",
                description="Contact 555-123-4567 for support",
                success_conditions=[SuccessCondition(type="state_check", expression="True")],
            )
        ],
    )
    result = r.redact_compiler_input(ci)
    assert "alice@example.com" not in result.domain
    assert "bob@test.com" not in str(result.entities[0].fields[0].default)
    assert "555-123-4567" not in result.tasks[0].description
